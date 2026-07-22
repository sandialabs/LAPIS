//===- Einsum.cpp -------------------------------------------------------===//

#include "lapis/Transform/Einsum.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace mlir::lapis {

//===----------------------------------------------------------------------===//
// Einsum expression
//===----------------------------------------------------------------------===//

EinsumExpression::EinsumExpression(llvm::SmallVector<TensorAccess, 4> operands,
                                   TensorAccess result,
                                   llvm::DenseMap<IndexId, int64_t> extents,
                                   llvm::SmallVector<IndexId, 4> indexOrder)
    : operands(std::move(operands)), result(std::move(result)),
      extents(std::move(extents)), indexOrder(std::move(indexOrder)) {}

llvm::Expected<EinsumExpression>
EinsumExpression::create(llvm::ArrayRef<TensorAccess> operands,
                         TensorAccess result,
                         llvm::ArrayRef<IndexExtent> extents) {
  if (operands.empty())
    return llvm::createStringError(
        "an einsum expression must have at least one operand");

  llvm::DenseSet<IndexId> operandIndices;
  llvm::SmallVector<IndexId, 4> indexOrder;
  for (const TensorAccess &operand : operands) {
    for (IndexId index : operand.indices) {
      if (operandIndices.insert(index).second)
        indexOrder.push_back(index);
    }
  }

  llvm::DenseSet<IndexId> resultIndices;
  for (IndexId index : result.indices) {
    if (!resultIndices.insert(index).second)
      return llvm::createStringError("result index " + llvm::Twine(index) +
                                     " appears more than once");
    if (!operandIndices.contains(index))
      return llvm::createStringError("result index " + llvm::Twine(index) +
                                     " does not appear in an operand");
  }

  llvm::DenseMap<IndexId, int64_t> extentMap;
  for (const IndexExtent &indexExtent : extents) {
    if (indexExtent.extent <= 0)
      return llvm::createStringError("extent for index " +
                                     llvm::Twine(indexExtent.index) +
                                     " must be positive");

    auto [iterator, inserted] =
        extentMap.insert({indexExtent.index, indexExtent.extent});
    if (!inserted && iterator->second != indexExtent.extent)
      return llvm::createStringError("conflicting extents for index " +
                                     llvm::Twine(indexExtent.index));
  }

  for (IndexId index : indexOrder) {
    if (!extentMap.contains(index))
      return llvm::createStringError("missing extent for index " +
                                     llvm::Twine(index));
  }

  for (const auto &[index, extent] : extentMap) {
    (void)extent;
    if (!operandIndices.contains(index))
      return llvm::createStringError("extent provided for unused index " +
                                     llvm::Twine(index));
  }

  return EinsumExpression(llvm::SmallVector<TensorAccess, 4>(operands),
                          std::move(result), std::move(extentMap),
                          std::move(indexOrder));
}

std::optional<int64_t> EinsumExpression::getExtent(IndexId index) const {
  auto iterator = extents.find(index);
  if (iterator == extents.end())
    return std::nullopt;
  return iterator->second;
}

llvm::SmallVector<IndexId, 4> EinsumExpression::getReductionIndices() const {
  llvm::DenseSet<IndexId> resultIndices(result.indices.begin(),
                                        result.indices.end());
  llvm::SmallVector<IndexId, 4> reductionIndices;
  for (IndexId index : indexOrder) {
    if (!resultIndices.contains(index))
      reductionIndices.push_back(index);
  }
  return reductionIndices;
}

} // namespace mlir::lapis

namespace mlir::lapis {

//===----------------------------------------------------------------------===//
// Expression composition
//===----------------------------------------------------------------------===//

namespace {

bool isReservedDenseMapKey(IndexId index) {
  return index == llvm::DenseMapInfo<IndexId>::getEmptyKey() ||
         index == llvm::DenseMapInfo<IndexId>::getTombstoneKey();
}

llvm::Expected<IndexId>
findFreshIndex(const llvm::DenseSet<IndexId> &unavailableIndices) {
  constexpr uint64_t maxIndex = std::numeric_limits<IndexId>::max();
  for (uint64_t candidate = 0; candidate <= maxIndex; ++candidate) {
    IndexId index = static_cast<IndexId>(candidate);
    if (isReservedDenseMapKey(index))
      continue;
    if (!unavailableIndices.contains(index))
      return index;
  }
  return llvm::createStringError("no index identifier is available");
}

TensorAccess remapAccess(const TensorAccess &access,
                         const llvm::DenseMap<IndexId, IndexId> &indexMap) {
  TensorAccess remapped;
  remapped.indices.reserve(access.indices.size());
  for (IndexId index : access.indices)
    remapped.indices.push_back(indexMap.lookup(index));
  return remapped;
}

} // namespace

EinsumComposition::EinsumComposition(
    EinsumExpression expression,
    llvm::DenseMap<IndexId, IndexId> producerIndexMap)
    : expression(std::move(expression)),
      producerIndexMap(std::move(producerIndexMap)) {}

TensorAccess
EinsumComposition::remapProducerAccess(const TensorAccess &access) const {
  return remapAccess(access, producerIndexMap);
}

llvm::Expected<EinsumComposition>
composeEinsumExpressionsWithProvenance(const EinsumExpression &producer,
                                       const EinsumExpression &consumer,
                                       std::size_t consumerOperand) {
  if (consumerOperand >= consumer.getOperands().size())
    return llvm::createStringError(
        "consumer operand " + llvm::Twine(consumerOperand) +
        " is out of range for an expression with " +
        llvm::Twine(consumer.getOperands().size()) + " operands");

  const TensorAccess &producerResult = producer.getResult();
  const TensorAccess &replacedOperand = consumer.getOperands()[consumerOperand];
  if (producerResult.indices.size() != replacedOperand.indices.size())
    return llvm::createStringError("producer result rank " +
                                   llvm::Twine(producerResult.indices.size()) +
                                   " does not match consumer operand rank " +
                                   llvm::Twine(replacedOperand.indices.size()));

  llvm::DenseMap<IndexId, IndexId> producerIndexMap;
  for (std::size_t axis = 0; axis < producerResult.indices.size(); ++axis) {
    IndexId producerIndex = producerResult.indices[axis];
    IndexId consumerIndex = replacedOperand.indices[axis];
    if (producer.getExtent(producerIndex) != consumer.getExtent(consumerIndex))
      return llvm::createStringError(
          "producer result extent does not match consumer operand extent at "
          "axis " +
          llvm::Twine(axis));
    producerIndexMap.insert({producerIndex, consumerIndex});
  }

  llvm::DenseSet<IndexId> consumerIndices(consumer.getIndexOrder().begin(),
                                          consumer.getIndexOrder().end());
  llvm::DenseSet<IndexId> unavailableIndices = consumerIndices;
  unavailableIndices.insert(producer.getIndexOrder().begin(),
                            producer.getIndexOrder().end());

  for (IndexId producerIndex : producer.getReductionIndices()) {
    IndexId composedIndex = producerIndex;
    if (consumerIndices.contains(producerIndex)) {
      auto freshIndex = findFreshIndex(unavailableIndices);
      if (!freshIndex)
        return freshIndex.takeError();
      composedIndex = *freshIndex;
    }
    producerIndexMap.insert({producerIndex, composedIndex});
    unavailableIndices.insert(composedIndex);
  }

  llvm::SmallVector<TensorAccess, 4> composedOperands;
  composedOperands.reserve(consumer.getOperands().size() - 1 +
                           producer.getOperands().size());
  for (std::size_t operand = 0; operand < consumer.getOperands().size();
       ++operand) {
    if (operand != consumerOperand) {
      composedOperands.push_back(consumer.getOperands()[operand]);
      continue;
    }
    for (const TensorAccess &producerOperand : producer.getOperands())
      composedOperands.push_back(
          remapAccess(producerOperand, producerIndexMap));
  }

  llvm::SmallVector<IndexExtent, 8> composedExtents;
  composedExtents.reserve(consumer.getIndexOrder().size() +
                          producer.getIndexOrder().size());
  for (IndexId consumerIndex : consumer.getIndexOrder())
    composedExtents.push_back(
        {consumerIndex, *consumer.getExtent(consumerIndex)});
  for (IndexId producerIndex : producer.getIndexOrder())
    composedExtents.push_back({producerIndexMap.lookup(producerIndex),
                               *producer.getExtent(producerIndex)});

  auto expression = EinsumExpression::create(
      composedOperands, consumer.getResult(), composedExtents);
  if (!expression)
    return expression.takeError();
  return EinsumComposition(std::move(*expression), std::move(producerIndexMap));
}

llvm::Expected<EinsumExpression>
composeEinsumExpressions(const EinsumExpression &producer,
                         const EinsumExpression &consumer,
                         std::size_t consumerOperand) {
  auto composition = composeEinsumExpressionsWithProvenance(producer, consumer,
                                                            consumerOperand);
  if (!composition)
    return composition.takeError();
  return composition->takeExpression();
}

} // namespace mlir::lapis

namespace mlir::lapis {

//===----------------------------------------------------------------------===//
// linalg.generic extraction
//===----------------------------------------------------------------------===//

namespace {

llvm::Error unsupported(const llvm::Twine &message) {
  return llvm::createStringError("unsupported linalg.generic: " + message);
}

llvm::Expected<TensorAccess>
extractTensorAccess(Value value, AffineMap indexingMap,
                    llvm::SmallVectorImpl<IndexExtent> &extents,
                    const llvm::Twine &description) {
  auto tensorType = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorType)
    return unsupported(description + " must be a ranked tensor");
  if (!tensorType.hasStaticShape())
    return unsupported(description + " must have a static shape");
  if (indexingMap.getNumSymbols() != 0)
    return unsupported(description + " indexing map must not contain symbols");
  if (indexingMap.getNumResults() !=
      static_cast<unsigned>(tensorType.getRank()))
    return unsupported(description + " indexing-map rank does not match its "
                                     "tensor rank");

  TensorAccess access;
  access.indices.reserve(indexingMap.getNumResults());
  for (const auto &[axis, result] : llvm::enumerate(indexingMap.getResults())) {
    auto dimension = dyn_cast<AffineDimExpr>(result);
    if (!dimension)
      return unsupported(description +
                         " indexing map must contain only dimensions");

    int64_t extent = tensorType.getDimSize(axis);
    if (extent <= 0)
      return unsupported(description + " dimension " + llvm::Twine(axis) +
                         " must have a positive extent");

    IndexId index = dimension.getPosition();
    access.indices.push_back(index);
    extents.push_back(IndexExtent{index, extent});
  }
  return access;
}

llvm::Error verifySumProductBody(linalg::GenericOp generic,
                                 unsigned inputCount) {
  Block &body = generic.getRegion().front();
  if (body.getNumArguments() != inputCount + 1)
    return unsupported("body argument count does not match two inputs and one "
                       "output");

  arith::MulFOp multiplication;
  arith::AddFOp addition;
  for (Operation &operation : body.without_terminator()) {
    if (auto multiply = dyn_cast<arith::MulFOp>(operation)) {
      if (multiplication)
        return unsupported("body must contain exactly one arith.mulf");
      multiplication = multiply;
      continue;
    }
    if (auto add = dyn_cast<arith::AddFOp>(operation)) {
      if (addition)
        return unsupported("body must contain exactly one arith.addf");
      addition = add;
      continue;
    }
    return unsupported("body contains an operation other than arith.mulf, "
                       "arith.addf, and linalg.yield");
  }

  if (!multiplication)
    return unsupported("body must contain exactly one arith.mulf");
  if (!addition)
    return unsupported("body must contain exactly one arith.addf");

  llvm::DenseSet<unsigned> multipliedInputs;
  for (Value operand : multiplication.getOperands()) {
    auto blockArgument = dyn_cast<BlockArgument>(operand);
    if (!blockArgument || blockArgument.getOwner() != &body ||
        blockArgument.getArgNumber() >= inputCount)
      return unsupported("arith.mulf operands must be input block arguments");
    multipliedInputs.insert(blockArgument.getArgNumber());
  }
  if (multipliedInputs.size() != inputCount)
    return unsupported("arith.mulf must use each input exactly once");

  Value accumulator = body.getArgument(inputCount);
  Value product = multiplication.getResult();
  bool accumulatorPlusProduct =
      (addition.getLhs() == accumulator && addition.getRhs() == product) ||
      (addition.getRhs() == accumulator && addition.getLhs() == product);
  if (!accumulatorPlusProduct)
    return unsupported(
        "arith.addf must combine the output accumulator and product");

  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yield || yield.getNumOperands() != 1 ||
      yield.getOperand(0) != addition.getResult())
    return unsupported("linalg.yield must return the sum");

  return llvm::Error::success();
}

llvm::Error verifyIteratorTypes(linalg::GenericOp generic,
                                const TensorAccess &result) {
  auto iteratorTypes = generic.getIteratorTypesArray();
  llvm::DenseSet<IndexId> resultIndices(result.indices.begin(),
                                        result.indices.end());
  for (const auto &[index, iteratorType] : llvm::enumerate(iteratorTypes)) {
    bool isResultIndex = resultIndices.contains(index);
    utils::IteratorType expected = isResultIndex
                                       ? utils::IteratorType::parallel
                                       : utils::IteratorType::reduction;
    if (iteratorType != expected)
      return unsupported("iterator " + llvm::Twine(index) + " must be " +
                         (isResultIndex ? "parallel" : "reduction"));
  }
  return llvm::Error::success();
}

bool isKnownZero(Value value) {
  if (matchPattern(value, m_AnyZeroFloat()))
    return true;
  auto fill = value.getDefiningOp<linalg::FillOp>();
  return fill && matchPattern(fill.getInputs().front(), m_AnyZeroFloat());
}

} // namespace

ExtractedEinsumExpression::ExtractedEinsumExpression(
    linalg::GenericOp source, EinsumExpression expression,
    llvm::ArrayRef<Value> operands, Value init, Value result)
    : source(source), expression(std::move(expression)), operands(operands),
      init(init), result(result) {}

llvm::Expected<ExtractedEinsumExpression>
extractEinsumExpression(linalg::GenericOp generic) {
  auto inputOperands = generic.getDpsInputOperands();
  if (inputOperands.size() != 2)
    return unsupported("expected exactly two input operands, but found " +
                       llvm::Twine(inputOperands.size()));
  if (generic.getDpsInits().size() != 1 || generic.getNumResults() != 1)
    return unsupported("expected exactly one tensor output and result");

  auto indexingMaps = generic.getIndexingMapsArray();
  if (indexingMaps.size() != inputOperands.size() + 1)
    return unsupported("indexing-map count does not match operand count");

  auto iteratorTypes = generic.getIteratorTypesArray();
  for (AffineMap indexingMap : indexingMaps) {
    if (indexingMap.getNumDims() != iteratorTypes.size())
      return unsupported(
          "indexing-map input count does not match iterator count");
  }

  llvm::SmallVector<TensorAccess, 4> operandAccesses;
  llvm::SmallVector<Value, 4> operands;
  llvm::SmallVector<IndexExtent, 8> extents;
  operandAccesses.reserve(inputOperands.size());
  operands.reserve(inputOperands.size());
  for (const auto &[operandNumber, operand] : llvm::enumerate(inputOperands)) {
    auto access =
        extractTensorAccess(operand->get(), indexingMaps[operandNumber],
                            extents, "input " + llvm::Twine(operandNumber));
    if (!access)
      return access.takeError();
    operandAccesses.push_back(std::move(*access));
    operands.push_back(operand->get());
  }

  Value output = generic.getDpsInits().front();
  auto resultAccess =
      extractTensorAccess(output, indexingMaps.back(), extents, "output");
  if (!resultAccess)
    return resultAccess.takeError();

  if (llvm::Error error = verifyIteratorTypes(generic, *resultAccess))
    return std::move(error);
  if (llvm::Error error = verifySumProductBody(generic, inputOperands.size()))
    return std::move(error);
  if (!isKnownZero(output))
    return unsupported(
        "output initializer must be a provable floating-point zero");

  auto expression = EinsumExpression::create(operandAccesses,
                                             std::move(*resultAccess), extents);
  if (!expression)
    return expression.takeError();

  return ExtractedEinsumExpression(generic, std::move(*expression), operands,
                                   output, generic.getResult(0));
}

} // namespace mlir::lapis

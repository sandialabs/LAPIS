//===- AlgebraicKernelFusion.cpp -----------------------------------------===//

#include "lapis/Transform/AlgebraicKernelFusion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstddef>
#include <limits>
#include <optional>
#include <utility>

namespace mlir::lapis {

//===----------------------------------------------------------------------===//
// Algebraic region discovery
//===----------------------------------------------------------------------===//

namespace {

struct PartialRegion {
  explicit PartialRegion(const ExtractedEinsumExpression &extracted)
      : expression(extracted.getExpression()),
        operands(extracted.getOperands().begin(),
                 extracted.getOperands().end()) {
    OperandSubset allOperands =
        (OperandSubset{1} << extracted.getOperands().size()) - 1;
    results.push_back(LinalgEinsumRegionOutput{
        extracted.getSource(), extracted.getResult(), allOperands,
        extracted.getExpression().getResult()});
  }

  EinsumExpression expression;
  llvm::SmallVector<Value, 4> operands;
  llvm::SmallVector<linalg::GenericOp, 4> operations;
  llvm::SmallVector<LinalgEinsumRegionOutput, 4> results;
};

llvm::DenseSet<Operation *>
getOperationSet(llvm::ArrayRef<linalg::GenericOp> operations) {
  llvm::DenseSet<Operation *> result;
  for (linalg::GenericOp operation : operations)
    result.insert(operation);
  return result;
}

llvm::Expected<OperandSubset>
remapConsumerSubset(OperandSubset subset, unsigned consumerOperand,
                    unsigned producerOperandCount,
                    unsigned consumerOperandCount) {
  unsigned composedOperandCount =
      consumerOperandCount - 1 + producerOperandCount;
  if (composedOperandCount > kMaxTrackedOperands)
    return llvm::createStringError("composed region has more than " +
                                   llvm::Twine(kMaxTrackedOperands) +
                                   " operands");

  OperandSubset beforeMask =
      consumerOperand == 0 ? 0 : (OperandSubset{1} << consumerOperand) - 1;
  OperandSubset before = subset & beforeMask;
  bool containsProducerResult =
      (subset & (OperandSubset{1} << consumerOperand)) != 0;
  OperandSubset after = subset >> (consumerOperand + 1);

  OperandSubset producerMask;
  if (producerOperandCount == kMaxTrackedOperands)
    producerMask = std::numeric_limits<OperandSubset>::max();
  else
    producerMask = (OperandSubset{1} << producerOperandCount) - 1;

  return before |
         (containsProducerResult ? producerMask << consumerOperand : 0) |
         (after << (consumerOperand + producerOperandCount));
}

} // namespace

LinalgEinsumRegion::LinalgEinsumRegion(
    llvm::ArrayRef<linalg::GenericOp> operations, EinsumExpression expression,
    llvm::ArrayRef<Value> operands,
    llvm::ArrayRef<LinalgEinsumRegionOutput> outputs)
    : operations(operations), expression(std::move(expression)),
      operands(operands), outputs(outputs) {}

llvm::Expected<llvm::SmallVector<LinalgEinsumRegion, 4>>
discoverLinalgEinsumRegions(func::FuncOp function) {
  llvm::SmallVector<ExtractedEinsumExpression, 8> candidates;
  llvm::DenseMap<Operation *, unsigned> candidateIndices;

  // A region never crosses a block boundary, so nested control-flow regions
  // can be considered independently without changing composition legality.
  function.walk([&](linalg::GenericOp generic) {
    auto extracted = extractEinsumExpression(generic);
    if (!extracted) {
      // Unsupported generics are intentional region boundaries. The strict
      // extraction API remains available to callers that need the reason.
      llvm::consumeError(extracted.takeError());
      return;
    }

    candidateIndices.insert(
        {generic.getOperation(), static_cast<unsigned>(candidates.size())});
    candidates.push_back(std::move(*extracted));
  });

  auto getEligibleConsumer =
      [&](unsigned producerIndex) -> std::optional<unsigned> {
    const auto &producer = candidates[producerIndex];
    Value result = producer.getResult();
    std::optional<unsigned> consumerIndex;
    for (OpOperand &use : result.getUses()) {
      auto iterator = candidateIndices.find(use.getOwner());
      if (iterator == candidateIndices.end())
        continue;

      const auto &consumer = candidates[iterator->second];
      if (producer.getSource()->getBlock() != consumer.getSource()->getBlock())
        continue;
      if (!llvm::is_contained(consumer.getOperands(), result))
        continue;

      // One result occurrence in one consumer is the largest structure that
      // can be represented without duplicating a producer or introducing an
      // algebraic DAG.
      if (consumerIndex)
        return std::nullopt;
      consumerIndex = iterator->second;
    }
    return consumerIndex;
  };

  auto getEligibleProducer =
      [&](Value operand, unsigned consumerIndex) -> std::optional<unsigned> {
    auto generic = operand.getDefiningOp<linalg::GenericOp>();
    if (!generic)
      return std::nullopt;
    auto iterator = candidateIndices.find(generic.getOperation());
    if (iterator == candidateIndices.end())
      return std::nullopt;
    auto consumer = getEligibleConsumer(iterator->second);
    if (!consumer || *consumer != consumerIndex)
      return std::nullopt;
    return iterator->second;
  };

  auto buildRegion = [&](auto &&self,
                         unsigned rootIndex) -> llvm::Expected<PartialRegion> {
    const auto &root = candidates[rootIndex];
    PartialRegion region(root);

    for (Value originalOperand : root.getOperands()) {
      auto producerIndex = getEligibleProducer(originalOperand, rootIndex);
      if (!producerIndex)
        continue;

      auto producerRegion = self(self, *producerIndex);
      if (!producerRegion)
        return producerRegion.takeError();

      auto operandPosition = llvm::find(region.operands, originalOperand);
      if (operandPosition == region.operands.end())
        return llvm::createStringError(
            "internal error: producer result is absent from composed operands");
      std::size_t consumerOperand =
          static_cast<std::size_t>(operandPosition - region.operands.begin());

      auto composed = composeEinsumExpressionsWithProvenance(
          producerRegion->expression, region.expression, consumerOperand);
      if (!composed)
        return composed.takeError();

      for (LinalgEinsumRegionOutput &result : region.results) {
        auto remapped = remapConsumerSubset(
            result.operandSubset, static_cast<unsigned>(consumerOperand),
            static_cast<unsigned>(producerRegion->operands.size()),
            static_cast<unsigned>(region.operands.size()));
        if (!remapped)
          return remapped.takeError();
        result.operandSubset = *remapped;
      }
      for (LinalgEinsumRegionOutput &result : producerRegion->results) {
        result.operandSubset <<= consumerOperand;
        result.access = composed->remapProducerAccess(result.access);
        region.results.push_back(std::move(result));
      }
      region.expression = composed->takeExpression();

      operandPosition = region.operands.begin() + consumerOperand;
      operandPosition = region.operands.erase(operandPosition);
      region.operands.insert(operandPosition, producerRegion->operands.begin(),
                             producerRegion->operands.end());
      region.operations.append(producerRegion->operations);
    }

    region.operations.push_back(root.getSource());
    return region;
  };

  llvm::SmallVector<LinalgEinsumRegion, 4> regions;
  for (unsigned candidate = 0; candidate < candidates.size(); ++candidate) {
    if (getEligibleConsumer(candidate))
      continue;

    auto region = buildRegion(buildRegion, candidate);
    if (!region)
      return region.takeError();
    if (region->operations.size() < 2)
      continue;

    llvm::DenseSet<Operation *> regionOperations =
        getOperationSet(region->operations);

    llvm::SmallVector<LinalgEinsumRegionOutput, 2> outputs;
    for (linalg::GenericOp operation : region->operations) {
      auto result = llvm::find_if(region->results,
                                  [&](const LinalgEinsumRegionOutput &r) {
                                    return r.source == operation;
                                  });
      if (result == region->results.end())
        return llvm::createStringError(
            "internal error: region operation has no algebraic result");

      bool isRoot = operation == candidates[candidate].getSource();
      bool hasExternalUse =
          llvm::any_of(result->value.getUses(), [&](auto &use) {
            return !regionOperations.contains(use.getOwner());
          });
      if (isRoot || hasExternalUse) {
        outputs.push_back(LinalgEinsumRegionOutput{
            result->source, result->value, result->operandSubset,
            std::move(result->access)});
      }
    }

    regions.push_back(LinalgEinsumRegion(region->operations,
                                         std::move(region->expression),
                                         region->operands, outputs));
  }
  return regions;
}

//===----------------------------------------------------------------------===//
// Candidate evaluation
//===----------------------------------------------------------------------===//

namespace {

llvm::Expected<WorkCost> getSourceOperationWork(linalg::GenericOp operation) {
  auto extracted = extractEinsumExpression(operation);
  if (!extracted)
    return extracted.takeError();

  const EinsumExpression &expression = extracted->getExpression();
  auto costModel = ContractionCostModel::create(expression);
  if (!costModel)
    return costModel.takeError();
  return costModel->getContractionWork(OperandSubset{1}, OperandSubset{2});
}

llvm::Expected<WorkCost> checkedAdd(WorkCost accumulated, WorkCost next,
                                    std::size_t operationNumber) {
  if (accumulated > std::numeric_limits<WorkCost>::max() - next)
    return llvm::createStringError(
        "source region work overflows at operation " +
        llvm::Twine(operationNumber));
  return accumulated + next;
}

} // namespace

llvm::Expected<FusionCandidateEvaluation>
evaluateFusionCandidate(const LinalgEinsumRegion &region,
                        const ContractionPlan &optimizedPlan) {
  WorkCost originalWork = 0;
  for (const auto &[operationNumber, operation] :
       llvm::enumerate(region.getOperations())) {
    auto operationWork = getSourceOperationWork(operation);
    if (!operationWork)
      return operationWork.takeError();
    auto total = checkedAdd(originalWork, *operationWork, operationNumber);
    if (!total)
      return total.takeError();
    originalWork = *total;
  }

  auto costModel = ContractionCostModel::create(region.getExpression());
  if (!costModel)
    return costModel.takeError();
  auto optimizedCost = costModel->evaluatePlan(optimizedPlan);
  if (!optimizedCost)
    return optimizedCost.takeError();
  if (optimizedCost->totalWork > originalWork)
    return llvm::createStringError(
        "optimized contraction plan is more expensive than the source region");

  return FusionCandidateEvaluation{
      /*originalWork=*/originalWork,
      /*optimizedWork=*/optimizedCost->totalWork,
      /*originalKernelCount=*/
      static_cast<unsigned>(region.getOperations().size()),
      /*fusedKernelCount=*/1};
}

//===----------------------------------------------------------------------===//
// Contraction materialization
//===----------------------------------------------------------------------===//

namespace {

llvm::Error invalidMaterialization(const llvm::Twine &message) {
  return llvm::createStringError("cannot materialize contraction plan: " +
                                 message);
}

llvm::Expected<llvm::SmallVector<int64_t, 4>>
getShape(const EinsumExpression &expression, const TensorAccess &access) {
  llvm::SmallVector<int64_t, 4> shape;
  shape.reserve(access.indices.size());
  for (IndexId index : access.indices) {
    std::optional<int64_t> extent = expression.getExtent(index);
    if (!extent)
      return invalidMaterialization("index " + llvm::Twine(index) +
                                    " has no extent");
    shape.push_back(*extent);
  }
  return shape;
}

llvm::Error verifyTensor(Value value, const EinsumExpression &expression,
                         const TensorAccess &access, Type elementType,
                         const llvm::Twine &description) {
  auto tensorType = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorType)
    return invalidMaterialization(description + " is not a ranked tensor");
  if (tensorType.getElementType() != elementType)
    return invalidMaterialization(description +
                                  " has an inconsistent element type");

  auto expectedShape = getShape(expression, access);
  if (!expectedShape)
    return expectedShape.takeError();
  if (!llvm::equal(tensorType.getShape(), *expectedShape))
    return invalidMaterialization(description +
                                  " shape does not match its einsum access");
  return llvm::Error::success();
}

llvm::SmallVector<IndexId, 4>
getIterationOrder(const EinsumExpression &expression, const TensorAccess &lhs,
                  const TensorAccess &rhs) {
  llvm::DenseSet<IndexId> usedIndices(lhs.indices.begin(), lhs.indices.end());
  usedIndices.insert(rhs.indices.begin(), rhs.indices.end());

  llvm::SmallVector<IndexId, 4> iterationOrder;
  iterationOrder.reserve(usedIndices.size());
  for (IndexId index : expression.getIndexOrder()) {
    if (usedIndices.contains(index))
      iterationOrder.push_back(index);
  }
  return iterationOrder;
}

AffineMap makeIndexingMap(MLIRContext *context, const TensorAccess &access,
                          llvm::ArrayRef<IndexId> iterationOrder) {
  llvm::DenseMap<IndexId, unsigned> dimensions;
  for (const auto &[dimension, index] : llvm::enumerate(iterationOrder))
    dimensions.insert({index, static_cast<unsigned>(dimension)});

  llvm::SmallVector<AffineExpr, 4> results;
  results.reserve(access.indices.size());
  for (IndexId index : access.indices)
    results.push_back(getAffineDimExpr(dimensions.lookup(index), context));
  return AffineMap::get(iterationOrder.size(), /*symbolCount=*/0, results,
                        context);
}

llvm::SmallVector<utils::IteratorType, 4>
getIteratorTypes(llvm::ArrayRef<IndexId> iterationOrder,
                 const TensorAccess &result) {
  llvm::DenseSet<IndexId> resultIndices(result.indices.begin(),
                                        result.indices.end());
  llvm::SmallVector<utils::IteratorType, 4> iteratorTypes;
  iteratorTypes.reserve(iterationOrder.size());
  for (IndexId index : iterationOrder) {
    iteratorTypes.push_back(resultIndices.contains(index)
                                ? utils::IteratorType::parallel
                                : utils::IteratorType::reduction);
  }
  return iteratorTypes;
}

llvm::Error
verifyMaterializable(const LinalgEinsumRegion &region,
                     const ContractionPlan &plan, Type &elementType,
                     llvm::SmallVectorImpl<PlanValueId> &outputPlanValues) {
  const EinsumExpression &expression = region.getExpression();
  if (llvm::Error error = verifyContractionPlan(expression, plan))
    return error;
  if (plan.getSteps().empty() ||
      plan.getResult() != plan.getStepResult(plan.getSteps().size() - 1))
    return invalidMaterialization("plan result must be its final step");
  if (region.getOperands().size() != expression.getOperands().size())
    return invalidMaterialization(
        "region operand count does not match its einsum expression");
  if (region.getOperands().empty())
    return invalidMaterialization("region has no operands");
  if (region.getOutputs().empty())
    return invalidMaterialization("region has no externally visible outputs");

  auto firstType =
      dyn_cast<RankedTensorType>(region.getOperands().front().getType());
  if (!firstType || !isa<FloatType>(firstType.getElementType()))
    return invalidMaterialization(
        "region operands must be ranked tensors of floating-point values");
  elementType = firstType.getElementType();

  for (const auto &[operandNumber, operand] :
       llvm::enumerate(region.getOperands())) {
    if (llvm::Error error = verifyTensor(
            operand, expression, expression.getOperands()[operandNumber],
            elementType, "operand " + llvm::Twine(operandNumber)))
      return error;
  }

  llvm::DenseSet<Operation *> regionOperations =
      getOperationSet(region.getOperations());

  outputPlanValues.reserve(region.getOutputs().size());
  for (const auto &indexedOutput : llvm::enumerate(region.getOutputs())) {
    std::size_t outputNumber = indexedOutput.index();
    LinalgEinsumRegionOutput output = indexedOutput.value();
    if (!output.source ||
        !regionOperations.contains(output.source.getOperation()))
      return invalidMaterialization("output " + llvm::Twine(outputNumber) +
                                    " is not produced by the region");
    if (output.source.getResult(0) != output.value)
      return invalidMaterialization("output " + llvm::Twine(outputNumber) +
                                    " does not match its source result");

    auto planValue = findPlanValueForOperandSubset(plan, output.operandSubset);
    if (!planValue)
      return planValue.takeError();

    llvm::ArrayRef<IndexId> planAccess;
    if (*planValue < plan.getOperandCount()) {
      planAccess = expression.getOperands()[*planValue].indices;
    } else {
      planAccess =
          plan.getSteps()[*planValue - plan.getOperandCount()].resultIndices;
    }
    if (!llvm::equal(planAccess, output.access.indices))
      return invalidMaterialization(
          "output " + llvm::Twine(outputNumber) +
          " index order does not match its plan value");

    Value init = output.source.getDpsInits().front();
    if (llvm::Error error = verifyTensor(
            init, expression, output.access, elementType,
            "output " + llvm::Twine(outputNumber) + " initializer"))
      return error;
    if (init.getType() != output.value.getType())
      return invalidMaterialization("output " + llvm::Twine(outputNumber) +
                                    " initializer and result types differ");
    outputPlanValues.push_back(*planValue);
  }

  LinalgEinsumRegionOutput rootOutput = region.getOutputs().back();
  if (outputPlanValues.back() != plan.getResult() ||
      rootOutput.source != region.getRoot())
    return invalidMaterialization(
        "the final region output must be the root plan result");

  for (const ContractionStep &step : plan.getSteps()) {
    auto shape = getShape(expression, TensorAccess{step.resultIndices});
    if (!shape)
      return shape.takeError();
  }
  return llvm::Error::success();
}

} // namespace

llvm::Expected<llvm::SmallVector<Value, 2>> materializeContractionPlan(
    RewriterBase &rewriter, const LinalgEinsumRegion &region,
    const ContractionPlan &plan, std::optional<std::uint64_t> fusionGroup) {
  Type elementType;
  llvm::SmallVector<PlanValueId, 2> outputPlanValues;
  if (llvm::Error error =
          verifyMaterializable(region, plan, elementType, outputPlanValues))
    return std::move(error);

  const EinsumExpression &expression = region.getExpression();
  llvm::SmallVector<TensorAccess, 8> valueAccesses(
      expression.getOperands().begin(), expression.getOperands().end());
  llvm::SmallVector<Value, 8> values(region.getOperands().begin(),
                                     region.getOperands().end());
  valueAccesses.reserve(values.size() + plan.getSteps().size());
  values.reserve(values.size() + plan.getSteps().size());

  Location location = region.getRoot().getLoc();
  rewriter.setInsertionPoint(region.getRoot());
  Value zero;

  llvm::DenseMap<PlanValueId, unsigned> outputNumbers;
  for (const auto &[outputNumber, planValue] :
       llvm::enumerate(outputPlanValues))
    outputNumbers.insert({planValue, static_cast<unsigned>(outputNumber)});

  for (const auto &[stepNumber, step] : llvm::enumerate(plan.getSteps())) {
    const TensorAccess &lhsAccess = valueAccesses[step.lhs];
    const TensorAccess &rhsAccess = valueAccesses[step.rhs];
    TensorAccess resultAccess{step.resultIndices};

    auto resultShape = getShape(expression, resultAccess);
    if (!resultShape)
      llvm_unreachable("materialization shape was validated before mutation");

    PlanValueId stepResult = plan.getStepResult(stepNumber);
    auto output = outputNumbers.find(stepResult);
    RankedTensorType resultType;
    Value init;
    if (output != outputNumbers.end()) {
      LinalgEinsumRegionOutput regionOutput =
          region.getOutputs()[output->second];
      init = regionOutput.source.getDpsInits().front();
      resultType = cast<RankedTensorType>(init.getType());
    } else {
      resultType = RankedTensorType::get(*resultShape, elementType);
      auto empty = rewriter.create<tensor::EmptyOp>(
          location, resultType.getShape(), elementType);
      if (!zero)
        zero = rewriter.create<arith::ConstantOp>(
            location, rewriter.getZeroAttr(elementType));
      auto fill = rewriter.create<linalg::FillOp>(
          location, TypeRange{resultType}, ValueRange{zero},
          ValueRange{empty.getResult()});
      init = fill.getResult(0);
    }

    llvm::SmallVector<IndexId, 4> iterationOrder =
        getIterationOrder(expression, lhsAccess, rhsAccess);
    llvm::SmallVector<AffineMap, 3> indexingMaps{
        makeIndexingMap(rewriter.getContext(), lhsAccess, iterationOrder),
        makeIndexingMap(rewriter.getContext(), rhsAccess, iterationOrder),
        makeIndexingMap(rewriter.getContext(), resultAccess, iterationOrder)};
    llvm::SmallVector<utils::IteratorType, 4> iteratorTypes =
        getIteratorTypes(iterationOrder, resultAccess);

    auto contraction = rewriter.create<linalg::GenericOp>(
        location, TypeRange{resultType},
        ValueRange{values[step.lhs], values[step.rhs]}, ValueRange{init},
        indexingMaps, iteratorTypes,
        [](OpBuilder &builder, Location bodyLocation, ValueRange arguments) {
          Value product = builder.create<arith::MulFOp>(
              bodyLocation, arguments[0], arguments[1]);
          Value sum = builder.create<arith::AddFOp>(bodyLocation, arguments[2],
                                                    product);
          builder.create<linalg::YieldOp>(bodyLocation, sum);
        });
    if (fusionGroup) {
      contraction->setAttr(
          kAlgebraicFusionGroupAttr,
          rewriter.getI64IntegerAttr(static_cast<std::int64_t>(*fusionGroup)));
    }
    valueAccesses.push_back(std::move(resultAccess));
    values.push_back(contraction.getResult(0));
  }

  llvm::DenseSet<Operation *> regionOperations =
      getOperationSet(region.getOperations());

  llvm::SmallVector<Value, 2> replacements;
  replacements.reserve(region.getOutputs().size());
  for (const auto &indexedOutput : llvm::enumerate(region.getOutputs())) {
    std::size_t outputNumber = indexedOutput.index();
    LinalgEinsumRegionOutput output = indexedOutput.value();
    Value replacement = values[outputPlanValues[outputNumber]];
    output.value.replaceUsesWithIf(replacement, [&](OpOperand &use) {
      return !regionOperations.contains(use.getOwner());
    });
    replacements.push_back(replacement);
  }
  for (linalg::GenericOp operation : llvm::reverse(region.getOperations()))
    rewriter.eraseOp(operation);
  return replacements;
}

//===----------------------------------------------------------------------===//
// Fusion-set selection
//===----------------------------------------------------------------------===//
namespace {

unsigned commonDomainPrefix(llvm::ArrayRef<int64_t> lhs,
                            llvm::ArrayRef<int64_t> rhs) {
  unsigned prefix = 0;
  while (prefix < std::min(lhs.size(), rhs.size()) &&
         lhs[prefix] == rhs[prefix])
    ++prefix;
  return prefix;
}

struct FusionAtom {
  llvm::SmallVector<linalg::GenericOp, 4> operations;
  unsigned firstPosition = 0;
  unsigned lastPosition = 0;
};

struct FusionSequence {
  llvm::SmallVector<int64_t, 4> domain;
  bool terminalReduction = false;
};

/// Returns the static leading parallel domain. A reduction-only operation is
/// represented by its complete domain because it may terminate a fused
/// producer-consumer sequence.
std::optional<std::pair<llvm::SmallVector<int64_t, 4>, bool>>
getFusibleDomain(linalg::GenericOp operation) {
  llvm::SmallVector<int64_t, 4> loopRanges = operation.getStaticLoopRanges();
  auto iteratorTypes = operation.getIteratorTypesArray();
  if (loopRanges.empty() || loopRanges.size() != iteratorTypes.size())
    return std::nullopt;

  unsigned leadingParallel = 0;
  while (leadingParallel < iteratorTypes.size() &&
         iteratorTypes[leadingParallel] == utils::IteratorType::parallel)
    ++leadingParallel;

  bool reductionOnly =
      llvm::all_of(iteratorTypes, [](utils::IteratorType type) {
        return type == utils::IteratorType::reduction;
      });
  if (leadingParallel == 0 && !reductionOnly)
    return std::nullopt;

  unsigned domainSize = reductionOnly ? loopRanges.size() : leadingParallel;
  llvm::SmallVector<int64_t, 4> domain;
  domain.reserve(domainSize);
  for (int64_t extent : llvm::ArrayRef(loopRanges).take_front(domainSize)) {
    if (ShapedType::isDynamic(extent) || extent <= 0)
      return std::nullopt;
    domain.push_back(extent);
  }
  return std::pair(std::move(domain), reductionOnly);
}

bool appendToFusionSequence(FusionSequence &sequence,
                            linalg::GenericOp operation) {
  auto domain = getFusibleDomain(operation);
  if (!domain || sequence.terminalReduction)
    return false;

  auto &[operationDomain, reductionOnly] = *domain;
  if (sequence.domain.empty()) {
    // A reduction has no independently partitioned producer domain. It can
    // close a sequence, but cannot initiate one for the loop-fusion primitive
    // used later in the pipeline.
    if (reductionOnly)
      return false;
    sequence.domain = std::move(operationDomain);
    return true;
  }

  unsigned prefix = commonDomainPrefix(sequence.domain, operationDomain);
  if (prefix == 0)
    return false;
  if (reductionOnly && prefix != operationDomain.size())
    return false;
  sequence.domain.resize(prefix);
  sequence.terminalReduction = reductionOnly;
  return true;
}

bool isTransparentBetweenFusionAtoms(Operation &operation) {
  if (isa<tensor::EmptyOp, linalg::FillOp>(operation))
    return true;
  return operation.getName().getDialectNamespace() ==
         arith::ArithDialect::getDialectNamespace();
}

bool valueTransitivelyDependsOnMember(
    Value value, const llvm::DenseSet<Operation *> &members, Block *block) {
  llvm::SmallVector<Operation *, 4> worklist;
  llvm::DenseSet<Operation *> visited;
  if (Operation *definition = value.getDefiningOp())
    worklist.push_back(definition);

  while (!worklist.empty()) {
    Operation *operation = worklist.pop_back_val();
    if (!visited.insert(operation).second)
      continue;
    if (members.contains(operation))
      return true;
    if (operation->getBlock() != block)
      continue;
    for (Value operand : operation->getOperands()) {
      if (Operation *definition = operand.getDefiningOp())
        worklist.push_back(definition);
    }
  }
  return false;
}

/// A skipped scalar expression may use a result from an earlier member and be
/// captured by a later member. Moving the whole set to its final operation
/// would then create a cyclic capture. Reject such non-closed sets before
/// bufferization and outlining.
bool hasDependencyLeavingAndReentering(
    llvm::ArrayRef<linalg::GenericOp> operations) {
  llvm::DenseSet<Operation *> members;
  for (linalg::GenericOp operation : operations)
    members.insert(operation);

  Block *block = operations.front()->getBlock();
  Operation *first = operations.front();
  for (linalg::GenericOp operation : operations) {
    llvm::SmallVector<Value, 8> usedValues(operation->operand_begin(),
                                           operation->operand_end());
    llvm::SetVector<Value> captures;
    getUsedValuesDefinedAbove(operation->getRegions(), captures);
    llvm::append_range(usedValues, captures);
    for (Value used : usedValues) {
      Operation *definition = used.getDefiningOp();
      if (!definition || members.contains(definition) ||
          definition->getBlock() != block || definition->isBeforeInBlock(first))
        continue;
      if (valueTransitivelyDependsOnMember(used, members, block))
        return true;
    }
  }
  return false;
}

bool canFuseAtoms(llvm::ArrayRef<FusionAtom> atoms) {
  if (atoms.size() < 2)
    return false;

  llvm::SmallVector<linalg::GenericOp, 8> operations;
  FusionSequence sequence;
  for (const FusionAtom &atom : atoms) {
    for (linalg::GenericOp operation : atom.operations) {
      if (!appendToFusionSequence(sequence, operation))
        return false;
      operations.push_back(operation);
    }
  }
  return !hasDependencyLeavingAndReentering(operations);
}

void partitionFusionAtoms(llvm::MutableArrayRef<FusionAtom> atoms,
                          std::uint64_t &nextFusionGroup, Builder &builder) {
  if (atoms.size() < 2)
    return;

  struct Partition {
    unsigned kernelCount = std::numeric_limits<unsigned>::max();
    unsigned next = 0;
  };
  llvm::SmallVector<Partition, 8> best(atoms.size() + 1);
  best.back() = Partition{0, static_cast<unsigned>(atoms.size())};

  for (std::size_t begin = atoms.size(); begin-- > 0;) {
    best[begin] = Partition{1 + best[begin + 1].kernelCount,
                            static_cast<unsigned>(begin + 1)};
    for (std::size_t end = begin + 2; end <= atoms.size(); ++end) {
      if (!canFuseAtoms(atoms.slice(begin, end - begin)))
        continue;
      unsigned kernelCount = 1 + best[end].kernelCount;
      // Prefer the longest first set when two exact partitions remove the
      // same number of kernel boundaries. This makes selection deterministic.
      if (kernelCount < best[begin].kernelCount ||
          (kernelCount == best[begin].kernelCount && end > best[begin].next)) {
        best[begin] = Partition{kernelCount, static_cast<unsigned>(end)};
      }
    }
  }

  for (unsigned begin = 0; begin < atoms.size();) {
    unsigned end = best[begin].next;
    if (end - begin >= 2) {
      IntegerAttr group = builder.getI64IntegerAttr(
          static_cast<std::int64_t>(nextFusionGroup++));
      for (FusionAtom &atom : atoms.slice(begin, end - begin)) {
        for (linalg::GenericOp operation : atom.operations)
          operation->setAttr(kAlgebraicFusionGroupAttr, group);
      }
    }
    begin = end;
  }
}

void selectFusionSets(Block &block, std::uint64_t &nextFusionGroup,
                      Builder &builder) {
  llvm::SmallVector<FusionAtom, 8> atoms;
  llvm::DenseMap<std::int64_t, unsigned> markedAtoms;

  unsigned position = 0;
  for (Operation &operation : block) {
    const unsigned currentPosition = position++;
    auto generic = dyn_cast<linalg::GenericOp>(operation);
    if (!generic)
      continue;

    auto group = generic->getAttrOfType<IntegerAttr>(kAlgebraicFusionGroupAttr);
    if (!group) {
      atoms.push_back(FusionAtom{{generic}, currentPosition, currentPosition});
      continue;
    }

    auto [iterator, inserted] =
        markedAtoms.try_emplace(group.getInt(), atoms.size());
    if (inserted) {
      atoms.push_back(FusionAtom{{generic}, currentPosition, currentPosition});
      continue;
    }
    FusionAtom &atom = atoms[iterator->second];
    atom.operations.push_back(generic);
    atom.lastPosition = currentPosition;
  }

  llvm::SmallVector<FusionAtom, 8> run;
  auto finishRun = [&] {
    partitionFusionAtoms(run, nextFusionGroup, builder);
    run.clear();
  };
  for (FusionAtom &atom : atoms) {
    if (!run.empty()) {
      const FusionAtom &previous = run.back();
      if (previous.lastPosition >= atom.firstPosition) {
        // Pre-existing atoms must not interleave. Leave their original group
        // attributes intact and conservatively decline further grouping.
        finishRun();
      } else {
        auto first = std::next(block.begin(), previous.lastPosition + 1);
        auto last = std::next(block.begin(), atom.firstPosition);
        if (llvm::any_of(llvm::make_range(first, last),
                         [](Operation &operation) {
                           return !isTransparentBetweenFusionAtoms(operation);
                         }))
          finishRun();
      }
    }
    run.push_back(std::move(atom));
  }
  finishRun();

  for (Operation &operation : block) {
    for (Region &region : operation.getRegions()) {
      for (Block &nested : region)
        selectFusionSets(nested, nextFusionGroup, builder);
    }
  }
}

void selectFusionSets(ModuleOp module, std::uint64_t &nextFusionGroup) {
  Builder builder(module.getContext());
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    for (Block &block : function.getBody())
      selectFusionSets(block, nextFusionGroup, builder);
  }
}

} // namespace
} // namespace mlir::lapis

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//

using namespace mlir;

namespace mlir {
#define GEN_PASS_DEF_ALGEBRAICKERNELFUSIONPASS
#include "lapis/Transform/Passes.h.inc"
} // namespace mlir

namespace {

struct PlannedRegion {
  mlir::lapis::LinalgEinsumRegion region;
  mlir::lapis::ContractionPlan plan;
  std::uint64_t fusionGroup;
};

struct AlgebraicKernelFusionPass
    : public impl::AlgebraicKernelFusionPassBase<AlgebraicKernelFusionPass> {
  using impl::AlgebraicKernelFusionPassBase<
      AlgebraicKernelFusionPass>::AlgebraicKernelFusionPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    llvm::SmallVector<PlannedRegion, 4> plannedRegions;
    std::uint64_t nextFusionGroup = 0;
    auto fail = [&](Operation *operation, llvm::StringRef context,
                    llvm::Error error) {
      operation->emitError(context) << llvm::toString(std::move(error));
      signalPassFailure();
    };

    // Finish every fallible discovery and planning step before modifying the
    // module. A planner limitation therefore cannot leave a partially
    // transformed module behind.
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      auto regions = mlir::lapis::discoverLinalgEinsumRegions(function);
      if (!regions) {
        fail(function,
             "algebraic region discovery failed: ", regions.takeError());
        return;
      }

      for (mlir::lapis::LinalgEinsumRegion &region : *regions) {
        auto planner = mlir::lapis::ExactContractionPlanner::create(
            region.getExpression());
        if (!planner) {
          fail(region.getRoot(),
               "contraction planner creation failed: ", planner.takeError());
          return;
        }

        llvm::SmallVector<mlir::lapis::ContractionResultConstraint, 2>
            requiredResults;
        requiredResults.reserve(region.getOutputs().size());
        for (const mlir::lapis::LinalgEinsumRegionOutput &output :
             region.getOutputs()) {
          requiredResults.push_back(mlir::lapis::ContractionResultConstraint{
              output.operandSubset, output.access});
        }

        auto plan = planner->plan(requiredResults);
        if (!plan) {
          fail(region.getRoot(),
               "contraction planning failed: ", plan.takeError());
          return;
        }

        auto evaluation = mlir::lapis::evaluateFusionCandidate(region, *plan);
        if (!evaluation) {
          fail(region.getRoot(),
               "fusion candidate evaluation failed: ", evaluation.takeError());
          return;
        }
        if (!evaluation->shouldRewrite())
          continue;
        plannedRegions.push_back(PlannedRegion{
            std::move(region), std::move(*plan), nextFusionGroup++});
      }
    }

    // A region root may remain an external operand of another region when it
    // has multiple uses. Rewrite consumers first so their cached leaf Values
    // are not invalidated when the producer root is replaced and erased.
    llvm::DenseMap<Value, unsigned> regionResults;
    for (const auto &[regionNumber, planned] :
         llvm::enumerate(plannedRegions)) {
      for (const mlir::lapis::LinalgEinsumRegionOutput &output :
           planned.region.getOutputs()) {
        regionResults.insert(
            {output.value, static_cast<unsigned>(regionNumber)});
      }
    }

    llvm::SmallVector<llvm::SmallVector<unsigned, 2>, 4> consumers(
        plannedRegions.size());
    for (const auto &[consumerNumber, planned] :
         llvm::enumerate(plannedRegions)) {
      for (Value operand : planned.region.getOperands()) {
        auto producer = regionResults.find(operand);
        if (producer != regionResults.end())
          consumers[producer->second].push_back(
              static_cast<unsigned>(consumerNumber));
      }
    }

    llvm::SmallVector<unsigned, 4> state(plannedRegions.size(), 0);
    llvm::SmallVector<unsigned, 4> materializationOrder;
    auto scheduleConsumersFirst = [&](auto &&self,
                                      unsigned regionNumber) -> LogicalResult {
      if (state[regionNumber] == 2)
        return success();
      if (state[regionNumber] == 1)
        return failure();
      state[regionNumber] = 1;
      for (unsigned consumer : consumers[regionNumber]) {
        if (failed(self(self, consumer)))
          return failure();
      }
      state[regionNumber] = 2;
      materializationOrder.push_back(regionNumber);
      return success();
    };
    for (unsigned regionNumber = 0; regionNumber < plannedRegions.size();
         ++regionNumber) {
      if (failed(
              scheduleConsumersFirst(scheduleConsumersFirst, regionNumber))) {
        module.emitError("algebraic region dependency graph contains a cycle");
        signalPassFailure();
        return;
      }
    }

    IRRewriter rewriter(&getContext());
    for (unsigned regionNumber : materializationOrder) {
      PlannedRegion &planned = plannedRegions[regionNumber];
      auto result = mlir::lapis::materializeContractionPlan(
          rewriter, planned.region, planned.plan, planned.fusionGroup);
      if (!result) {
        fail(planned.region.getRoot(),
             "contraction materialization failed: ", result.takeError());
        return;
      }
    }

    // Treat algebraically planned regions as indivisible atoms and find an
    // exact, deterministic partition of each block-local atom sequence. A set
    // must have a statically matching leading domain and be closed under SSA
    // dependencies. The post-bufferization legality pass still has final
    // authority over aliasing and memory dependences.
    mlir::lapis::selectFusionSets(module, nextFusionGroup);
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createAlgebraicKernelFusionPass() {
  return std::make_unique<AlgebraicKernelFusionPass>();
}

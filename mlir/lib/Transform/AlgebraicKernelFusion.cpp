//===- AlgebraicKernelFusion.cpp -----------------------------------------===//

#include "lapis/Transform/AlgebraicKernelFusion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
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

llvm::Expected<StorageVolume> getStaticTensorVolume(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || !type.hasStaticShape())
    return llvm::createStringError(
        "logical storage requires a statically shaped tensor result");

  StorageVolume volume = 1;
  for (const auto &[dimensionNumber, extent] :
       llvm::enumerate(type.getShape())) {
    if (extent <= 0 || volume > std::numeric_limits<StorageVolume>::max() /
                                    static_cast<StorageVolume>(extent))
      return llvm::createStringError(
          "logical storage volume overflows at result dimension " +
          llvm::Twine(dimensionNumber));
    volume *= static_cast<StorageVolume>(extent);
  }
  return volume;
}

llvm::Expected<StorageVolume> checkedAddStorage(StorageVolume accumulated,
                                                StorageVolume next,
                                                const llvm::Twine &context) {
  if (accumulated > std::numeric_limits<StorageVolume>::max() - next)
    return llvm::createStringError("logical storage volume overflows " +
                                   context);
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

  llvm::DenseSet<Value> preservedValues;
  for (const LinalgEinsumRegionOutput &output : region.getOutputs())
    preservedValues.insert(output.value);

  StorageVolume originalTemporaryVolume = 0;
  for (linalg::GenericOp operation : region.getOperations()) {
    for (Value result : operation->getResults()) {
      if (preservedValues.contains(result))
        continue;
      auto resultVolume = getStaticTensorVolume(result);
      if (!resultVolume)
        return resultVolume.takeError();
      auto totalVolume =
          checkedAddStorage(originalTemporaryVolume, *resultVolume,
                            "while evaluating the source contraction region");
      if (!totalVolume)
        return totalVolume.takeError();
      originalTemporaryVolume = *totalVolume;
    }
  }

  StorageVolume optimizedTemporaryVolume =
      optimizedCost->totalIntermediateVolume;
  llvm::DenseSet<OperandSubset> preservedSubsets;
  OperandSubset allOperands = costModel->getIndexLiveness().getAllOperands();
  for (const LinalgEinsumRegionOutput &output : region.getOutputs()) {
    if (output.operandSubset == allOperands ||
        !preservedSubsets.insert(output.operandSubset).second)
      continue;
    auto resultVolume = costModel->getResultVolume(output.operandSubset);
    if (!resultVolume)
      return resultVolume.takeError();
    if (*resultVolume > optimizedTemporaryVolume)
      return llvm::createStringError(
          "preserved contraction result exceeds planned temporary volume");
    optimizedTemporaryVolume -= *resultVolume;
  }
  if (optimizedCost->totalWork == originalWork &&
      optimizedTemporaryVolume > originalTemporaryVolume)
    return llvm::createStringError(
        "equal-work contraction plan increases logical temporary volume");

  return FusionCandidateEvaluation{/*originalWork=*/originalWork,
                                   /*optimizedWork=*/optimizedCost->totalWork,
                                   /*originalTemporaryVolume=*/
                                   originalTemporaryVolume,
                                   /*optimizedTemporaryVolume=*/
                                   optimizedTemporaryVolume};
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

struct FusionAtom {
  llvm::SmallVector<linalg::GenericOp, 4> operations;
  unsigned firstPosition = 0;
  unsigned lastPosition = 0;
};

struct FusionSequence {
  llvm::SmallVector<int64_t, 4> domain;
  bool terminalReduction = false;
};

struct FusibleDomain {
  llvm::SmallVector<int64_t, 4> extents;
  bool reductionOnly = false;
};

/// Returns every static parallel dimension in lowering order. The dense
/// Linalg lowering hoists parallel iterators around reductions, so a parallel
/// iterator remains fusible even when it is not a leading iterator in the
/// structured operation. A reduction-only operation is represented by its
/// complete domain because it may terminate a fused producer-consumer
/// sequence.
std::optional<FusibleDomain> getFusibleDomain(linalg::GenericOp operation) {
  llvm::SmallVector<int64_t, 4> loopRanges = operation.getStaticLoopRanges();
  auto iteratorTypes = operation.getIteratorTypesArray();
  if (loopRanges.empty() || loopRanges.size() != iteratorTypes.size())
    return std::nullopt;

  bool reductionOnly =
      llvm::all_of(iteratorTypes, [](utils::IteratorType type) {
        return type == utils::IteratorType::reduction;
      });

  llvm::SmallVector<int64_t, 4> domain;
  for (const auto &[extent, iteratorType] :
       llvm::zip(loopRanges, iteratorTypes)) {
    if (!reductionOnly && iteratorType != utils::IteratorType::parallel)
      continue;
    if (ShapedType::isDynamic(extent) || extent <= 0)
      return std::nullopt;
    domain.push_back(extent);
  }
  if (domain.empty())
    return std::nullopt;
  return FusibleDomain{std::move(domain), reductionOnly};
}

llvm::SmallVector<int64_t, 4> intersectDomains(llvm::ArrayRef<int64_t> lhs,
                                               llvm::ArrayRef<int64_t> rhs) {
  llvm::SmallVector<int64_t, 4> remaining(rhs);
  llvm::SmallVector<int64_t, 4> intersection;
  for (int64_t extent : lhs) {
    auto match = llvm::find(remaining, extent);
    if (match == remaining.end())
      continue;
    intersection.push_back(extent);
    remaining.erase(match);
  }
  return intersection;
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

  if (reductionOnly) {
    auto common = intersectDomains(operationDomain, sequence.domain);
    if (common.size() != operationDomain.size())
      return false;
    sequence.domain = std::move(operationDomain);
  } else {
    sequence.domain = intersectDomains(sequence.domain, operationDomain);
    if (sequence.domain.empty())
      return false;
  }
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

bool hasReductionIterator(linalg::GenericOp operation) {
  return llvm::is_contained(operation.getIteratorTypesArray(),
                            utils::IteratorType::reduction);
}

bool hasOnlyReductionIterators(linalg::GenericOp operation) {
  auto iteratorTypes = operation.getIteratorTypesArray();
  return !iteratorTypes.empty() &&
         llvm::all_of(iteratorTypes, [](utils::IteratorType type) {
           return type == utils::IteratorType::reduction;
         });
}

bool hasConnectedDataflow(llvm::ArrayRef<FusionAtom> atoms) {
  llvm::DenseMap<Operation *, unsigned> atomNumbers;
  for (const auto &[atomNumber, atom] : llvm::enumerate(atoms)) {
    for (linalg::GenericOp operation : atom.operations)
      atomNumbers.insert({operation, static_cast<unsigned>(atomNumber)});
  }

  llvm::SmallVector<llvm::SmallVector<unsigned, 4>, 8> neighbors(atoms.size());
  for (const auto &[atomNumber, atom] : llvm::enumerate(atoms)) {
    for (linalg::GenericOp operation : atom.operations) {
      for (Value operand : operation->getOperands()) {
        Operation *definition = operand.getDefiningOp();
        if (!definition)
          continue;
        auto producer = atomNumbers.find(definition);
        if (producer == atomNumbers.end() || producer->second == atomNumber)
          continue;
        neighbors[atomNumber].push_back(producer->second);
        neighbors[producer->second].push_back(
            static_cast<unsigned>(atomNumber));
      }
    }
  }

  llvm::SmallVector<bool, 8> visited(atoms.size(), false);
  llvm::SmallVector<unsigned, 8> worklist{0};
  visited.front() = true;
  while (!worklist.empty()) {
    unsigned atomNumber = worklist.pop_back_val();
    for (unsigned neighbor : neighbors[atomNumber]) {
      if (visited[neighbor])
        continue;
      visited[neighbor] = true;
      worklist.push_back(neighbor);
    }
  }
  return llvm::all_of(visited, [](bool value) { return value; });
}

bool resultEscapesFusionSet(linalg::GenericOp operation,
                            const llvm::DenseSet<Operation *> &members) {
  return llvm::any_of(operation->getResults(), [&](Value result) {
    return llvm::any_of(result.getUses(), [&](OpOperand &use) {
      return !members.contains(use.getOwner());
    });
  });
}

/// A terminal reduction may consume a pointwise producer without changing the
/// reduction hierarchy. By contrast, absorbing an escaping reduction result
/// into another reduction couples two independently lowerable reductions while
/// leaving the producer storage externally required. That transformation has
/// no algebraic work or materialization benefit, so leave it unselected.
bool introducesEscapingNestedReduction(
    llvm::ArrayRef<linalg::GenericOp> operations) {
  if (operations.size() < 2 || !hasOnlyReductionIterators(operations.back()))
    return false;

  llvm::DenseSet<Operation *> members;
  for (linalg::GenericOp operation : operations)
    members.insert(operation);

  return llvm::any_of(operations.drop_back(), [&](linalg::GenericOp operation) {
    return hasReductionIterator(operation) &&
           resultEscapesFusionSet(operation, members);
  });
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
  // Judge connectivity over the complete set rather than requiring every
  // prefix to be connected. This admits fan-in such as dx, dy, dz -> rhs while
  // still rejecting same-domain operations with no dataflow relationship.
  return hasConnectedDataflow(atoms) &&
         !introducesEscapingNestedReduction(operations) &&
         !hasDependencyLeavingAndReentering(operations);
}

llvm::Expected<StorageVolume>
getEliminatedTemporaryVolume(llvm::ArrayRef<FusionAtom> atoms) {
  llvm::DenseSet<Operation *> members;
  for (const FusionAtom &atom : atoms) {
    for (linalg::GenericOp operation : atom.operations)
      members.insert(operation);
  }

  StorageVolume eliminated = 0;
  for (const FusionAtom &atom : atoms) {
    for (linalg::GenericOp operation : atom.operations) {
      for (Value result : operation->getResults()) {
        if (result.use_empty() ||
            llvm::any_of(result.getUses(), [&](OpOperand &use) {
              return !members.contains(use.getOwner());
            }))
          continue;
        auto resultVolume = getStaticTensorVolume(result);
        if (!resultVolume)
          return resultVolume.takeError();
        auto total =
            checkedAddStorage(eliminated, *resultVolume,
                              "while evaluating a candidate fusion partition");
        if (!total)
          return total.takeError();
        eliminated = *total;
      }
    }
  }
  return eliminated;
}

LogicalResult partitionFusionAtoms(llvm::MutableArrayRef<FusionAtom> atoms,
                                   std::uint64_t &nextFusionGroup,
                                   Builder &builder) {
  if (atoms.size() < 2)
    return success();

  struct Partition {
    StorageVolume eliminatedVolume = 0;
    unsigned kernelCount = std::numeric_limits<unsigned>::max();
    unsigned next = 0;
  };
  llvm::SmallVector<Partition, 8> best(atoms.size() + 1);
  best.back() = Partition{/*eliminatedVolume=*/0, /*kernelCount=*/0,
                          /*next=*/static_cast<unsigned>(atoms.size())};

  for (std::size_t begin = atoms.size(); begin-- > 0;) {
    best[begin] = Partition{/*eliminatedVolume=*/
                            best[begin + 1].eliminatedVolume,
                            /*kernelCount=*/
                            1 + best[begin + 1].kernelCount,
                            /*next=*/static_cast<unsigned>(begin + 1)};
    for (std::size_t end = begin + 2; end <= atoms.size(); ++end) {
      auto candidate = atoms.slice(begin, end - begin);
      if (!canFuseAtoms(candidate))
        continue;
      auto groupVolume = getEliminatedTemporaryVolume(candidate);
      if (!groupVolume) {
        atoms.front().operations.front().emitError(
            "fusion storage evaluation failed: ")
            << llvm::toString(groupVolume.takeError());
        return failure();
      }
      auto eliminatedVolume =
          checkedAddStorage(*groupVolume, best[end].eliminatedVolume,
                            "while combining fusion partitions");
      if (!eliminatedVolume) {
        atoms.front().operations.front().emitError(
            "fusion storage evaluation failed: ")
            << llvm::toString(eliminatedVolume.takeError());
        return failure();
      }
      unsigned kernelCount = 1 + best[end].kernelCount;
      // Logical materialization is the platform-independent profitability
      // objective for equal-work fusion. Kernel count and longest-first order
      // are deterministic tie-breakers after storage volume.
      if (*eliminatedVolume > best[begin].eliminatedVolume ||
          (*eliminatedVolume == best[begin].eliminatedVolume &&
           (kernelCount < best[begin].kernelCount ||
            (kernelCount == best[begin].kernelCount &&
             end > best[begin].next)))) {
        best[begin] = Partition{*eliminatedVolume, kernelCount,
                                static_cast<unsigned>(end)};
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
  return success();
}

LogicalResult selectFusionSets(Block &block, std::uint64_t &nextFusionGroup,
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
  auto finishRun = [&]() -> LogicalResult {
    if (failed(partitionFusionAtoms(run, nextFusionGroup, builder)))
      return failure();
    run.clear();
    return success();
  };
  for (FusionAtom &atom : atoms) {
    if (!run.empty()) {
      const FusionAtom &previous = run.back();
      if (previous.lastPosition >= atom.firstPosition) {
        // Pre-existing atoms must not interleave. Leave their original group
        // attributes intact and conservatively decline further grouping.
        if (failed(finishRun()))
          return failure();
      } else {
        auto first = std::next(block.begin(), previous.lastPosition + 1);
        auto last = std::next(block.begin(), atom.firstPosition);
        if (llvm::any_of(llvm::make_range(first, last),
                         [](Operation &operation) {
                           return !isTransparentBetweenFusionAtoms(operation);
                         }))
          if (failed(finishRun()))
            return failure();
      }
    }
    run.push_back(std::move(atom));
  }
  if (failed(finishRun()))
    return failure();

  for (Operation &operation : block) {
    for (Region &region : operation.getRegions()) {
      for (Block &nested : region) {
        if (failed(selectFusionSets(nested, nextFusionGroup, builder)))
          return failure();
      }
    }
  }
  return success();
}

LogicalResult selectFusionSets(ModuleOp module,
                               std::uint64_t &nextFusionGroup) {
  Builder builder(module.getContext());
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    for (Block &block : function.getBody()) {
      if (failed(selectFusionSets(block, nextFusionGroup, builder)))
        return failure();
    }
  }
  return success();
}

struct IndexedTensorAccess {
  unsigned operation;
  AffineMap map;
};

class DimensionEquivalence {
public:
  explicit DimensionEquivalence(unsigned size) : parents(size) {
    for (unsigned index = 0; index < size; ++index)
      parents[index] = index;
  }

  unsigned find(unsigned value) {
    if (parents[value] != value)
      parents[value] = find(parents[value]);
    return parents[value];
  }

  void unite(unsigned lhs, unsigned rhs) {
    lhs = find(lhs);
    rhs = find(rhs);
    if (lhs != rhs)
      parents[std::max(lhs, rhs)] = std::min(lhs, rhs);
  }

private:
  llvm::SmallVector<unsigned, 16> parents;
};

void addIndexedAccess(
    llvm::DenseMap<Value, llvm::SmallVector<IndexedTensorAccess, 2>> &accesses,
    Value value, unsigned operation, AffineMap map) {
  if (isa<ShapedType>(value.getType()))
    accesses[value].push_back(IndexedTensorAccess{operation, map});
}

/// Aligns the parallel iterators of one selected group without moving a
/// parallel iterator across a reduction iterator. Tensor axes shared between
/// operations establish semantic dimension equivalence; equal static extents
/// provide a deterministic fallback for independent sibling operations. The
/// post-lowering dependence analysis retains final authority over fusion.
LogicalResult
alignFusionGroupIterators(llvm::ArrayRef<linalg::GenericOp> operations,
                          RewriterBase &rewriter) {
  if (operations.size() < 2)
    return success();

  llvm::SmallVector<unsigned, 8> dimensionOffsets{0};
  llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4> loopRanges;
  llvm::SmallVector<llvm::SmallVector<utils::IteratorType, 4>, 4> iteratorTypes;
  for (linalg::GenericOp operation : operations) {
    loopRanges.push_back(operation.getStaticLoopRanges());
    auto types = operation.getIteratorTypesArray();
    iteratorTypes.emplace_back(types.begin(), types.end());
    if (loopRanges.back().size() != iteratorTypes.back().size())
      return success();
    dimensionOffsets.push_back(dimensionOffsets.back() +
                               iteratorTypes.back().size());
  }

  DimensionEquivalence equivalence(dimensionOffsets.back());
  llvm::DenseMap<Value, llvm::SmallVector<IndexedTensorAccess, 2>> accesses;
  for (const auto &indexedOperation : llvm::enumerate(operations)) {
    unsigned operationNumber = indexedOperation.index();
    linalg::GenericOp operation = indexedOperation.value();
    for (OpOperand *input : operation.getDpsInputOperands()) {
      addIndexedAccess(accesses, input->get(), operationNumber,
                       operation.getMatchingIndexingMap(input));
    }

    auto inits = operation.getDpsInitsMutable();
    for (const auto &[resultNumber, result] :
         llvm::enumerate(operation.getResults())) {
      OpOperand &init = inits[resultNumber];
      addIndexedAccess(accesses, result, operationNumber,
                       operation.getMatchingIndexingMap(&init));
    }
  }

  for (auto &entry : accesses) {
    auto &valueAccesses = entry.second;
    for (std::size_t lhsNumber = 0; lhsNumber < valueAccesses.size();
         ++lhsNumber) {
      const IndexedTensorAccess &lhs = valueAccesses[lhsNumber];
      for (std::size_t rhsNumber = lhsNumber + 1;
           rhsNumber < valueAccesses.size(); ++rhsNumber) {
        const IndexedTensorAccess &rhs = valueAccesses[rhsNumber];
        if (lhs.operation == rhs.operation ||
            lhs.map.getNumResults() != rhs.map.getNumResults())
          continue;
        for (auto [lhsExpression, rhsExpression] :
             llvm::zip(lhs.map.getResults(), rhs.map.getResults())) {
          auto lhsDimension = dyn_cast<AffineDimExpr>(lhsExpression);
          auto rhsDimension = dyn_cast<AffineDimExpr>(rhsExpression);
          if (!lhsDimension || !rhsDimension)
            continue;
          equivalence.unite(
              dimensionOffsets[lhs.operation] + lhsDimension.getPosition(),
              dimensionOffsets[rhs.operation] + rhsDimension.getPosition());
        }
      }
    }
  }

  bool terminalReduction =
      llvm::all_of(iteratorTypes.back(), [](utils::IteratorType iteratorType) {
        return iteratorType == utils::IteratorType::reduction;
      });
  unsigned referenceOperation = terminalReduction ? operations.size() - 1 : 0;

  llvm::SmallVector<llvm::DenseMap<unsigned, unsigned>, 4> rootDimensions(
      operations.size());
  llvm::SmallVector<llvm::DenseSet<unsigned>, 4> ambiguousRoots(
      operations.size());
  for (unsigned operationNumber = 0; operationNumber < operations.size();
       ++operationNumber) {
    for (const auto &[dimension, iteratorType] :
         llvm::enumerate(iteratorTypes[operationNumber])) {
      bool eligible =
          iteratorType == utils::IteratorType::parallel ||
          (terminalReduction && operationNumber == referenceOperation &&
           iteratorType == utils::IteratorType::reduction);
      if (!eligible)
        continue;
      unsigned root = equivalence.find(dimensionOffsets[operationNumber] +
                                       static_cast<unsigned>(dimension));
      auto [iterator, inserted] = rootDimensions[operationNumber].try_emplace(
          root, static_cast<unsigned>(dimension));
      if (!inserted && iterator->second != dimension) {
        rootDimensions[operationNumber].erase(root);
        ambiguousRoots[operationNumber].insert(root);
      }
    }
  }

  llvm::SmallVector<llvm::SmallVector<unsigned, 4>, 4> desiredDimensions(
      operations.size());
  llvm::SmallVector<llvm::DenseSet<unsigned>, 4> usedDimensions(
      operations.size());
  llvm::DenseSet<unsigned> selectedRoots;

  for (unsigned referenceDimension = 0;
       referenceDimension < iteratorTypes[referenceOperation].size();
       ++referenceDimension) {
    utils::IteratorType referenceType =
        iteratorTypes[referenceOperation][referenceDimension];
    if ((!terminalReduction &&
         referenceType != utils::IteratorType::parallel) ||
        (terminalReduction && referenceType != utils::IteratorType::reduction))
      continue;

    unsigned root = equivalence.find(dimensionOffsets[referenceOperation] +
                                     referenceDimension);
    if (!selectedRoots.insert(root).second)
      continue;
    bool presentEverywhere = true;
    for (unsigned operationNumber = 0; operationNumber < operations.size();
         ++operationNumber) {
      if (ambiguousRoots[operationNumber].contains(root) ||
          !rootDimensions[operationNumber].contains(root)) {
        presentEverywhere = false;
        break;
      }
    }
    if (!presentEverywhere)
      continue;
    for (unsigned operationNumber = 0; operationNumber < operations.size();
         ++operationNumber) {
      unsigned dimension = rootDimensions[operationNumber].lookup(root);
      desiredDimensions[operationNumber].push_back(dimension);
      usedDimensions[operationNumber].insert(dimension);
    }
  }

  // Complete the common prefix by shape for sibling operations that do not
  // communicate through a tensor. Exact tensor-axis matches above are chosen
  // first, which disambiguates repeated extents in producer-consumer chains.
  for (unsigned referenceDimension = 0;
       referenceDimension < iteratorTypes[referenceOperation].size();
       ++referenceDimension) {
    utils::IteratorType referenceType =
        iteratorTypes[referenceOperation][referenceDimension];
    if ((!terminalReduction &&
         referenceType != utils::IteratorType::parallel) ||
        (terminalReduction &&
         referenceType != utils::IteratorType::reduction) ||
        usedDimensions[referenceOperation].contains(referenceDimension))
      continue;

    int64_t extent = loopRanges[referenceOperation][referenceDimension];
    if (ShapedType::isDynamic(extent) || extent <= 0)
      continue;
    llvm::SmallVector<unsigned, 4> matches(operations.size());
    matches[referenceOperation] = referenceDimension;
    bool presentEverywhere = true;
    for (unsigned operationNumber = 0; operationNumber < operations.size();
         ++operationNumber) {
      if (operationNumber == referenceOperation)
        continue;
      std::optional<unsigned> match;
      for (unsigned dimension = 0;
           dimension < iteratorTypes[operationNumber].size(); ++dimension) {
        if (iteratorTypes[operationNumber][dimension] ==
                utils::IteratorType::parallel &&
            !usedDimensions[operationNumber].contains(dimension) &&
            loopRanges[operationNumber][dimension] == extent) {
          match = dimension;
          break;
        }
      }
      if (!match) {
        presentEverywhere = false;
        break;
      }
      matches[operationNumber] = *match;
    }
    if (!presentEverywhere)
      continue;
    for (unsigned operationNumber = 0; operationNumber < operations.size();
         ++operationNumber) {
      desiredDimensions[operationNumber].push_back(matches[operationNumber]);
      usedDimensions[operationNumber].insert(matches[operationNumber]);
    }
  }

  llvm::SmallVector<llvm::SmallVector<unsigned, 4>, 4> permutations;
  permutations.reserve(operations.size());
  for (unsigned operationNumber = 0; operationNumber < operations.size();
       ++operationNumber) {
    llvm::SmallVector<unsigned, 4> permutation;
    for (unsigned dimension = 0;
         dimension < iteratorTypes[operationNumber].size(); ++dimension)
      permutation.push_back(dimension);

    if (terminalReduction && operationNumber == referenceOperation) {
      permutations.push_back(std::move(permutation));
      continue;
    }

    llvm::SmallVector<unsigned, 4> parallelSlots;
    for (const auto &[dimension, iteratorType] :
         llvm::enumerate(iteratorTypes[operationNumber])) {
      if (iteratorType == utils::IteratorType::parallel)
        parallelSlots.push_back(static_cast<unsigned>(dimension));
    }
    for (unsigned dimension : parallelSlots) {
      if (!usedDimensions[operationNumber].contains(dimension))
        desiredDimensions[operationNumber].push_back(dimension);
    }
    if (parallelSlots.size() != desiredDimensions[operationNumber].size())
      return failure();
    for (auto [slot, dimension] :
         llvm::zip(parallelSlots, desiredDimensions[operationNumber]))
      permutation[slot] = dimension;
    permutations.push_back(std::move(permutation));
  }

  for (unsigned operationNumber = 0; operationNumber < operations.size();
       ++operationNumber) {
    linalg::GenericOp operation = operations[operationNumber];
    llvm::ArrayRef<unsigned> permutation = permutations[operationNumber];
    bool identity =
        llvm::all_of(llvm::enumerate(permutation), [](const auto &indexed) {
          return indexed.index() == indexed.value();
        });
    if (!identity && failed(linalg::interchangeGenericOp(rewriter, operation,
                                                         permutation))) {
      operation.emitError(
          "failed to align parallel iterators for algebraic fusion");
      return failure();
    }
  }
  return success();
}

LogicalResult alignFusionGroupIterators(Block &block, RewriterBase &rewriter) {
  llvm::SmallVector<llvm::SmallVector<linalg::GenericOp, 4>, 4> groups;
  llvm::DenseMap<std::int64_t, unsigned> groupNumbers;
  for (Operation &operation : block) {
    auto generic = dyn_cast<linalg::GenericOp>(operation);
    if (!generic)
      continue;
    auto group = generic->getAttrOfType<IntegerAttr>(kAlgebraicFusionGroupAttr);
    if (!group)
      continue;
    auto [iterator, inserted] =
        groupNumbers.try_emplace(group.getInt(), groups.size());
    if (inserted)
      groups.emplace_back();
    groups[iterator->second].push_back(generic);
  }

  for (auto &group : groups) {
    if (failed(alignFusionGroupIterators(group, rewriter)))
      return failure();
  }
  for (Operation &operation : block) {
    for (Region &region : operation.getRegions()) {
      for (Block &nested : region) {
        if (failed(alignFusionGroupIterators(nested, rewriter)))
          return failure();
      }
    }
  }
  return success();
}

LogicalResult alignFusionGroupIterators(ModuleOp module) {
  IRRewriter rewriter(module.getContext());
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    for (Block &block : function.getBody()) {
      if (failed(alignFusionGroupIterators(block, rewriter)))
        return failure();
    }
  }
  return success();
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
    // must form a connected producer-consumer graph, have a non-empty
    // statically matching parallel domain (allowing permutations), preserve
    // reduction topology, and be closed under SSA dependencies. Equal-work
    // partitions maximize eliminated logical temporary volume before using
    // kernel count as a tie-breaker. The post-bufferization legality pass still
    // has final authority over aliasing and memory dependences.
    if (failed(mlir::lapis::selectFusionSets(module, nextFusionGroup))) {
      signalPassFailure();
      return;
    }

    // Linalg-to-loop conversion faithfully preserves the iterator schedule
    // encoded by each operation. Align semantically corresponding parallel
    // dimensions now so the selected groups lower to compatible loop prefixes.
    // Reduction iterators retain their original positions and ordering.
    if (failed(mlir::lapis::alignFusionGroupIterators(module))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createAlgebraicKernelFusionPass() {
  return std::make_unique<AlgebraicKernelFusionPass>();
}

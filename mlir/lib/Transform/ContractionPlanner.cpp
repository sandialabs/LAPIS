//===- ContractionPlanner.cpp -------------------------------------------===//

#include "lapis/Transform/ContractionPlanner.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace mlir::lapis {

//===----------------------------------------------------------------------===//
// Plan representation, index liveness, and verification
//===----------------------------------------------------------------------===//

namespace {

OperandSubset makeAllOperands(unsigned operandCount) {
  if (operandCount == kMaxTrackedOperands)
    return std::numeric_limits<OperandSubset>::max();
  return (OperandSubset{1} << operandCount) - 1;
}

llvm::Error invalidStep(std::size_t step, const llvm::Twine &message) {
  return llvm::createStringError("contraction step " + llvm::Twine(step) +
                                 ": " + message);
}

bool haveSameIndices(llvm::ArrayRef<IndexId> lhs, llvm::ArrayRef<IndexId> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  llvm::DenseSet<IndexId> lhsIndices(lhs.begin(), lhs.end());
  if (lhsIndices.size() != lhs.size())
    return false;
  return llvm::all_of(
      rhs, [&](IndexId index) { return lhsIndices.contains(index); });
}

} // namespace

ContractionPlan::ContractionPlan(unsigned operandCount,
                                 llvm::ArrayRef<ContractionStep> steps,
                                 PlanValueId result)
    : operandCount(operandCount), steps(steps), result(result) {}

IndexLiveness::IndexLiveness(unsigned operandCount, OperandSubset allOperands,
                             llvm::DenseMap<IndexId, OperandSubset> indexUsers,
                             llvm::SmallVector<IndexId, 4> indexOrder,
                             llvm::SmallVector<IndexId, 4> resultOrder,
                             llvm::DenseSet<IndexId> resultIndices)
    : operandCount(operandCount), allOperands(allOperands),
      indexUsers(std::move(indexUsers)), indexOrder(std::move(indexOrder)),
      resultOrder(std::move(resultOrder)),
      resultIndices(std::move(resultIndices)) {}

llvm::Expected<IndexLiveness>
IndexLiveness::create(const EinsumExpression &expression) {
  std::size_t expressionOperandCount = expression.getOperands().size();
  if (expressionOperandCount > kMaxTrackedOperands)
    return llvm::createStringError(
        "index liveness supports at most " + llvm::Twine(kMaxTrackedOperands) +
        " operands, but expression has " + llvm::Twine(expressionOperandCount));
  unsigned operandCount = static_cast<unsigned>(expressionOperandCount);

  llvm::DenseMap<IndexId, OperandSubset> indexUsers;
  for (const auto &[operand, access] :
       llvm::enumerate(expression.getOperands())) {
    OperandSubset operandBit = OperandSubset{1} << operand;
    for (IndexId index : access.indices)
      indexUsers[index] |= operandBit;
  }

  llvm::SmallVector<IndexId, 4> resultOrder(expression.getResult().indices);
  llvm::DenseSet<IndexId> resultIndices(resultOrder.begin(), resultOrder.end());
  return IndexLiveness(
      operandCount, makeAllOperands(operandCount), std::move(indexUsers),
      llvm::SmallVector<IndexId, 4>(expression.getIndexOrder()),
      std::move(resultOrder), std::move(resultIndices));
}

OperandSubset IndexLiveness::getUsers(IndexId index) const {
  return indexUsers.lookup(index);
}

llvm::Expected<llvm::SmallVector<IndexId, 4>>
IndexLiveness::getLiveIndices(OperandSubset operands) const {
  if (operands == 0)
    return llvm::createStringError(
        "cannot compute index liveness for an empty operand subset");
  if ((operands & ~allOperands) != 0)
    return llvm::createStringError(
        "operand subset contains an operand outside the expression");

  llvm::SmallVector<IndexId, 4> liveIndices;
  for (IndexId index : resultOrder) {
    if ((getUsers(index) & operands) != 0)
      liveIndices.push_back(index);
  }

  OperandSubset outsideOperands = allOperands ^ operands;
  for (IndexId index : indexOrder) {
    if (resultIndices.contains(index))
      continue;
    OperandSubset users = getUsers(index);
    if ((users & operands) != 0 && (users & outsideOperands) != 0)
      liveIndices.push_back(index);
  }
  return liveIndices;
}

llvm::Expected<llvm::SmallVector<OperandSubset, 8>>
computePlanOperandSubsets(const ContractionPlan &plan) {
  unsigned operandCount = plan.getOperandCount();
  if (operandCount == 0)
    return llvm::createStringError(
        "a contraction plan must have at least one operand");
  if (operandCount > kMaxTrackedOperands)
    return llvm::createStringError("operand-subset tracking supports at most " +
                                   llvm::Twine(kMaxTrackedOperands) +
                                   " operands, but plan has " +
                                   llvm::Twine(operandCount));

  llvm::SmallVector<OperandSubset, 8> valueOperands;
  valueOperands.reserve(operandCount + plan.getSteps().size());
  for (unsigned operand = 0; operand < operandCount; ++operand)
    valueOperands.push_back(OperandSubset{1} << operand);

  for (const auto &[stepNumber, step] : llvm::enumerate(plan.getSteps())) {
    if (step.lhs >= valueOperands.size())
      return invalidStep(stepNumber,
                         "left input refers to unavailable plan value " +
                             llvm::Twine(step.lhs));
    if (step.rhs >= valueOperands.size())
      return invalidStep(stepNumber,
                         "right input refers to unavailable plan value " +
                             llvm::Twine(step.rhs));

    OperandSubset lhsOperands = valueOperands[step.lhs];
    OperandSubset rhsOperands = valueOperands[step.rhs];
    if ((lhsOperands & rhsOperands) != 0)
      return invalidStep(stepNumber,
                         "left and right inputs contain overlapping operands");
    valueOperands.push_back(lhsOperands | rhsOperands);
  }
  return valueOperands;
}

llvm::Expected<PlanValueId>
findPlanValueForOperandSubset(const ContractionPlan &plan,
                              OperandSubset operands) {
  if (operands == 0)
    return llvm::createStringError(
        "cannot find a plan value for an empty operand subset");
  auto valueOperands = computePlanOperandSubsets(plan);
  if (!valueOperands)
    return valueOperands.takeError();
  auto value = llvm::find(*valueOperands, operands);
  if (value == valueOperands->end())
    return llvm::createStringError(
        "contraction plan does not produce operand subset " +
        llvm::Twine(operands));
  return static_cast<PlanValueId>(value - valueOperands->begin());
}

llvm::Error verifyContractionPlan(const EinsumExpression &expression,
                                  const ContractionPlan &plan) {
  if (plan.getOperandCount() != expression.getOperands().size())
    return llvm::createStringError(
        "plan operand count " + llvm::Twine(plan.getOperandCount()) +
        " does not match expression operand count " +
        llvm::Twine(expression.getOperands().size()));

  std::size_t expectedSteps = expression.getOperands().size() - 1;
  if (plan.getSteps().size() != expectedSteps)
    return llvm::createStringError(
        "complete binary contraction plan requires " +
        llvm::Twine(expectedSteps) + " steps, but plan has " +
        llvm::Twine(plan.getSteps().size()));

  auto liveness = IndexLiveness::create(expression);
  if (!liveness)
    return liveness.takeError();

  auto valueOperands = computePlanOperandSubsets(plan);
  if (!valueOperands)
    return valueOperands.takeError();

  for (const auto &[stepNumber, step] : llvm::enumerate(plan.getSteps())) {
    PlanValueId stepResult = plan.getStepResult(stepNumber);
    auto expectedResultIndices =
        liveness->getLiveIndices((*valueOperands)[stepResult]);
    if (!expectedResultIndices)
      return expectedResultIndices.takeError();
    if (!haveSameIndices(step.resultIndices, *expectedResultIndices))
      return invalidStep(stepNumber,
                         "result indices do not match index liveness");
  }

  if (plan.getResult() >= valueOperands->size())
    return llvm::createStringError(
        "plan result refers to unavailable plan value " +
        llvm::Twine(plan.getResult()));
  if ((*valueOperands)[plan.getResult()] != liveness->getAllOperands())
    return llvm::createStringError(
        "plan result does not contain every expression operand");

  llvm::ArrayRef<IndexId> actualResultIndices;
  if (plan.getResult() < plan.getOperandCount()) {
    actualResultIndices = expression.getOperands()[plan.getResult()].indices;
  } else {
    actualResultIndices =
        plan.getSteps()[plan.getResult() - plan.getOperandCount()]
            .resultIndices;
  }
  if (!llvm::equal(actualResultIndices, expression.getResult().indices))
    return llvm::createStringError(
        "plan result indices do not match expression result indices");

  return llvm::Error::success();
}

} // namespace mlir::lapis

namespace mlir::lapis {

//===----------------------------------------------------------------------===//
// Platform-independent cost model
//===----------------------------------------------------------------------===//

namespace {

llvm::Expected<WorkCost> checkedMultiply(WorkCost lhs, WorkCost rhs,
                                         IndexId index) {
  if (rhs != 0 && lhs > std::numeric_limits<WorkCost>::max() / rhs)
    return llvm::createStringError(
        "contraction work overflows while multiplying extent for index " +
        llvm::Twine(index));
  return lhs * rhs;
}

llvm::Expected<WorkCost> checkedAdd(WorkCost lhs, WorkCost rhs,
                                    std::size_t step) {
  if (lhs > std::numeric_limits<WorkCost>::max() - rhs)
    return llvm::createStringError("total contraction work overflows at step " +
                                   llvm::Twine(step));
  return lhs + rhs;
}

} // namespace

ContractionCostModel::ContractionCostModel(EinsumExpression expression,
                                           IndexLiveness liveness)
    : expression(std::move(expression)), liveness(std::move(liveness)) {}

llvm::Expected<ContractionCostModel>
ContractionCostModel::create(const EinsumExpression &expression) {
  auto liveness = IndexLiveness::create(expression);
  if (!liveness)
    return liveness.takeError();
  return ContractionCostModel(expression, std::move(*liveness));
}

llvm::Expected<llvm::SmallVector<IndexId, 4>>
ContractionCostModel::getValueIndices(OperandSubset operands) const {
  if (operands == 0)
    return llvm::createStringError(
        "cannot compute work for an empty operand subset");
  if ((operands & ~liveness.getAllOperands()) != 0)
    return llvm::createStringError(
        "operand subset contains an operand outside the expression");

  if ((operands & (operands - 1)) == 0) {
    for (unsigned operand = 0; operand < expression.getOperands().size();
         ++operand) {
      if (operands == (OperandSubset{1} << operand))
        return llvm::SmallVector<IndexId, 4>(
            expression.getOperands()[operand].indices);
    }
  }
  return liveness.getLiveIndices(operands);
}

llvm::Expected<WorkCost>
ContractionCostModel::getContractionWork(OperandSubset lhs,
                                         OperandSubset rhs) const {
  if ((lhs & rhs) != 0)
    return llvm::createStringError(
        "cannot contract overlapping operand subsets");

  auto lhsIndices = getValueIndices(lhs);
  if (!lhsIndices)
    return lhsIndices.takeError();
  auto rhsIndices = getValueIndices(rhs);
  if (!rhsIndices)
    return rhsIndices.takeError();

  llvm::DenseSet<IndexId> iterationIndices(lhsIndices->begin(),
                                           lhsIndices->end());
  iterationIndices.insert(rhsIndices->begin(), rhsIndices->end());

  WorkCost work = 1;
  for (IndexId index : expression.getIndexOrder()) {
    if (!iterationIndices.contains(index))
      continue;
    auto extent = expression.getExtent(index);
    if (!extent)
      return llvm::createStringError("missing extent for index " +
                                     llvm::Twine(index));
    auto product = checkedMultiply(work, static_cast<WorkCost>(*extent), index);
    if (!product)
      return product.takeError();
    work = *product;
  }
  return work;
}

llvm::Expected<ContractionPlanCost>
ContractionCostModel::evaluatePlan(const ContractionPlan &plan) const {
  if (llvm::Error error = verifyContractionPlan(expression, plan))
    return std::move(error);

  auto valueOperands = computePlanOperandSubsets(plan);
  if (!valueOperands)
    return valueOperands.takeError();

  ContractionPlanCost planCost;
  planCost.totalWork = 0;
  planCost.stepCosts.reserve(plan.getSteps().size());
  for (const auto &[stepNumber, step] : llvm::enumerate(plan.getSteps())) {
    auto stepCost = getContractionWork((*valueOperands)[step.lhs],
                                       (*valueOperands)[step.rhs]);
    if (!stepCost)
      return stepCost.takeError();
    auto totalWork = checkedAdd(planCost.totalWork, *stepCost, stepNumber);
    if (!totalWork)
      return totalWork.takeError();
    planCost.stepCosts.push_back(*stepCost);
    planCost.totalWork = *totalWork;
  }
  return planCost;
}

} // namespace mlir::lapis

namespace mlir::lapis {

//===----------------------------------------------------------------------===//
// Exact subset dynamic-programming planner
//===----------------------------------------------------------------------===//

namespace {

struct PlannerState {
  WorkCost totalWork;
  OperandSubset lhs;
  OperandSubset rhs;
};

bool isSingleton(OperandSubset operands) {
  return (operands & (operands - 1)) == 0;
}

std::optional<WorkCost> checkedPlanWork(WorkCost lhsWork, WorkCost rhsWork,
                                        WorkCost stepWork) {
  constexpr WorkCost max = std::numeric_limits<WorkCost>::max();
  if (lhsWork > max - rhsWork)
    return std::nullopt;
  WorkCost childWork = lhsWork + rhsWork;
  if (childWork > max - stepWork)
    return std::nullopt;
  return childWork + stepWork;
}

bool isBetterPlan(WorkCost totalWork, OperandSubset lhs, OperandSubset rhs,
                  const std::optional<PlannerState> &current) {
  if (!current)
    return true;
  if (totalWork != current->totalWork)
    return totalWork < current->totalWork;
  return std::tie(lhs, rhs) < std::tie(current->lhs, current->rhs);
}

bool haveSameIndexSet(llvm::ArrayRef<IndexId> lhs,
                      llvm::ArrayRef<IndexId> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  llvm::SmallDenseSet<IndexId, 4> lhsIndices(lhs.begin(), lhs.end());
  if (lhsIndices.size() != lhs.size())
    return false;
  return llvm::all_of(
      rhs, [&](IndexId index) { return lhsIndices.contains(index); });
}

llvm::Expected<llvm::SmallVector<OperandSubset, 2>>
verifyRequiredResults(llvm::ArrayRef<ContractionResultConstraint> required,
                      OperandSubset allOperands,
                      const IndexLiveness &liveness) {
  llvm::SmallVector<OperandSubset, 2> subsets;
  subsets.reserve(required.size());
  for (const auto &[resultNumber, result] : llvm::enumerate(required)) {
    OperandSubset subset = result.operands;
    if (subset == 0)
      return llvm::createStringError("required contraction result " +
                                     llvm::Twine(resultNumber) + " is empty");
    if ((subset & ~allOperands) != 0)
      return llvm::createStringError(
          "required contraction result " + llvm::Twine(resultNumber) +
          " contains an operand outside the expression");

    auto liveIndices = liveness.getLiveIndices(subset);
    if (!liveIndices)
      return liveIndices.takeError();
    if (!haveSameIndexSet(result.access.indices, *liveIndices))
      return llvm::createStringError("required contraction result " +
                                     llvm::Twine(resultNumber) +
                                     " indices do not match index liveness");

    for (const auto &[existingNumber, existing] : llvm::enumerate(required)) {
      if (existingNumber >= resultNumber)
        break;
      if (existing.operands == subset &&
          !llvm::equal(existing.access.indices, result.access.indices))
        return llvm::createStringError(
            "required contraction results for the same operand subset have "
            "different index orders");
    }
    subsets.push_back(subset);
  }

  for (std::size_t lhs = 0; lhs < subsets.size(); ++lhs) {
    for (std::size_t rhs = lhs + 1; rhs < subsets.size(); ++rhs) {
      OperandSubset intersection = subsets[lhs] & subsets[rhs];
      if (intersection != 0 && intersection != subsets[lhs] &&
          intersection != subsets[rhs])
        return llvm::createStringError("required contraction results " +
                                       llvm::Twine(lhs) + " and " +
                                       llvm::Twine(rhs) + " partially overlap");
    }
  }
  return subsets;
}

bool isCompatibleWithRequiredSubsets(OperandSubset operands,
                                     llvm::ArrayRef<OperandSubset> required) {
  return llvm::all_of(required, [&](OperandSubset requiredOperands) {
    OperandSubset intersection = operands & requiredOperands;
    return intersection == 0 || intersection == operands ||
           intersection == requiredOperands;
  });
}

} // namespace

ExactContractionPlanner::ExactContractionPlanner(ContractionCostModel costModel)
    : costModel(std::move(costModel)) {}

llvm::Expected<ExactContractionPlanner>
ExactContractionPlanner::create(const EinsumExpression &expression) {
  std::size_t operandCount = expression.getOperands().size();
  if (operandCount > kMaxExactPlannerOperands)
    return llvm::createStringError(
        "exact contraction planning supports at most " +
        llvm::Twine(kMaxExactPlannerOperands) +
        " operands, but expression has " + llvm::Twine(operandCount));

  auto costModel = ContractionCostModel::create(expression);
  if (!costModel)
    return costModel.takeError();
  return ExactContractionPlanner(std::move(*costModel));
}

llvm::Expected<ContractionPlan> ExactContractionPlanner::plan(
    llvm::ArrayRef<ContractionResultConstraint> requiredResults) const {
  const EinsumExpression &expression = costModel.getExpression();
  unsigned operandCount =
      static_cast<unsigned>(expression.getOperands().size());

  const OperandSubset stateCount = OperandSubset{1} << operandCount;
  const OperandSubset allOperands = stateCount - 1;
  auto requiredSubsets = verifyRequiredResults(requiredResults, allOperands,
                                               costModel.getIndexLiveness());
  if (!requiredSubsets)
    return requiredSubsets.takeError();

  if (operandCount == 1) {
    ContractionPlan identity(/*operandCount=*/1, /*steps=*/{}, /*result=*/0);
    if (llvm::Error error = verifyContractionPlan(expression, identity))
      return std::move(error);
    return identity;
  }

  std::vector<std::optional<PlannerState>> best(stateCount);

  for (unsigned operand = 0; operand < operandCount; ++operand) {
    OperandSubset singleton = OperandSubset{1} << operand;
    best[singleton] = PlannerState{/*totalWork=*/0, /*lhs=*/0, /*rhs=*/0};
  }

  for (OperandSubset operands = 1; operands <= allOperands; ++operands) {
    if (!isCompatibleWithRequiredSubsets(operands, *requiredSubsets))
      continue;
    if (isSingleton(operands))
      continue;

    // Requiring the lowest set bit to appear on one side enumerates each
    // unordered bipartition exactly once.
    OperandSubset anchor = operands & (~operands + 1);
    for (OperandSubset lhs = (operands - 1) & operands; lhs != 0;
         lhs = (lhs - 1) & operands) {
      if ((lhs & anchor) == 0)
        continue;
      OperandSubset rhs = operands ^ lhs;
      if (rhs == 0 || !best[lhs] || !best[rhs])
        continue;

      auto stepWork = costModel.getContractionWork(lhs, rhs);
      if (!stepWork) {
        // Valid, disjoint subsets satisfy every cost-model precondition. Its
        // only possible failure here is an unrepresentable arithmetic result,
        // which cannot participate in a representable optimum.
        llvm::consumeError(stepWork.takeError());
        continue;
      }

      auto totalWork = checkedPlanWork(best[lhs]->totalWork,
                                       best[rhs]->totalWork, *stepWork);
      if (!totalWork)
        continue;

      auto [canonicalLhs, canonicalRhs] = std::minmax(lhs, rhs);
      if (isBetterPlan(*totalWork, canonicalLhs, canonicalRhs, best[operands]))
        best[operands] = PlannerState{*totalWork, canonicalLhs, canonicalRhs};
    }
  }

  if (!best[allOperands])
    return llvm::createStringError(
        "no representable exact contraction plan satisfies the required "
        "operand subsets");

  llvm::SmallVector<ContractionStep, 4> steps;
  steps.reserve(operandCount - 1);
  const IndexLiveness &liveness = costModel.getIndexLiveness();

  auto emitPlan = [&](auto &&self,
                      OperandSubset operands) -> llvm::Expected<PlanValueId> {
    if (isSingleton(operands)) {
      for (unsigned operand = 0; operand < operandCount; ++operand) {
        if (operands == (OperandSubset{1} << operand))
          return operand;
      }
      return llvm::createStringError(
          "exact contraction planner produced an invalid singleton subset");
    }

    const PlannerState &state = *best[operands];
    auto lhs = self(self, state.lhs);
    if (!lhs)
      return lhs.takeError();
    auto rhs = self(self, state.rhs);
    if (!rhs)
      return rhs.takeError();
    llvm::SmallVector<IndexId, 4> resultIndices;
    auto constraint = llvm::find_if(requiredResults, [&](const auto &required) {
      return required.operands == operands;
    });
    if (constraint != requiredResults.end()) {
      resultIndices.assign(constraint->access.indices.begin(),
                           constraint->access.indices.end());
    } else {
      auto liveIndices = liveness.getLiveIndices(operands);
      if (!liveIndices)
        return liveIndices.takeError();
      resultIndices = std::move(*liveIndices);
    }

    steps.push_back(ContractionStep{*lhs, *rhs, std::move(resultIndices)});
    return operandCount + static_cast<PlanValueId>(steps.size()) - 1;
  };

  auto result = emitPlan(emitPlan, allOperands);
  if (!result)
    return result.takeError();

  ContractionPlan plan(operandCount, steps, *result);
  if (llvm::Error error = verifyContractionPlan(expression, plan))
    return std::move(error);
  for (OperandSubset required : *requiredSubsets) {
    auto value = findPlanValueForOperandSubset(plan, required);
    if (!value)
      return value.takeError();
  }

  auto reconstructedCost = costModel.evaluatePlan(plan);
  if (!reconstructedCost)
    return reconstructedCost.takeError();
  if (reconstructedCost->totalWork != best[allOperands]->totalWork)
    return llvm::createStringError(
        "reconstructed contraction plan does not match its planned work");

  return plan;
}

} // namespace mlir::lapis

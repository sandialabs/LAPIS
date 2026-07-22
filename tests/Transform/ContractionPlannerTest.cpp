#include "lapis/Transform/ContractionPlanner.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "gtest/gtest.h"
#include <initializer_list>
#include <limits>

namespace mlir::lapis::contraction_plan_tests {
namespace {

TensorAccess access(std::initializer_list<IndexId> indices) {
  return TensorAccess{llvm::SmallVector<IndexId, 4>(indices)};
}

ContractionStep step(PlanValueId lhs, PlanValueId rhs,
                     std::initializer_list<IndexId> resultIndices) {
  return {lhs, rhs,
          llvm::SmallVector<IndexId, 4>(resultIndices.begin(),
                                        resultIndices.end())};
}

void expectIndices(llvm::ArrayRef<IndexId> actual,
                   std::initializer_list<IndexId> expected) {
  EXPECT_TRUE(llvm::equal(actual, expected));
}

void expectValid(const EinsumExpression &expression,
                 const ContractionPlan &plan) {
  if (llvm::Error error = verifyContractionPlan(expression, plan))
    ADD_FAILURE() << llvm::toString(std::move(error));
}

void expectInvalid(const EinsumExpression &expression,
                   const ContractionPlan &plan) {
  llvm::Error error = verifyContractionPlan(expression, plan);
  EXPECT_TRUE(static_cast<bool>(error));
  if (error)
    llvm::consumeError(std::move(error));
}

void expectInvalidSubset(
    llvm::Expected<llvm::SmallVector<OperandSubset, 8>> subsets) {
  EXPECT_FALSE(static_cast<bool>(subsets));
  if (!subsets)
    llvm::consumeError(subsets.takeError());
}

TEST(IndexLivenessTest, TracksThreeWayReductionIndexAcrossBoundary) {
  auto expression =
      EinsumExpression::create({access({0, 2}), access({2, 1}), access({2})},
                               access({0, 1}), {{0, 3}, {1, 5}, {2, 7}});
  ASSERT_TRUE(static_cast<bool>(expression));

  auto liveness = IndexLiveness::create(*expression);
  ASSERT_TRUE(static_cast<bool>(liveness));
  EXPECT_EQ(liveness->getUsers(2), OperandSubset{0b111});

  auto abIndices = liveness->getLiveIndices(0b011);
  ASSERT_TRUE(static_cast<bool>(abIndices));
  expectIndices(*abIndices, {0, 1, 2});

  auto allIndices = liveness->getLiveIndices(0b111);
  ASSERT_TRUE(static_cast<bool>(allIndices));
  expectIndices(*allIndices, {0, 1});
}

TEST(IndexLivenessTest, DropsReductionClosedInsideSubset) {
  auto expression = EinsumExpression::create(
      {access({0, 2}), access({2, 3}), access({3, 1})}, access({0, 1}),
      {{0, 3}, {1, 5}, {2, 7}, {3, 11}});
  ASSERT_TRUE(static_cast<bool>(expression));

  auto liveness = IndexLiveness::create(*expression);
  ASSERT_TRUE(static_cast<bool>(liveness));
  auto abIndices = liveness->getLiveIndices(0b011);
  ASSERT_TRUE(static_cast<bool>(abIndices));
  expectIndices(*abIndices, {0, 3});
}

TEST(IndexLivenessTest, PreservesFinalResultOrder) {
  auto expression =
      EinsumExpression::create({access({0, 2}), access({2, 1})}, access({1, 0}),
                               {{0, 3}, {1, 5}, {2, 7}});
  ASSERT_TRUE(static_cast<bool>(expression));

  auto liveness = IndexLiveness::create(*expression);
  ASSERT_TRUE(static_cast<bool>(liveness));
  auto indices = liveness->getLiveIndices(0b11);
  ASSERT_TRUE(static_cast<bool>(indices));
  expectIndices(*indices, {1, 0});
}

TEST(IndexLivenessTest, RejectsInvalidSubsets) {
  auto expression =
      EinsumExpression::create({access({0})}, access({0}), {{0, 3}});
  ASSERT_TRUE(static_cast<bool>(expression));

  auto liveness = IndexLiveness::create(*expression);
  ASSERT_TRUE(static_cast<bool>(liveness));

  auto empty = liveness->getLiveIndices(0);
  EXPECT_FALSE(static_cast<bool>(empty));
  if (!empty)
    llvm::consumeError(empty.takeError());

  auto outside = liveness->getLiveIndices(0b10);
  EXPECT_FALSE(static_cast<bool>(outside));
  if (!outside)
    llvm::consumeError(outside.takeError());
}

TEST(IndexLivenessTest, SupportsFullOperandMask) {
  llvm::SmallVector<TensorAccess, 4> operands;
  operands.resize(kMaxTrackedOperands, access({}));
  auto expression = EinsumExpression::create(operands, access({}), {});
  ASSERT_TRUE(static_cast<bool>(expression));

  auto liveness = IndexLiveness::create(*expression);
  ASSERT_TRUE(static_cast<bool>(liveness));
  EXPECT_EQ(liveness->getAllOperands(),
            std::numeric_limits<OperandSubset>::max());

  auto indices = liveness->getLiveIndices(liveness->getAllOperands());
  ASSERT_TRUE(static_cast<bool>(indices));
  EXPECT_TRUE(indices->empty());
}

TEST(IndexLivenessTest, RejectsMoreOperandsThanSubsetCanRepresent) {
  llvm::SmallVector<TensorAccess, 4> operands;
  operands.resize(kMaxTrackedOperands + 1, access({}));
  auto expression = EinsumExpression::create(operands, access({}), {});
  ASSERT_TRUE(static_cast<bool>(expression));

  auto liveness = IndexLiveness::create(*expression);
  EXPECT_FALSE(static_cast<bool>(liveness));
  if (!liveness)
    llvm::consumeError(liveness.takeError());
}

TEST(OperandSubsetTest, TracksEveryPlanValue) {
  ContractionPlan plan(3, {step(1, 2, {2}), step(0, 3, {0})}, 4);
  auto subsets = computePlanOperandSubsets(plan);
  ASSERT_TRUE(static_cast<bool>(subsets));

  ASSERT_EQ(subsets->size(), 5u);
  EXPECT_EQ((*subsets)[0], OperandSubset{0b001});
  EXPECT_EQ((*subsets)[1], OperandSubset{0b010});
  EXPECT_EQ((*subsets)[2], OperandSubset{0b100});
  EXPECT_EQ((*subsets)[3], OperandSubset{0b110});
  EXPECT_EQ((*subsets)[4], OperandSubset{0b111});
}

TEST(OperandSubsetTest, RejectsForwardReference) {
  ContractionPlan plan(3, {step(4, 1, {}), step(0, 2, {})}, 4);
  expectInvalidSubset(computePlanOperandSubsets(plan));
}

TEST(OperandSubsetTest, RejectsOverlappingSubtrees) {
  ContractionPlan plan(3, {step(0, 1, {}), step(3, 1, {})}, 4);
  expectInvalidSubset(computePlanOperandSubsets(plan));
}

TEST(ContractionPlanTest, AcceptsBothABxOrders) {
  auto expression =
      EinsumExpression::create({access({0, 2}), access({2, 1}), access({1})},
                               access({0}), {{0, 3}, {1, 5}, {2, 7}});
  ASSERT_TRUE(static_cast<bool>(expression));

  ContractionPlan bxFirst(3, {step(1, 2, {2}), step(0, 3, {0})}, 4);
  ContractionPlan abFirst(3, {step(0, 1, {0, 1}), step(3, 2, {0})}, 4);
  expectValid(*expression, bxFirst);
  expectValid(*expression, abFirst);
}

TEST(ContractionPlanTest, RejectsPrematureThreeWayReduction) {
  auto expression =
      EinsumExpression::create({access({0, 2}), access({2, 1}), access({2})},
                               access({0, 1}), {{0, 3}, {1, 5}, {2, 7}});
  ASSERT_TRUE(static_cast<bool>(expression));

  ContractionPlan plan(3, {step(0, 1, {0, 1}), step(3, 2, {0, 1})}, 4);
  expectInvalid(*expression, plan);
}

TEST(ContractionPlanTest, AcceptsPermutedIntermediateOrder) {
  auto expression =
      EinsumExpression::create({access({0, 2}), access({2, 1}), access({1})},
                               access({0}), {{0, 3}, {1, 5}, {2, 7}});
  ASSERT_TRUE(static_cast<bool>(expression));

  ContractionPlan plan(3, {step(0, 1, {1, 0}), step(3, 2, {0})}, 4);
  expectValid(*expression, plan);

  ContractionPlan duplicateIndex(3, {step(0, 1, {0, 0}), step(3, 2, {0})}, 4);
  expectInvalid(*expression, duplicateIndex);
}

TEST(ContractionPlanTest, AcceptsOuterProductWithPermutedResult) {
  auto expression = EinsumExpression::create({access({0}), access({1})},
                                             access({1, 0}), {{0, 3}, {1, 5}});
  ASSERT_TRUE(static_cast<bool>(expression));

  ContractionPlan plan(2, {step(0, 1, {1, 0})}, 2);
  expectValid(*expression, plan);
}

TEST(ContractionPlanTest, AcceptsOneSidedReductionDuringBinaryStep) {
  auto expression =
      EinsumExpression::create({access({0}), access({})}, access({}), {{0, 3}});
  ASSERT_TRUE(static_cast<bool>(expression));

  ContractionPlan plan(2, {step(0, 1, {})}, 2);
  expectValid(*expression, plan);
}

TEST(ContractionPlanTest, AcceptsSingleOperandIdentity) {
  auto expression =
      EinsumExpression::create({access({0})}, access({0}), {{0, 3}});
  ASSERT_TRUE(static_cast<bool>(expression));

  ContractionPlan plan(1, {}, 0);
  expectValid(*expression, plan);
}

TEST(ContractionPlanTest, RejectsSingleOperandUnaryReduction) {
  auto expression =
      EinsumExpression::create({access({0})}, access({}), {{0, 3}});
  ASSERT_TRUE(static_cast<bool>(expression));

  ContractionPlan plan(1, {}, 0);
  expectInvalid(*expression, plan);
}

TEST(ContractionPlanTest, RejectsIncompletePlan) {
  auto expression =
      EinsumExpression::create({access({0}), access({1}), access({2})},
                               access({0, 1, 2}), {{0, 3}, {1, 5}, {2, 7}});
  ASSERT_TRUE(static_cast<bool>(expression));

  ContractionPlan plan(3, {step(0, 1, {0, 1})}, 3);
  expectInvalid(*expression, plan);
}

TEST(ContractionPlanTest, RejectsNonrootResult) {
  auto expression =
      EinsumExpression::create({access({0}), access({1}), access({2})},
                               access({0, 1, 2}), {{0, 3}, {1, 5}, {2, 7}});
  ASSERT_TRUE(static_cast<bool>(expression));

  ContractionPlan plan(3, {step(0, 1, {0, 1}), step(3, 2, {0, 1, 2})}, 3);
  expectInvalid(*expression, plan);
}

TEST(ContractionPlanTest, RejectsOperandCountMismatch) {
  auto expression = EinsumExpression::create({access({0}), access({1})},
                                             access({0, 1}), {{0, 3}, {1, 5}});
  ASSERT_TRUE(static_cast<bool>(expression));

  ContractionPlan plan(1, {}, 0);
  expectInvalid(*expression, plan);
}

} // namespace
} // namespace mlir::lapis::contraction_plan_tests

namespace mlir::lapis::contraction_cost_tests {
namespace {

TensorAccess access(std::initializer_list<IndexId> indices) {
  return TensorAccess{llvm::SmallVector<IndexId, 4>(indices)};
}

ContractionStep step(PlanValueId lhs, PlanValueId rhs,
                     std::initializer_list<IndexId> resultIndices) {
  return {lhs, rhs,
          llvm::SmallVector<IndexId, 4>(resultIndices.begin(),
                                        resultIndices.end())};
}

template <typename T> void expectInvalid(llvm::Expected<T> value) {
  EXPECT_FALSE(static_cast<bool>(value));
  if (!value)
    llvm::consumeError(value.takeError());
}

TEST(ContractionCostTest, CountsSharedIndexOnce) {
  auto expression =
      EinsumExpression::create({access({0, 2}), access({2, 1})}, access({0, 1}),
                               {{0, 2}, {1, 5}, {2, 3}});
  ASSERT_TRUE(static_cast<bool>(expression));
  auto model = ContractionCostModel::create(*expression);
  ASSERT_TRUE(static_cast<bool>(model));

  auto work = model->getContractionWork(0b01, 0b10);
  ASSERT_TRUE(static_cast<bool>(work));
  EXPECT_EQ(*work, 30u);
}

TEST(ContractionCostTest, CountsRepeatedLeafIndexOnce) {
  auto expression = EinsumExpression::create({access({0, 0}), access({})},
                                             access({}), {{0, 4}});
  ASSERT_TRUE(static_cast<bool>(expression));
  auto model = ContractionCostModel::create(*expression);
  ASSERT_TRUE(static_cast<bool>(model));

  auto work = model->getContractionWork(0b01, 0b10);
  ASSERT_TRUE(static_cast<bool>(work));
  EXPECT_EQ(*work, 4u);
}

TEST(ContractionCostTest, ScalarContractionHasUnitWork) {
  auto expression =
      EinsumExpression::create({access({}), access({})}, access({}), {});
  ASSERT_TRUE(static_cast<bool>(expression));
  auto model = ContractionCostModel::create(*expression);
  ASSERT_TRUE(static_cast<bool>(model));

  auto work = model->getContractionWork(0b01, 0b10);
  ASSERT_TRUE(static_cast<bool>(work));
  EXPECT_EQ(*work, 1u);
}

TEST(ContractionCostTest, RejectsOverlappingOrInvalidSubsets) {
  auto expression = EinsumExpression::create({access({0}), access({1})},
                                             access({0, 1}), {{0, 2}, {1, 3}});
  ASSERT_TRUE(static_cast<bool>(expression));
  auto model = ContractionCostModel::create(*expression);
  ASSERT_TRUE(static_cast<bool>(model));

  expectInvalid(model->getContractionWork(0b01, 0b01));
  expectInvalid(model->getContractionWork(0, 0b01));
  expectInvalid(model->getContractionWork(0b01, 0b100));
}

TEST(ContractionCostTest, DetectsStepProductOverflow) {
  constexpr int64_t largeExtent = int64_t{1} << 32;
  auto expression =
      EinsumExpression::create({access({0}), access({1})}, access({0, 1}),
                               {{0, largeExtent}, {1, largeExtent}});
  ASSERT_TRUE(static_cast<bool>(expression));
  auto model = ContractionCostModel::create(*expression);
  ASSERT_TRUE(static_cast<bool>(model));

  expectInvalid(model->getContractionWork(0b01, 0b10));
}

TEST(ContractionCostTest, DetectsPlanTotalOverflow) {
  constexpr int64_t largeExtent = std::numeric_limits<int64_t>::max();
  auto expression = EinsumExpression::create(
      {access({0}), access({}), access({}), access({})}, access({0}),
      {{0, largeExtent}});
  ASSERT_TRUE(static_cast<bool>(expression));
  auto model = ContractionCostModel::create(*expression);
  ASSERT_TRUE(static_cast<bool>(model));

  ContractionPlan plan(4, {step(0, 1, {0}), step(4, 2, {0}), step(5, 3, {0})},
                       6);
  expectInvalid(model->evaluatePlan(plan));
}

TEST(ContractionCostTest, RefusesToEvaluateInvalidPlan) {
  auto expression = EinsumExpression::create({access({0}), access({1})},
                                             access({0, 1}), {{0, 2}, {1, 3}});
  ASSERT_TRUE(static_cast<bool>(expression));
  auto model = ContractionCostModel::create(*expression);
  ASSERT_TRUE(static_cast<bool>(model));

  ContractionPlan invalidPlan(2, {step(0, 1, {0})}, 2);
  expectInvalid(model->evaluatePlan(invalidPlan));
}

} // namespace
} // namespace mlir::lapis::contraction_cost_tests

namespace mlir::lapis::contraction_planner_tests {
namespace {

TensorAccess access(std::initializer_list<IndexId> indices) {
  return TensorAccess{llvm::SmallVector<IndexId, 4>(indices)};
}

template <typename T> void expectInvalid(llvm::Expected<T> value) {
  EXPECT_FALSE(static_cast<bool>(value));
  if (!value)
    llvm::consumeError(value.takeError());
}

void expectIndices(llvm::ArrayRef<IndexId> actual,
                   std::initializer_list<IndexId> expected) {
  EXPECT_TRUE(llvm::equal(actual, expected));
}

void expectValid(const EinsumExpression &expression,
                 const ContractionPlan &plan) {
  if (llvm::Error error = verifyContractionPlan(expression, plan))
    ADD_FAILURE() << llvm::toString(std::move(error));
}

TEST(ExactContractionPlannerTest, FindsGlobalOptimumThatDefeatsGreedy) {
  // Matrix chain dimensions [20, 3, 32, 32, 2]. Choosing AB first has
  // immediate cost 1920 but leads to total work 5248. The global optimum is
  // A(B(CD)), with total work 2048 + 192 + 120 = 2360.
  auto expression = EinsumExpression::create(
      {access({0, 1}), access({1, 2}), access({2, 3}), access({3, 4})},
      access({0, 4}), {{0, 20}, {1, 3}, {2, 32}, {3, 32}, {4, 2}});
  ASSERT_TRUE(static_cast<bool>(expression));

  auto planner = ExactContractionPlanner::create(*expression);
  ASSERT_TRUE(static_cast<bool>(planner));
  auto plan = planner->plan();
  ASSERT_TRUE(static_cast<bool>(plan));

  ASSERT_EQ(plan->getSteps().size(), 3u);
  EXPECT_EQ(plan->getSteps()[0].lhs, 2u);
  EXPECT_EQ(plan->getSteps()[0].rhs, 3u);
  expectIndices(plan->getSteps()[0].resultIndices, {4, 2});
  EXPECT_EQ(plan->getSteps()[1].lhs, 1u);
  EXPECT_EQ(plan->getSteps()[1].rhs, 4u);
  expectIndices(plan->getSteps()[1].resultIndices, {4, 1});
  EXPECT_EQ(plan->getSteps()[2].lhs, 0u);
  EXPECT_EQ(plan->getSteps()[2].rhs, 5u);
  expectIndices(plan->getSteps()[2].resultIndices, {0, 4});

  auto costModel = ContractionCostModel::create(*expression);
  ASSERT_TRUE(static_cast<bool>(costModel));
  auto cost = costModel->evaluatePlan(*plan);
  ASSERT_TRUE(static_cast<bool>(cost));
  EXPECT_EQ(cost->stepCosts, (llvm::SmallVector<WorkCost, 4>{2048, 192, 120}));
  EXPECT_EQ(cost->totalWork, 2360u);
}

TEST(ExactContractionPlannerTest, OptimizesABx) {
  auto expression =
      EinsumExpression::create({access({0, 2}), access({2, 1}), access({1})},
                               access({0}), {{0, 2}, {1, 5}, {2, 3}});
  ASSERT_TRUE(static_cast<bool>(expression));

  auto planner = ExactContractionPlanner::create(*expression);
  ASSERT_TRUE(static_cast<bool>(planner));
  auto plan = planner->plan();
  ASSERT_TRUE(static_cast<bool>(plan));

  ASSERT_EQ(plan->getSteps().size(), 2u);
  EXPECT_EQ(plan->getSteps()[0].lhs, 1u);
  EXPECT_EQ(plan->getSteps()[0].rhs, 2u);
  expectIndices(plan->getSteps()[0].resultIndices, {2});
  EXPECT_EQ(plan->getSteps()[1].lhs, 0u);
  EXPECT_EQ(plan->getSteps()[1].rhs, 3u);
  expectIndices(plan->getSteps()[1].resultIndices, {0});

  auto costModel = ContractionCostModel::create(*expression);
  ASSERT_TRUE(static_cast<bool>(costModel));
  auto cost = costModel->evaluatePlan(*plan);
  ASSERT_TRUE(static_cast<bool>(cost));
  EXPECT_EQ(cost->totalWork, 21u);
}

TEST(ExactContractionPlannerTest,
     PreservesRequiredIntermediateSubsetAndIndexOrder) {
  auto expression =
      EinsumExpression::create({access({0, 2}), access({2, 1}), access({1})},
                               access({0}), {{0, 2}, {1, 5}, {2, 3}});
  ASSERT_TRUE(static_cast<bool>(expression));

  auto planner = ExactContractionPlanner::create(*expression);
  ASSERT_TRUE(static_cast<bool>(planner));
  ContractionResultConstraint requiredAB{/*operands=*/0b011,
                                         /*access=*/access({1, 0})};
  auto plan = planner->plan({requiredAB});
  ASSERT_TRUE(static_cast<bool>(plan));

  auto ab = findPlanValueForOperandSubset(*plan, 0b011);
  ASSERT_TRUE(static_cast<bool>(ab));
  ASSERT_GE(*ab, plan->getOperandCount());
  expectIndices(plan->getSteps()[*ab - plan->getOperandCount()].resultIndices,
                {1, 0});

  auto costModel = ContractionCostModel::create(*expression);
  ASSERT_TRUE(static_cast<bool>(costModel));
  auto cost = costModel->evaluatePlan(*plan);
  ASSERT_TRUE(static_cast<bool>(cost));
  EXPECT_EQ(cost->totalWork, 40u);
}

TEST(ExactContractionPlannerTest, RejectsPartiallyOverlappingRequirements) {
  auto expression = EinsumExpression::create(
      {access({}), access({}), access({}), access({})}, access({}), {});
  ASSERT_TRUE(static_cast<bool>(expression));

  auto planner = ExactContractionPlanner::create(*expression);
  ASSERT_TRUE(static_cast<bool>(planner));
  expectInvalid(
      planner->plan({ContractionResultConstraint{0b0011, access({})},
                     ContractionResultConstraint{0b0110, access({})}}));
}

TEST(ExactContractionPlannerTest, UsesDeterministicSubsetTieBreaking) {
  auto expression = EinsumExpression::create(
      {access({}), access({}), access({})}, access({}), {});
  ASSERT_TRUE(static_cast<bool>(expression));

  auto planner = ExactContractionPlanner::create(*expression);
  ASSERT_TRUE(static_cast<bool>(planner));
  auto firstPlan = planner->plan();
  ASSERT_TRUE(static_cast<bool>(firstPlan));
  auto secondPlan = planner->plan();
  ASSERT_TRUE(static_cast<bool>(secondPlan));

  for (const ContractionPlan *plan : {&*firstPlan, &*secondPlan}) {
    ASSERT_EQ(plan->getSteps().size(), 2u);
    EXPECT_EQ(plan->getSteps()[0].lhs, 1u);
    EXPECT_EQ(plan->getSteps()[0].rhs, 2u);
    EXPECT_EQ(plan->getSteps()[1].lhs, 0u);
    EXPECT_EQ(plan->getSteps()[1].rhs, 3u);
    EXPECT_EQ(plan->getResult(), 4u);
  }
}

TEST(ExactContractionPlannerTest, PreservesThreeWaySharedIndexLiveness) {
  auto expression =
      EinsumExpression::create({access({0, 2}), access({2, 1}), access({2})},
                               access({0, 1}), {{0, 3}, {1, 5}, {2, 7}});
  ASSERT_TRUE(static_cast<bool>(expression));

  auto planner = ExactContractionPlanner::create(*expression);
  ASSERT_TRUE(static_cast<bool>(planner));
  auto plan = planner->plan();
  ASSERT_TRUE(static_cast<bool>(plan));

  expectValid(*expression, *plan);
  auto subsets = computePlanOperandSubsets(*plan);
  ASSERT_TRUE(static_cast<bool>(subsets));
  for (const auto &[stepNumber, step] : llvm::enumerate(plan->getSteps())) {
    OperandSubset subset = (*subsets)[plan->getStepResult(stepNumber)];
    if (subset != OperandSubset{0b111})
      EXPECT_TRUE(llvm::is_contained(step.resultIndices, IndexId{2}));
  }
}

TEST(ExactContractionPlannerTest, HandlesSingleOperandIdentity) {
  auto expression =
      EinsumExpression::create({access({0})}, access({0}), {{0, 4}});
  ASSERT_TRUE(static_cast<bool>(expression));

  auto planner = ExactContractionPlanner::create(*expression);
  ASSERT_TRUE(static_cast<bool>(planner));
  auto plan = planner->plan();
  ASSERT_TRUE(static_cast<bool>(plan));
  EXPECT_TRUE(plan->getSteps().empty());
  EXPECT_EQ(plan->getResult(), 0u);
}

TEST(ExactContractionPlannerTest, RejectsSingleOperandUnaryReduction) {
  auto expression =
      EinsumExpression::create({access({0})}, access({}), {{0, 4}});
  ASSERT_TRUE(static_cast<bool>(expression));

  auto planner = ExactContractionPlanner::create(*expression);
  ASSERT_TRUE(static_cast<bool>(planner));
  expectInvalid(planner->plan());
}

TEST(ExactContractionPlannerTest, RejectsExpressionsAboveExactLimit) {
  llvm::SmallVector<TensorAccess, 4> operands(kMaxExactPlannerOperands + 1,
                                              access({}));
  auto expression = EinsumExpression::create(operands, access({}), {});
  ASSERT_TRUE(static_cast<bool>(expression));

  expectInvalid(ExactContractionPlanner::create(*expression));
}

TEST(ExactContractionPlannerTest, ReportsUnrepresentableWork) {
  constexpr int64_t largeExtent = int64_t{1} << 32;
  auto expression =
      EinsumExpression::create({access({0}), access({1})}, access({0, 1}),
                               {{0, largeExtent}, {1, largeExtent}});
  ASSERT_TRUE(static_cast<bool>(expression));

  auto planner = ExactContractionPlanner::create(*expression);
  ASSERT_TRUE(static_cast<bool>(planner));
  expectInvalid(planner->plan());
}

TEST(ExactContractionPlannerTest,
     SkipsOverflowingCandidateWhenAnotherPlanIsRepresentable) {
  constexpr int64_t largeExtent = int64_t{1} << 32;
  auto expression = EinsumExpression::create(
      {access({0}), access({1}), access({})}, access({}),
      {{0, largeExtent}, {1, largeExtent}});
  ASSERT_TRUE(static_cast<bool>(expression));

  auto planner = ExactContractionPlanner::create(*expression);
  ASSERT_TRUE(static_cast<bool>(planner));
  auto plan = planner->plan();
  ASSERT_TRUE(static_cast<bool>(plan));

  auto costModel = ContractionCostModel::create(*expression);
  ASSERT_TRUE(static_cast<bool>(costModel));
  auto cost = costModel->evaluatePlan(*plan);
  ASSERT_TRUE(static_cast<bool>(cost));
  EXPECT_EQ(cost->totalWork, WorkCost{1} << 33);
}

TEST(ExactContractionPlannerTest, ReportsUnrepresentableTotalWork) {
  constexpr int64_t largeExtent = std::numeric_limits<int64_t>::max();
  auto expression = EinsumExpression::create(
      {access({0}), access({0}), access({0}), access({0})}, access({0}),
      {{0, largeExtent}});
  ASSERT_TRUE(static_cast<bool>(expression));

  auto planner = ExactContractionPlanner::create(*expression);
  ASSERT_TRUE(static_cast<bool>(planner));
  expectInvalid(planner->plan());
}

} // namespace
} // namespace mlir::lapis::contraction_planner_tests

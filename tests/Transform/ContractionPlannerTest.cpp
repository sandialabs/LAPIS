#include "lapis/Transform/ContractionPlanner.h"
#include "lapis/Transform/AlgebraicKernelFusion.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "gtest/gtest.h"

#include <initializer_list>

namespace mlir::lapis {
namespace {

TensorAccess access(std::initializer_list<IndexId> indices) {
  return TensorAccess{llvm::SmallVector<IndexId, 4>(indices)};
}

void expectIndices(llvm::ArrayRef<IndexId> actual,
                   std::initializer_list<IndexId> expected) {
  EXPECT_TRUE(llvm::equal(actual, expected));
}

TEST(ContractionPlannerTest, FindsAndVerifiesGlobalMinimum) {
  // Matrix-chain dimensions [20, 3, 32, 32, 2]. A greedy AB-first choice
  // costs 5248 work units; exact subset DP finds A(B(CD)) at cost 2360.
  auto expression = EinsumExpression::create(
      {access({0, 1}), access({1, 2}), access({2, 3}), access({3, 4})},
      access({0, 4}), {{0, 20}, {1, 3}, {2, 32}, {3, 32}, {4, 2}});
  ASSERT_TRUE(static_cast<bool>(expression));

  auto liveness = IndexLiveness::create(*expression);
  ASSERT_TRUE(static_cast<bool>(liveness));
  auto cdIndices = liveness->getLiveIndices(OperandSubset{0b1100});
  ASSERT_TRUE(static_cast<bool>(cdIndices));
  expectIndices(*cdIndices, {4, 2});

  auto planner = ExactContractionPlanner::create(*expression);
  ASSERT_TRUE(static_cast<bool>(planner));
  auto plan = planner->plan();
  ASSERT_TRUE(static_cast<bool>(plan));

  if (llvm::Error error = verifyContractionPlan(*expression, *plan))
    ADD_FAILURE() << llvm::toString(std::move(error));

  ASSERT_EQ(plan->getSteps().size(), 3u);
  EXPECT_EQ(plan->getSteps()[0].lhs, 2u);
  EXPECT_EQ(plan->getSteps()[0].rhs, 3u);
  expectIndices(plan->getSteps()[0].resultIndices, {4, 2});
  expectIndices(plan->getSteps()[1].resultIndices, {4, 1});
  expectIndices(plan->getSteps()[2].resultIndices, {0, 4});

  auto subsets = computePlanOperandSubsets(*plan);
  ASSERT_TRUE(static_cast<bool>(subsets));
  EXPECT_EQ((*subsets)[plan->getStepResult(0)], OperandSubset{0b1100});

  auto costModel = ContractionCostModel::create(*expression);
  ASSERT_TRUE(static_cast<bool>(costModel));
  auto cost = costModel->evaluatePlan(*plan);
  ASSERT_TRUE(static_cast<bool>(cost));
  EXPECT_EQ(cost->stepCosts, (llvm::SmallVector<WorkCost, 4>{2048, 192, 120}));
  EXPECT_EQ(cost->stepResultVolumes,
            (llvm::SmallVector<StorageVolume, 4>{64, 6, 40}));
  EXPECT_EQ(cost->totalWork, 2360u);
  EXPECT_EQ(cost->totalIntermediateVolume, 70u);

  // The independent verifier must reject a plan whose advertised result is an
  // intermediate rather than the full contraction root.
  ContractionPlan invalid(plan->getOperandCount(), plan->getSteps(),
                          plan->getStepResult(1));
  llvm::Error error = verifyContractionPlan(*expression, invalid);
  EXPECT_TRUE(static_cast<bool>(error));
  if (error)
    llvm::consumeError(std::move(error));
}

TEST(ContractionPlannerTest, BreaksEqualWorkTiesByTemporaryVolume) {
  // For dimensions [3, 2, 3, 6], both (AB)C and A(BC) cost 72 work units.
  // Their intermediates contain 9 and 12 elements respectively.
  auto expression = EinsumExpression::create(
      {access({0, 1}), access({1, 2}), access({2, 3})}, access({0, 3}),
      {{0, 3}, {1, 2}, {2, 3}, {3, 6}});
  ASSERT_TRUE(static_cast<bool>(expression));

  auto planner = ExactContractionPlanner::create(*expression);
  ASSERT_TRUE(static_cast<bool>(planner));
  auto plan = planner->plan();
  ASSERT_TRUE(static_cast<bool>(plan));

  ASSERT_EQ(plan->getSteps().size(), 2u);
  EXPECT_EQ(plan->getSteps()[0].lhs, 0u);
  EXPECT_EQ(plan->getSteps()[0].rhs, 1u);
  expectIndices(plan->getSteps()[0].resultIndices, {0, 2});

  auto costModel = ContractionCostModel::create(*expression);
  ASSERT_TRUE(static_cast<bool>(costModel));
  auto selectedCost = costModel->evaluatePlan(*plan);
  ASSERT_TRUE(static_cast<bool>(selectedCost));
  EXPECT_EQ(selectedCost->totalWork, 72u);
  EXPECT_EQ(selectedCost->totalIntermediateVolume, 9u);

  ContractionPlan largerTemporary(
      /*operandCount=*/3,
      {ContractionStep{/*lhs=*/1, /*rhs=*/2, /*resultIndices=*/{3, 1}},
       ContractionStep{/*lhs=*/0, /*rhs=*/3, /*resultIndices=*/{0, 3}}},
      /*result=*/4);
  auto alternativeCost = costModel->evaluatePlan(largerTemporary);
  ASSERT_TRUE(static_cast<bool>(alternativeCost));
  EXPECT_EQ(alternativeCost->totalWork, 72u);
  EXPECT_EQ(alternativeCost->totalIntermediateVolume, 12u);
}

TEST(FusionCandidateEvaluationTest, OrdersWorkBeforeTemporaryVolume) {
  EXPECT_TRUE((FusionCandidateEvaluation{/*originalWork=*/100,
                                         /*optimizedWork=*/99,
                                         /*originalTemporaryVolume=*/1,
                                         /*optimizedTemporaryVolume=*/1000})
                  .shouldRewrite());
  EXPECT_TRUE((FusionCandidateEvaluation{/*originalWork=*/100,
                                         /*optimizedWork=*/100,
                                         /*originalTemporaryVolume=*/12,
                                         /*optimizedTemporaryVolume=*/9})
                  .shouldRewrite());
  EXPECT_FALSE((FusionCandidateEvaluation{/*originalWork=*/100,
                                          /*optimizedWork=*/100,
                                          /*originalTemporaryVolume=*/9,
                                          /*optimizedTemporaryVolume=*/9})
                   .shouldRewrite());
}

} // namespace
} // namespace mlir::lapis

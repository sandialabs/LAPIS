#include "lapis/Transform/ContractionPlanner.h"

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
  EXPECT_EQ(cost->stepCosts,
            (llvm::SmallVector<WorkCost, 4>{2048, 192, 120}));
  EXPECT_EQ(cost->totalWork, 2360u);

  // The independent verifier must reject a plan whose advertised result is an
  // intermediate rather than the full contraction root.
  ContractionPlan invalid(plan->getOperandCount(), plan->getSteps(),
                          plan->getStepResult(1));
  llvm::Error error = verifyContractionPlan(*expression, invalid);
  EXPECT_TRUE(static_cast<bool>(error));
  if (error)
    llvm::consumeError(std::move(error));
}

} // namespace
} // namespace mlir::lapis

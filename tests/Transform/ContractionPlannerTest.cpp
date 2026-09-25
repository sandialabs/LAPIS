#include "lapis/Transform/ContractionPlanner.h"
#include "TestSupport.h"
#include "lapis/Transform/AlgebraicKernelFusion.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <initializer_list>

namespace mlir::lapis {
namespace {

TensorAccess access(std::initializer_list<IndexId> indices) {
  return TensorAccess{llvm::SmallVector<IndexId, 4>(indices)};
}

void expectIndices(test::TestContext &context, llvm::ArrayRef<IndexId> actual,
                   std::initializer_list<IndexId> expected) {
  LAPIS_CHECK(context, llvm::equal(actual, expected));
}

bool findsAndVerifiesGlobalMinimum(test::TestContext &context) {
  // Matrix-chain dimensions [20, 3, 32, 32, 2]. A greedy AB-first choice
  // costs 5248 work units; exact subset DP finds A(B(CD)) at cost 2360.
  auto expression = EinsumExpression::create(
      {access({0, 1}), access({1, 2}), access({2, 3}), access({3, 4})},
      access({0, 4}), {{0, 20}, {1, 3}, {2, 32}, {3, 32}, {4, 2}});
  LAPIS_REQUIRE_EXPECTED(context, expression);

  auto liveness = IndexLiveness::create(*expression);
  LAPIS_REQUIRE_EXPECTED(context, liveness);
  auto cdIndices = liveness->getLiveIndices(OperandSubset{0b1100});
  LAPIS_REQUIRE_EXPECTED(context, cdIndices);
  expectIndices(context, *cdIndices, {4, 2});

  auto planner = ExactContractionPlanner::create(*expression);
  LAPIS_REQUIRE_EXPECTED(context, planner);
  auto plan = planner->plan();
  LAPIS_REQUIRE_EXPECTED(context, plan);

  if (llvm::Error error = verifyContractionPlan(*expression, *plan)) {
    context.fail(llvm::toString(std::move(error)), __FILE__, __LINE__);
    return false;
  }

  LAPIS_REQUIRE(context, plan->getSteps().size() == 3u);
  LAPIS_CHECK(context, plan->getSteps()[0].lhs == 2u);
  LAPIS_CHECK(context, plan->getSteps()[0].rhs == 3u);
  expectIndices(context, plan->getSteps()[0].resultIndices, {4, 2});
  expectIndices(context, plan->getSteps()[1].resultIndices, {4, 1});
  expectIndices(context, plan->getSteps()[2].resultIndices, {0, 4});

  auto subsets = computePlanOperandSubsets(*plan);
  LAPIS_REQUIRE_EXPECTED(context, subsets);
  LAPIS_CHECK(context,
              (*subsets)[plan->getStepResult(0)] == OperandSubset{0b1100});

  auto costModel = ContractionCostModel::create(*expression);
  LAPIS_REQUIRE_EXPECTED(context, costModel);
  auto cost = costModel->evaluatePlan(*plan);
  LAPIS_REQUIRE_EXPECTED(context, cost);
  LAPIS_CHECK(context, cost->stepCosts ==
                           (llvm::SmallVector<WorkCost, 4>{2048, 192, 120}));
  LAPIS_CHECK(context, cost->stepResultVolumes ==
                           (llvm::SmallVector<StorageVolume, 4>{64, 6, 40}));
  LAPIS_CHECK(context, cost->totalWork == 2360u);
  LAPIS_CHECK(context, cost->totalIntermediateVolume == 70u);

  // The independent verifier must reject a plan whose advertised result is an
  // intermediate rather than the full contraction root.
  ContractionPlan invalid(plan->getOperandCount(), plan->getSteps(),
                          plan->getStepResult(1));
  llvm::Error error = verifyContractionPlan(*expression, invalid);
  if (!error) {
    context.fail("invalid plan unexpectedly verified", __FILE__, __LINE__);
    return false;
  }
  llvm::consumeError(std::move(error));
  return true;
}

bool breaksEqualWorkTiesByTemporaryVolume(test::TestContext &context) {
  // For dimensions [3, 2, 3, 6], both (AB)C and A(BC) cost 72 work units.
  // Their intermediates contain 9 and 12 elements respectively.
  auto expression = EinsumExpression::create(
      {access({0, 1}), access({1, 2}), access({2, 3})}, access({0, 3}),
      {{0, 3}, {1, 2}, {2, 3}, {3, 6}});
  LAPIS_REQUIRE_EXPECTED(context, expression);

  auto planner = ExactContractionPlanner::create(*expression);
  LAPIS_REQUIRE_EXPECTED(context, planner);
  auto plan = planner->plan();
  LAPIS_REQUIRE_EXPECTED(context, plan);

  LAPIS_REQUIRE(context, plan->getSteps().size() == 2u);
  LAPIS_CHECK(context, plan->getSteps()[0].lhs == 0u);
  LAPIS_CHECK(context, plan->getSteps()[0].rhs == 1u);
  expectIndices(context, plan->getSteps()[0].resultIndices, {0, 2});

  auto costModel = ContractionCostModel::create(*expression);
  LAPIS_REQUIRE_EXPECTED(context, costModel);
  auto selectedCost = costModel->evaluatePlan(*plan);
  LAPIS_REQUIRE_EXPECTED(context, selectedCost);
  LAPIS_CHECK(context, selectedCost->totalWork == 72u);
  LAPIS_CHECK(context, selectedCost->totalIntermediateVolume == 9u);

  ContractionPlan largerTemporary(
      /*operandCount=*/3,
      {ContractionStep{/*lhs=*/1, /*rhs=*/2, /*resultIndices=*/{3, 1}},
       ContractionStep{/*lhs=*/0, /*rhs=*/3, /*resultIndices=*/{0, 3}}},
      /*result=*/4);
  auto alternativeCost = costModel->evaluatePlan(largerTemporary);
  LAPIS_REQUIRE_EXPECTED(context, alternativeCost);
  LAPIS_CHECK(context, alternativeCost->totalWork == 72u);
  LAPIS_CHECK(context, alternativeCost->totalIntermediateVolume == 12u);
  return true;
}

bool ordersWorkBeforeTemporaryVolume(test::TestContext &context) {
  LAPIS_CHECK(context,
              (FusionCandidateEvaluation{/*originalWork=*/100,
                                         /*optimizedWork=*/99,
                                         /*originalTemporaryVolume=*/1,
                                         /*optimizedTemporaryVolume=*/1000})
                  .shouldRewrite());
  LAPIS_CHECK(context,
              (FusionCandidateEvaluation{/*originalWork=*/100,
                                         /*optimizedWork=*/100,
                                         /*originalTemporaryVolume=*/12,
                                         /*optimizedTemporaryVolume=*/9})
                  .shouldRewrite());
  LAPIS_CHECK(context,
              !(FusionCandidateEvaluation{/*originalWork=*/100,
                                          /*optimizedWork=*/100,
                                          /*originalTemporaryVolume=*/9,
                                          /*optimizedTemporaryVolume=*/9})
                   .shouldRewrite());
  return true;
}

} // namespace

void runContractionPlannerTests(test::TestContext &context) {
  context.run("ContractionPlannerTest.FindsAndVerifiesGlobalMinimum",
              findsAndVerifiesGlobalMinimum);
  context.run("ContractionPlannerTest.BreaksEqualWorkTiesByTemporaryVolume",
              breaksEqualWorkTiesByTemporaryVolume);
  context.run("FusionCandidateEvaluationTest.OrdersWorkBeforeTemporaryVolume",
              ordersWorkBeforeTemporaryVolume);
}

} // namespace mlir::lapis

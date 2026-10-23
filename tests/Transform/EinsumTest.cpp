#include "lapis/Transform/Einsum.h"
#include "TestSupport.h"

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

bool validatesAndComposesExpressions(test::TestContext &context) {
  // The producer's reduction index collides with a consumer index and must be
  // alpha-renamed during composition.
  auto producer =
      EinsumExpression::create({access({0, 1}), access({1, 2})}, access({0, 2}),
                               {{0, 2}, {1, 7}, {2, 5}});
  auto consumer =
      EinsumExpression::create({access({3, 1}), access({1, 0})}, access({3, 0}),
                               {{3, 2}, {1, 5}, {0, 11}});
  LAPIS_REQUIRE_EXPECTED(context, producer);
  LAPIS_REQUIRE_EXPECTED(context, consumer);

  expectIndices(context, producer->getIndexOrder(), {0, 1, 2});
  expectIndices(context, producer->getReductionIndices(), {1});
  LAPIS_CHECK(context, producer->getExtent(1) == 7);

  auto composition =
      composeEinsumExpressionsWithProvenance(*producer, *consumer, 0);
  LAPIS_REQUIRE_EXPECTED(context, composition);
  const EinsumExpression &composed = composition->getExpression();

  LAPIS_REQUIRE(context, composed.getOperands().size() == 3u);
  const IndexId renamedReduction = composed.getOperands()[0].indices[1];
  LAPIS_CHECK(context, renamedReduction != 1u);
  LAPIS_CHECK(context, composed.getExtent(renamedReduction) == 7);
  expectIndices(context, composed.getOperands()[0].indices,
                {3, renamedReduction});
  expectIndices(context, composed.getOperands()[1].indices,
                {renamedReduction, 1});
  expectIndices(context, composed.getOperands()[2].indices, {1, 0});
  expectIndices(context, composed.getResult().indices, {3, 0});
  expectIndices(context,
                composition->remapProducerAccess(access({2, 1, 0})).indices,
                {1, renamedReduction, 3});

  // Keep one representative validation failure alongside the valid path.
  auto invalid =
      EinsumExpression::create({access({0})}, access({0}), {{0, 4}, {0, 8}});
  if (invalid) {
    context.fail("invalid expression was unexpectedly accepted", __FILE__,
                 __LINE__);
    return false;
  }
  llvm::consumeError(invalid.takeError());
  return true;
}

} // namespace

void runEinsumTests(test::TestContext &context) {
  context.run("EinsumTest.ValidatesAndComposesExpressions",
              validatesAndComposesExpressions);
}

} // namespace mlir::lapis

#include "lapis/Transform/Einsum.h"

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

TEST(EinsumTest, ValidatesAndComposesExpressions) {
  // The producer's reduction index collides with a consumer index and must be
  // alpha-renamed during composition.
  auto producer =
      EinsumExpression::create({access({0, 1}), access({1, 2})}, access({0, 2}),
                               {{0, 2}, {1, 7}, {2, 5}});
  auto consumer =
      EinsumExpression::create({access({3, 1}), access({1, 0})}, access({3, 0}),
                               {{3, 2}, {1, 5}, {0, 11}});
  ASSERT_TRUE(static_cast<bool>(producer));
  ASSERT_TRUE(static_cast<bool>(consumer));

  expectIndices(producer->getIndexOrder(), {0, 1, 2});
  expectIndices(producer->getReductionIndices(), {1});
  EXPECT_EQ(producer->getExtent(1), 7);

  auto composition =
      composeEinsumExpressionsWithProvenance(*producer, *consumer, 0);
  ASSERT_TRUE(static_cast<bool>(composition));
  const EinsumExpression &composed = composition->getExpression();

  ASSERT_EQ(composed.getOperands().size(), 3u);
  const IndexId renamedReduction = composed.getOperands()[0].indices[1];
  EXPECT_NE(renamedReduction, 1u);
  EXPECT_EQ(composed.getExtent(renamedReduction), 7);
  expectIndices(composed.getOperands()[0].indices, {3, renamedReduction});
  expectIndices(composed.getOperands()[1].indices, {renamedReduction, 1});
  expectIndices(composed.getOperands()[2].indices, {1, 0});
  expectIndices(composed.getResult().indices, {3, 0});
  expectIndices(composition->remapProducerAccess(access({2, 1, 0})).indices,
                {1, renamedReduction, 3});

  // Keep one representative validation failure alongside the valid path.
  auto invalid = EinsumExpression::create({access({0})}, access({0}),
                                          {{0, 4}, {0, 8}});
  EXPECT_FALSE(static_cast<bool>(invalid));
  if (!invalid)
    llvm::consumeError(invalid.takeError());
}

} // namespace
} // namespace mlir::lapis

#include "lapis/Transform/Einsum.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "gtest/gtest.h"
#include <initializer_list>
#include <string>

namespace mlir::lapis::einsum_expression_tests {
namespace {

TensorAccess access(std::initializer_list<IndexId> indices) {
  return TensorAccess{llvm::SmallVector<IndexId, 4>(indices)};
}

void expectIndices(llvm::ArrayRef<IndexId> actual,
                   std::initializer_list<IndexId> expected) {
  EXPECT_TRUE(llvm::equal(actual, expected));
}

void expectInvalid(llvm::Expected<EinsumExpression> expression) {
  EXPECT_FALSE(static_cast<bool>(expression));
  if (!expression)
    llvm::consumeError(expression.takeError());
}

TEST(EinsumExpressionTest, PreservesIndexOrderAndDerivesReductions) {
  const llvm::SmallVector<TensorAccess, 4> operands = {
      access({7, 2}), access({2, 9}), access({9})};
  const llvm::SmallVector<IndexExtent, 4> extents = {
      {2, 16}, {7, 8}, {9, 32}, {2, 16}};

  auto expression = EinsumExpression::create(operands, access({7}), extents);
  if (!expression) {
    FAIL() << llvm::toString(expression.takeError());
    return;
  }

  EXPECT_EQ(expression->getOperands().size(), 3u);
  expectIndices(expression->getIndexOrder(), {7, 2, 9});
  expectIndices(expression->getReductionIndices(), {2, 9});
  EXPECT_EQ(expression->getExtent(2), 16);
  EXPECT_EQ(expression->getExtent(42), std::nullopt);
}

TEST(EinsumExpressionTest, AcceptsRepeatedInputIndices) {
  auto expression = EinsumExpression::create({access({0, 0}), access({0})},
                                             access({}), {{0, 4}});
  if (!expression) {
    FAIL() << llvm::toString(expression.takeError());
    return;
  }

  expectIndices(expression->getIndexOrder(), {0});
  expectIndices(expression->getReductionIndices(), {0});
}

TEST(EinsumExpressionTest, AcceptsScalarExpression) {
  auto expression =
      EinsumExpression::create({access({}), access({})}, access({}), {});
  if (!expression) {
    FAIL() << llvm::toString(expression.takeError());
    return;
  }

  EXPECT_TRUE(expression->getIndexOrder().empty());
  EXPECT_TRUE(expression->getReductionIndices().empty());
  EXPECT_EQ(expression->getExtent(0), std::nullopt);
}

TEST(EinsumExpressionTest, PreservesResultIndexOrder) {
  auto expression =
      EinsumExpression::create({access({7, 2}), access({2, 9})}, access({9, 7}),
                               {{7, 8}, {2, 16}, {9, 32}});
  if (!expression) {
    FAIL() << llvm::toString(expression.takeError());
    return;
  }

  expectIndices(expression->getResult().indices, {9, 7});
  expectIndices(expression->getReductionIndices(), {2});
}

TEST(EinsumExpressionTest, OwnsItsConstructionData) {
  llvm::SmallVector<TensorAccess, 4> operands = {access({0, 1})};
  llvm::SmallVector<IndexExtent, 4> extents = {{0, 4}, {1, 8}};
  auto expression = EinsumExpression::create(operands, access({0}), extents);
  if (!expression) {
    FAIL() << llvm::toString(expression.takeError());
    return;
  }

  operands.front().indices.front() = 2;
  extents.front().extent = 16;

  expectIndices(expression->getOperands().front().indices, {0, 1});
  EXPECT_EQ(expression->getExtent(0), 4);
}

TEST(EinsumExpressionTest, SupportsMoreThanTwentySixIndices) {
  TensorAccess operand;
  llvm::SmallVector<IndexExtent, 32> extents;
  for (IndexId index = 0; index < 32; ++index) {
    operand.indices.push_back(index);
    extents.push_back({index, static_cast<int64_t>(index) + 1});
  }

  auto expression = EinsumExpression::create({operand}, access({31}), extents);
  if (!expression) {
    FAIL() << llvm::toString(expression.takeError());
    return;
  }

  EXPECT_EQ(expression->getIndexOrder().size(), 32u);
  EXPECT_EQ(expression->getReductionIndices().size(), 31u);
  EXPECT_EQ(expression->getExtent(31), 32);
}

TEST(EinsumExpressionTest, RejectsExpressionWithoutOperands) {
  expectInvalid(EinsumExpression::create({}, access({}), {}));
}

TEST(EinsumExpressionTest, RejectsConflictingExtents) {
  expectInvalid(
      EinsumExpression::create({access({0})}, access({0}), {{0, 4}, {0, 8}}));
}

TEST(EinsumExpressionTest, RejectsMissingExtent) {
  expectInvalid(
      EinsumExpression::create({access({0, 1})}, access({0}), {{0, 4}}));
}

TEST(EinsumExpressionTest, RejectsNonpositiveExtents) {
  expectInvalid(EinsumExpression::create({access({0})}, access({0}), {{0, 0}}));
  expectInvalid(
      EinsumExpression::create({access({0})}, access({0}), {{0, -4}}));
}

TEST(EinsumExpressionTest, RejectsDuplicateResultIndex) {
  expectInvalid(EinsumExpression::create({access({0, 1})}, access({0, 0}),
                                         {{0, 4}, {1, 8}}));
}

TEST(EinsumExpressionTest, RejectsResultIndexAbsentFromOperands) {
  expectInvalid(
      EinsumExpression::create({access({0})}, access({1}), {{0, 4}, {1, 8}}));
}

TEST(EinsumExpressionTest, RejectsUnusedExtent) {
  expectInvalid(
      EinsumExpression::create({access({0})}, access({0}), {{0, 4}, {1, 8}}));
}

} // namespace
} // namespace mlir::lapis::einsum_expression_tests

namespace mlir::lapis::einsum_composition_tests {
namespace {

TensorAccess access(std::initializer_list<IndexId> indices) {
  return TensorAccess{llvm::SmallVector<IndexId, 4>(indices)};
}

void expectIndices(llvm::ArrayRef<IndexId> actual,
                   std::initializer_list<IndexId> expected) {
  EXPECT_TRUE(llvm::equal(actual, expected));
}

void expectInvalid(llvm::Expected<EinsumExpression> expression) {
  EXPECT_FALSE(static_cast<bool>(expression));
  if (!expression)
    llvm::consumeError(expression.takeError());
}

TEST(EinsumCompositionTest, SubstitutesProducerAndPreservesOperandOrder) {
  auto producer =
      EinsumExpression::create({access({10, 12}), access({12, 11})},
                               access({10, 11}), {{10, 2}, {12, 3}, {11, 5}});
  auto consumer = EinsumExpression::create(
      {access({}), access({4, 8}), access({8})}, access({4}), {{4, 2}, {8, 5}});
  ASSERT_TRUE(static_cast<bool>(producer));
  ASSERT_TRUE(static_cast<bool>(consumer));

  auto composed = composeEinsumExpressions(*producer, *consumer, 1);
  if (!composed) {
    FAIL() << llvm::toString(composed.takeError());
    return;
  }

  ASSERT_EQ(composed->getOperands().size(), 4u);
  expectIndices(composed->getOperands()[0].indices, {});
  expectIndices(composed->getOperands()[1].indices, {4, 12});
  expectIndices(composed->getOperands()[2].indices, {12, 8});
  expectIndices(composed->getOperands()[3].indices, {8});
  expectIndices(composed->getResult().indices, {4});
  expectIndices(composed->getReductionIndices(), {12, 8});
}

TEST(EinsumCompositionTest, RenamesCollidingProducerReductionIndex) {
  auto producer =
      EinsumExpression::create({access({0, 1}), access({1, 2})}, access({0, 2}),
                               {{0, 2}, {1, 7}, {2, 5}});
  auto consumer =
      EinsumExpression::create({access({3, 1}), access({1, 0})}, access({3, 0}),
                               {{3, 2}, {1, 5}, {0, 11}});
  ASSERT_TRUE(static_cast<bool>(producer));
  ASSERT_TRUE(static_cast<bool>(consumer));

  auto composition =
      composeEinsumExpressionsWithProvenance(*producer, *consumer, 0);
  if (!composition) {
    FAIL() << llvm::toString(composition.takeError());
    return;
  }
  const EinsumExpression &composed = composition->getExpression();

  IndexId renamedReduction = composed.getOperands()[0].indices[1];
  EXPECT_EQ(composed.getOperands()[1].indices[0], renamedReduction);
  EXPECT_NE(renamedReduction, 0u);
  EXPECT_NE(renamedReduction, 1u);
  EXPECT_NE(renamedReduction, 3u);
  EXPECT_EQ(composed.getExtent(renamedReduction), 7);
  expectIndices(composed.getOperands()[0].indices, {3, renamedReduction});
  expectIndices(composed.getOperands()[1].indices, {renamedReduction, 1});
  expectIndices(composed.getOperands()[2].indices, {1, 0});

  TensorAccess internalAccess =
      composition->remapProducerAccess(access({2, 1, 0}));
  expectIndices(internalAccess.indices, {1, renamedReduction, 3});
}

TEST(EinsumCompositionTest, UnifiesAxesForRepeatedConsumerIndex) {
  auto producer = EinsumExpression::create({access({0, 1})}, access({0, 1}),
                                           {{0, 4}, {1, 4}});
  auto consumer =
      EinsumExpression::create({access({5, 5})}, access({}), {{5, 4}});
  ASSERT_TRUE(static_cast<bool>(producer));
  ASSERT_TRUE(static_cast<bool>(consumer));

  auto composed = composeEinsumExpressions(*producer, *consumer, 0);
  if (!composed) {
    FAIL() << llvm::toString(composed.takeError());
    return;
  }

  ASSERT_EQ(composed->getOperands().size(), 1u);
  expectIndices(composed->getOperands()[0].indices, {5, 5});
  expectIndices(composed->getReductionIndices(), {5});
}

TEST(EinsumCompositionTest, ComposesScalarProducer) {
  auto producer = EinsumExpression::create({access({0})}, access({}), {{0, 4}});
  auto consumer = EinsumExpression::create({access({}), access({7})},
                                           access({7}), {{7, 3}});
  ASSERT_TRUE(static_cast<bool>(producer));
  ASSERT_TRUE(static_cast<bool>(consumer));

  auto composed = composeEinsumExpressions(*producer, *consumer, 0);
  if (!composed) {
    FAIL() << llvm::toString(composed.takeError());
    return;
  }

  ASSERT_EQ(composed->getOperands().size(), 2u);
  expectIndices(composed->getOperands()[0].indices, {0});
  expectIndices(composed->getOperands()[1].indices, {7});
  expectIndices(composed->getResult().indices, {7});
  expectIndices(composed->getReductionIndices(), {0});
}

TEST(EinsumCompositionTest, RejectsOutOfRangeConsumerOperand) {
  auto producer =
      EinsumExpression::create({access({0})}, access({0}), {{0, 4}});
  auto consumer =
      EinsumExpression::create({access({1})}, access({1}), {{1, 4}});
  ASSERT_TRUE(static_cast<bool>(producer));
  ASSERT_TRUE(static_cast<bool>(consumer));

  expectInvalid(composeEinsumExpressions(*producer, *consumer, 1));
}

TEST(EinsumCompositionTest, RejectsRankMismatch) {
  auto producer = EinsumExpression::create({access({0, 1})}, access({0, 1}),
                                           {{0, 4}, {1, 8}});
  auto consumer =
      EinsumExpression::create({access({2})}, access({2}), {{2, 4}});
  ASSERT_TRUE(static_cast<bool>(producer));
  ASSERT_TRUE(static_cast<bool>(consumer));

  expectInvalid(composeEinsumExpressions(*producer, *consumer, 0));
}

TEST(EinsumCompositionTest, RejectsExtentMismatch) {
  auto producer =
      EinsumExpression::create({access({0})}, access({0}), {{0, 4}});
  auto consumer =
      EinsumExpression::create({access({1})}, access({1}), {{1, 8}});
  ASSERT_TRUE(static_cast<bool>(producer));
  ASSERT_TRUE(static_cast<bool>(consumer));

  expectInvalid(composeEinsumExpressions(*producer, *consumer, 0));
}

} // namespace
} // namespace mlir::lapis::einsum_composition_tests

namespace mlir::lapis::einsum_extraction_tests {
namespace {

class LinalgEinsumExtractionTest : public ::testing::Test {
protected:
  LinalgEinsumExtractionTest() {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    linalg::LinalgDialect, tensor::TensorDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  }

  linalg::GenericOp parseSingleGeneric(llvm::StringRef source) {
    module = parseSourceString<ModuleOp>(source, &context);
    if (!module) {
      ADD_FAILURE() << "failed to parse test module";
      return {};
    }

    llvm::SmallVector<linalg::GenericOp, 1> generics;
    module->walk(
        [&](linalg::GenericOp generic) { generics.push_back(generic); });
    if (generics.size() != 1) {
      ADD_FAILURE() << "expected exactly one linalg.generic, but found "
                    << generics.size();
      return {};
    }
    return generics.front();
  }

  template <typename T>
  void expectInvalid(llvm::Expected<T> value, llvm::StringRef message) {
    ASSERT_FALSE(static_cast<bool>(value));
    std::string error = llvm::toString(value.takeError());
    EXPECT_NE(error.find(message.str()), std::string::npos) << error;
  }

  void expectIndices(llvm::ArrayRef<IndexId> actual,
                     std::initializer_list<IndexId> expected) {
    EXPECT_TRUE(llvm::equal(actual, expected));
  }

  DialectRegistry registry;
  MLIRContext context;
  OwningOpRef<ModuleOp> module;
};

TEST_F(LinalgEinsumExtractionTest, AcceptsPointwiseProduct) {
  linalg::GenericOp generic = parseSingleGeneric(R"mlir(
module {
  func.func @pointwise(%a: tensor<8xf32>, %b: tensor<8xf32>)
      -> tensor<8xf32> {
    %init = arith.constant dense<0.0> : tensor<8xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%a, %b : tensor<8xf32>, tensor<8xf32>)
      outs(%init : tensor<8xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %rhs, %lhs : f32
      %sum = arith.addf %product, %acc : f32
      linalg.yield %sum : f32
    } -> tensor<8xf32>
    return %result : tensor<8xf32>
  }
}
)mlir");
  ASSERT_TRUE(generic);

  auto extracted = extractEinsumExpression(generic);
  ASSERT_TRUE(static_cast<bool>(extracted));
  expectIndices(extracted->getExpression().getOperands()[0].indices, {0});
  expectIndices(extracted->getExpression().getOperands()[1].indices, {0});
  expectIndices(extracted->getExpression().getResult().indices, {0});
  EXPECT_TRUE(extracted->getExpression().getReductionIndices().empty());
}

TEST_F(LinalgEinsumExtractionTest, AcceptsZeroFillInitializer) {
  linalg::GenericOp generic = parseSingleGeneric(R"mlir(
module {
  func.func @filled(%a: tensor<8xf32>, %b: tensor<8xf32>)
      -> tensor<8xf32> {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<8xf32>
    %init = linalg.fill ins(%zero : f32)
                        outs(%empty : tensor<8xf32>) -> tensor<8xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%a, %b : tensor<8xf32>, tensor<8xf32>)
      outs(%init : tensor<8xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<8xf32>
    return %result : tensor<8xf32>
  }
}
)mlir");
  ASSERT_TRUE(generic);

  auto extracted = extractEinsumExpression(generic);
  ASSERT_TRUE(static_cast<bool>(extracted));
  EXPECT_EQ(extracted->getInit(), generic.getOutputs()[0]);
}

TEST_F(LinalgEinsumExtractionTest, AcceptsScalarReduction) {
  linalg::GenericOp generic = parseSingleGeneric(R"mlir(
module {
  func.func @dot(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<f32> {
    %init = arith.constant dense<0.0> : tensor<f32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]
    } ins(%a, %b : tensor<4xf32>, tensor<4xf32>)
      outs(%init : tensor<f32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<f32>
    return %result : tensor<f32>
  }
}
)mlir");
  ASSERT_TRUE(generic);

  auto extracted = extractEinsumExpression(generic);
  ASSERT_TRUE(static_cast<bool>(extracted));
  expectIndices(extracted->getExpression().getResult().indices, {});
  expectIndices(extracted->getExpression().getReductionIndices(), {0});
  EXPECT_EQ(extracted->getExpression().getExtent(0), 4);
}

TEST_F(LinalgEinsumExtractionTest, PreservesPermutedResultOrder) {
  linalg::GenericOp generic = parseSingleGeneric(R"mlir(
module {
  func.func @transpose_product(%a: tensor<2x3xf32>, %b: tensor<2x3xf32>)
      -> tensor<3x2xf32> {
    %init = arith.constant dense<0.0> : tensor<3x2xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d1, d0)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%a, %b : tensor<2x3xf32>, tensor<2x3xf32>)
      outs(%init : tensor<3x2xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<3x2xf32>
    return %result : tensor<3x2xf32>
  }
}
)mlir");
  ASSERT_TRUE(generic);

  auto extracted = extractEinsumExpression(generic);
  ASSERT_TRUE(static_cast<bool>(extracted));
  expectIndices(extracted->getExpression().getResult().indices, {1, 0});
  EXPECT_EQ(extracted->getExpression().getExtent(0), 2);
  EXPECT_EQ(extracted->getExpression().getExtent(1), 3);
}

TEST_F(LinalgEinsumExtractionTest, RejectsDynamicShapes) {
  linalg::GenericOp generic = parseSingleGeneric(R"mlir(
module {
  func.func @dynamic(%a: tensor<?xf32>, %b: tensor<?xf32>,
                     %init: tensor<?xf32>) -> tensor<?xf32> {
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%a, %b : tensor<?xf32>, tensor<?xf32>)
      outs(%init : tensor<?xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<?xf32>
    return %result : tensor<?xf32>
  }
}
)mlir");
  ASSERT_TRUE(generic);

  expectInvalid(extractEinsumExpression(generic), "static shape");
}

TEST_F(LinalgEinsumExtractionTest, RejectsUnprovenOutputInitializer) {
  linalg::GenericOp generic = parseSingleGeneric(R"mlir(
module {
  func.func @accumulate(%a: tensor<8xf32>, %b: tensor<8xf32>,
                        %init: tensor<8xf32>) -> tensor<8xf32> {
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%a, %b : tensor<8xf32>, tensor<8xf32>)
      outs(%init : tensor<8xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<8xf32>
    return %result : tensor<8xf32>
  }
}
)mlir");
  ASSERT_TRUE(generic);

  expectInvalid(extractEinsumExpression(generic),
                "provable floating-point zero");
}

TEST_F(LinalgEinsumExtractionTest, RejectsBufferSemantics) {
  linalg::GenericOp generic = parseSingleGeneric(R"mlir(
module {
  func.func @buffer(%a: memref<8xf32>, %b: memref<8xf32>,
                    %out: memref<8xf32>) {
    linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%a, %b : memref<8xf32>, memref<8xf32>)
      outs(%out : memref<8xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(generic);

  expectInvalid(extractEinsumExpression(generic), "tensor output and result");
}

TEST_F(LinalgEinsumExtractionTest, RejectsNonSumProductBody) {
  linalg::GenericOp generic = parseSingleGeneric(R"mlir(
module {
  func.func @wrong_body(%a: tensor<8xf32>, %b: tensor<8xf32>,
                        %init: tensor<8xf32>) -> tensor<8xf32> {
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%a, %b : tensor<8xf32>, tensor<8xf32>)
      outs(%init : tensor<8xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %sum = arith.addf %lhs, %rhs : f32
      linalg.yield %sum : f32
    } -> tensor<8xf32>
    return %result : tensor<8xf32>
  }
}
)mlir");
  ASSERT_TRUE(generic);

  expectInvalid(extractEinsumExpression(generic), "one arith.mulf");
}

TEST_F(LinalgEinsumExtractionTest, RejectsProductWithoutAccumulator) {
  linalg::GenericOp generic = parseSingleGeneric(R"mlir(
module {
  func.func @wrong_add(%a: tensor<8xf32>, %b: tensor<8xf32>,
                       %init: tensor<8xf32>) -> tensor<8xf32> {
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%a, %b : tensor<8xf32>, tensor<8xf32>)
      outs(%init : tensor<8xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %lhs, %product : f32
      linalg.yield %sum : f32
    } -> tensor<8xf32>
    return %result : tensor<8xf32>
  }
}
)mlir");
  ASSERT_TRUE(generic);

  expectInvalid(extractEinsumExpression(generic), "output accumulator");
}

TEST_F(LinalgEinsumExtractionTest, RejectsInconsistentIteratorTypes) {
  linalg::GenericOp generic = parseSingleGeneric(R"mlir(
module {
  func.func @wrong_iterator(%a: tensor<2x3xf32>, %b: tensor<3xf32>,
                            %init: tensor<2xf32>) -> tensor<2xf32> {
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%a, %b : tensor<2x3xf32>, tensor<3xf32>)
      outs(%init : tensor<2xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<2xf32>
    return %result : tensor<2xf32>
  }
}
)mlir");
  ASSERT_TRUE(generic);

  expectInvalid(extractEinsumExpression(generic),
                "iterator 1 must be reduction");
}

TEST_F(LinalgEinsumExtractionTest, RejectsMoreThanTwoInputs) {
  linalg::GenericOp generic = parseSingleGeneric(R"mlir(
module {
  func.func @three_inputs(%a: tensor<8xf32>, %b: tensor<8xf32>,
                          %c: tensor<8xf32>, %init: tensor<8xf32>)
      -> tensor<8xf32> {
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%a, %b, %c : tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
      outs(%init : tensor<8xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %unused: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<8xf32>
    return %result : tensor<8xf32>
  }
}
)mlir");
  ASSERT_TRUE(generic);

  expectInvalid(extractEinsumExpression(generic), "exactly two input operands");
}

TEST_F(LinalgEinsumExtractionTest, RejectsZeroExtent) {
  linalg::GenericOp generic = parseSingleGeneric(R"mlir(
module {
  func.func @zero_extent(%a: tensor<0xf32>, %b: tensor<0xf32>,
                         %init: tensor<0xf32>) -> tensor<0xf32> {
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%a, %b : tensor<0xf32>, tensor<0xf32>)
      outs(%init : tensor<0xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<0xf32>
    return %result : tensor<0xf32>
  }
}
)mlir");
  ASSERT_TRUE(generic);

  expectInvalid(extractEinsumExpression(generic), "positive extent");
}

} // namespace
} // namespace mlir::lapis::einsum_extraction_tests

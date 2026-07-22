#include "lapis/Transform/AlgebraicKernelFusion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "gtest/gtest.h"
#include <initializer_list>
#include <string>

namespace mlir::lapis::region_tests {
namespace {

class LinalgEinsumRegionTest : public ::testing::Test {
protected:
  LinalgEinsumRegionTest() {
    registry
        .insert<arith::ArithDialect, func::FuncDialect, linalg::LinalgDialect,
                memref::MemRefDialect, scf::SCFDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  }

  func::FuncOp parseSingleFunction(llvm::StringRef source) {
    module = parseSourceString<ModuleOp>(source, &context);
    if (!module) {
      ADD_FAILURE() << "failed to parse test module";
      return {};
    }

    auto functions = module->getOps<func::FuncOp>();
    if (!llvm::hasSingleElement(functions)) {
      ADD_FAILURE() << "expected exactly one function";
      return {};
    }
    return *functions.begin();
  }

  void expectIndices(llvm::ArrayRef<IndexId> actual,
                     std::initializer_list<IndexId> expected) {
    EXPECT_TRUE(llvm::equal(actual, expected));
  }

  DialectRegistry registry;
  MLIRContext context;
  OwningOpRef<ModuleOp> module;
};

TEST_F(LinalgEinsumRegionTest, ComposesTwoProducerBranches) {
  func::FuncOp function = parseSingleFunction(R"mlir(
module {
  func.func @branches(%a: tensor<8xf32>, %b: tensor<8xf32>,
                      %c: tensor<8xf32>, %d: tensor<8xf32>)
      -> tensor<8xf32> {
    %zero0 = arith.constant dense<0.0> : tensor<8xf32>
    %ab = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%a, %b : tensor<8xf32>, tensor<8xf32>)
      outs(%zero0 : tensor<8xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<8xf32>

    %zero1 = arith.constant dense<0.0> : tensor<8xf32>
    %cd = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%c, %d : tensor<8xf32>, tensor<8xf32>)
      outs(%zero1 : tensor<8xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<8xf32>

    %zero2 = arith.constant dense<0.0> : tensor<8xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%ab, %cd : tensor<8xf32>, tensor<8xf32>)
      outs(%zero2 : tensor<8xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<8xf32>
    return %result : tensor<8xf32>
  }
}
)mlir");
  ASSERT_TRUE(function);

  auto regions = discoverLinalgEinsumRegions(function);
  ASSERT_TRUE(static_cast<bool>(regions));
  ASSERT_EQ(regions->size(), 1u);
  EXPECT_EQ(regions->front().getOperations().size(), 3u);
  EXPECT_EQ(regions->front().getOperands().size(), 4u);
  EXPECT_EQ(regions->front().getExpression().getOperands().size(), 4u);
  for (const TensorAccess &operand :
       regions->front().getExpression().getOperands())
    expectIndices(operand.indices, {0});

  auto planner =
      ExactContractionPlanner::create(regions->front().getExpression());
  ASSERT_TRUE(static_cast<bool>(planner));
  auto plan = planner->plan();
  ASSERT_TRUE(static_cast<bool>(plan));
  auto evaluation = evaluateFusionCandidate(regions->front(), *plan);
  ASSERT_TRUE(static_cast<bool>(evaluation));
  EXPECT_EQ(evaluation->originalWork, 24u);
  EXPECT_EQ(evaluation->optimizedWork, 24u);
  EXPECT_EQ(evaluation->originalKernelCount, 3u);
  EXPECT_EQ(evaluation->fusedKernelCount, 1u);
  EXPECT_TRUE(evaluation->shouldRewrite());
}

TEST_F(LinalgEinsumRegionTest, PCGDenominatorHasEqualContractionWork) {
  func::FuncOp function = parseSingleFunction(R"mlir(
module {
  func.func @pcg_denominator(%a: tensor<8x8xf64>, %p: tensor<8xf64>)
      -> (tensor<8xf64>, tensor<f64>) {
    %ap_zero = arith.constant dense<0.0> : tensor<8xf64>
    %ap = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%a, %p : tensor<8x8xf64>, tensor<8xf64>)
      outs(%ap_zero : tensor<8xf64>) {
    ^bb0(%lhs: f64, %rhs: f64, %acc: f64):
      %product = arith.mulf %lhs, %rhs : f64
      %sum = arith.addf %acc, %product : f64
      linalg.yield %sum : f64
    } -> tensor<8xf64>

    %pap_zero = arith.constant dense<0.0> : tensor<f64>
    %pap = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]
    } ins(%p, %ap : tensor<8xf64>, tensor<8xf64>)
      outs(%pap_zero : tensor<f64>) {
    ^bb0(%lhs: f64, %rhs: f64, %acc: f64):
      %product = arith.mulf %lhs, %rhs : f64
      %sum = arith.addf %acc, %product : f64
      linalg.yield %sum : f64
    } -> tensor<f64>
    return %ap, %pap : tensor<8xf64>, tensor<f64>
  }
}
)mlir");
  ASSERT_TRUE(function);

  auto regions = discoverLinalgEinsumRegions(function);
  ASSERT_TRUE(static_cast<bool>(regions));
  ASSERT_EQ(regions->size(), 1u);
  ASSERT_EQ(regions->front().getOutputs().size(), 2u);
  ASSERT_EQ(regions->front().getOperands().size(), 3u);
  EXPECT_EQ(regions->front().getOperands()[0], function.getArgument(1));
  EXPECT_EQ(regions->front().getOperands()[1], function.getArgument(0));
  EXPECT_EQ(regions->front().getOperands()[2], function.getArgument(1));

  auto planner =
      ExactContractionPlanner::create(regions->front().getExpression());
  ASSERT_TRUE(static_cast<bool>(planner));
  llvm::SmallVector<ContractionResultConstraint, 2> requiredResults;
  for (const LinalgEinsumRegionOutput &output : regions->front().getOutputs()) {
    requiredResults.push_back(
        ContractionResultConstraint{output.operandSubset, output.access});
  }
  auto plan = planner->plan(requiredResults);
  ASSERT_TRUE(static_cast<bool>(plan));
  EXPECT_EQ(regions->front().getOutputs()[0].operandSubset,
            OperandSubset{0b110});
  auto ap = findPlanValueForOperandSubset(*plan, 0b110);
  EXPECT_TRUE(static_cast<bool>(ap));
  auto evaluation = evaluateFusionCandidate(regions->front(), *plan);
  ASSERT_TRUE(static_cast<bool>(evaluation));
  EXPECT_EQ(evaluation->originalWork, 72u);
  EXPECT_EQ(evaluation->optimizedWork, 72u);
  EXPECT_EQ(evaluation->originalKernelCount, 2u);
  EXPECT_EQ(evaluation->fusedKernelCount, 1u);
  EXPECT_TRUE(evaluation->shouldRewrite());
}

TEST_F(LinalgEinsumRegionTest, UnsupportedConsumerTerminatesRegion) {
  func::FuncOp function = parseSingleFunction(R"mlir(
module {
  func.func @unsupported_consumer(%a: tensor<8xf32>, %b: tensor<8xf32>,
                                  %c: tensor<8xf32>) -> tensor<8xf32> {
    %zero0 = arith.constant dense<0.0> : tensor<8xf32>
    %ab = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%a, %b : tensor<8xf32>, tensor<8xf32>)
      outs(%zero0 : tensor<8xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<8xf32>

    %zero1 = arith.constant dense<0.0> : tensor<8xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%ab, %c : tensor<8xf32>, tensor<8xf32>)
      outs(%zero1 : tensor<8xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %difference = arith.subf %lhs, %rhs : f32
      %sum = arith.addf %acc, %difference : f32
      linalg.yield %sum : f32
    } -> tensor<8xf32>
    return %result : tensor<8xf32>
  }
}
)mlir");
  ASSERT_TRUE(function);

  auto regions = discoverLinalgEinsumRegions(function);
  ASSERT_TRUE(static_cast<bool>(regions));
  EXPECT_TRUE(regions->empty());
}

} // namespace
} // namespace mlir::lapis::region_tests

namespace mlir::lapis::materialization_tests {
namespace {

constexpr llvm::StringLiteral abx = R"mlir(
module {
  func.func @abx(%a: tensor<2x3xf32>, %b: tensor<3x5xf32>,
                 %x: tensor<5xf32>) -> tensor<2xf32> {
    %matrix_zero = arith.constant dense<0.0> : tensor<2x5xf32>
    %ab = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d2, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%a, %b : tensor<2x3xf32>, tensor<3x5xf32>)
      outs(%matrix_zero : tensor<2x5xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<2x5xf32>

    %vector_zero = arith.constant dense<0.0> : tensor<2xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%ab, %x : tensor<2x5xf32>, tensor<5xf32>)
      outs(%vector_zero : tensor<2xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<2xf32>
    return %result : tensor<2xf32>
  }
}
)mlir";

class LinalgContractionMaterializationTest : public ::testing::Test {
protected:
  LinalgContractionMaterializationTest() {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    linalg::LinalgDialect, tensor::TensorDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  }

  func::FuncOp parseABx() {
    module = parseSourceString<ModuleOp>(abx, &context);
    if (!module) {
      ADD_FAILURE() << "failed to parse ABx module";
      return {};
    }
    return *module->getOps<func::FuncOp>().begin();
  }

  DialectRegistry registry;
  MLIRContext context;
  OwningOpRef<ModuleOp> module;
};

TEST_F(LinalgContractionMaterializationTest,
       SelectsABxAndRejectsInvalidMaterialization) {
  func::FuncOp function = parseABx();
  ASSERT_TRUE(function);
  auto regions = discoverLinalgEinsumRegions(function);
  ASSERT_TRUE(static_cast<bool>(regions));
  ASSERT_EQ(regions->size(), 1u);
  Value originalResult = regions->front().getResult();

  auto planner =
      ExactContractionPlanner::create(regions->front().getExpression());
  ASSERT_TRUE(static_cast<bool>(planner));
  auto optimizedPlan = planner->plan();
  ASSERT_TRUE(static_cast<bool>(optimizedPlan));
  auto evaluation = evaluateFusionCandidate(regions->front(), *optimizedPlan);
  ASSERT_TRUE(static_cast<bool>(evaluation));
  EXPECT_EQ(evaluation->originalWork, 40u);
  EXPECT_EQ(evaluation->optimizedWork, 21u);
  EXPECT_TRUE(evaluation->shouldRewrite());

  llvm::SmallVector<ContractionStep, 2> steps{
      ContractionStep{/*lhs=*/1, /*rhs=*/2, /*resultIndices=*/{1}},
      ContractionStep{/*lhs=*/0, /*rhs=*/3, /*resultIndices=*/{0}}};
  ContractionPlan invalidPlan(/*operandCount=*/3, steps, /*result=*/4);

  IRRewriter rewriter(&context);
  auto replacement =
      materializeContractionPlan(rewriter, regions->front(), invalidPlan);
  ASSERT_FALSE(static_cast<bool>(replacement));
  std::string error = llvm::toString(replacement.takeError());
  EXPECT_NE(error.find("result indices do not match index liveness"),
            std::string::npos)
      << error;

  EXPECT_EQ(llvm::range_size(function.getOps<linalg::GenericOp>()), 2u);
  EXPECT_TRUE(function.getOps<tensor::EmptyOp>().empty());
  auto returnOp = cast<func::ReturnOp>(function.getBody().front().back());
  EXPECT_EQ(returnOp.getOperand(0), originalResult);
  EXPECT_TRUE(succeeded(verify(*module)));
}

} // namespace
} // namespace mlir::lapis::materialization_tests

namespace mlir::lapis::kernel_pipeline_tests {
namespace {

TEST(AlgebraicKernelPipelineTest, OutlinesAndFusesOnlyProvenCommonDomain) {
  DialectRegistry registry;
  registry.insert<arith::ArithDialect, func::FuncDialect, linalg::LinalgDialect,
                  memref::MemRefDialect, scf::SCFDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module = parseSourceString<ModuleOp>(R"mlir(
module {
  func.func @outline(%a: memref<4xf32>, %b: memref<4xf32>,
                     %tmp: memref<4xf32>, %c: memref<4xf32>,
                     %out: memref<4xf32>) {
    linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%a, %b : memref<4xf32>, memref<4xf32>)
      outs(%tmp : memref<4xf32>)
      attrs = {lapis.algebraic_fusion_group = 7 : i64} {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    }
    linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%tmp, %c : memref<4xf32>, memref<4xf32>)
      outs(%out : memref<4xf32>)
      attrs = {lapis.algebraic_fusion_group = 7 : i64} {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    }
    return
  }

  func.func @internalize(%a: memref<4xf32>, %b: memref<4xf32>,
                         %out: memref<4xf32>) {
    %zero = arith.constant 0.0 : f32
    %tmp = memref.alloc() : memref<4xf32>
    linalg.fill ins(%zero : f32) outs(%tmp : memref<4xf32>)
    linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%a, %b : memref<4xf32>, memref<4xf32>)
      outs(%tmp : memref<4xf32>)
      attrs = {lapis.algebraic_fusion_group = 11 : i64} {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    }
    linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%tmp, %b : memref<4xf32>, memref<4xf32>)
      outs(%out : memref<4xf32>)
      attrs = {lapis.algebraic_fusion_group = 11 : i64} {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    }
    return
  }

  func.func private @fusable(%input: memref<2x3xf32>,
                             %tmp: memref<2x3xf32>,
                             %out: memref<2x2xf32>)
      attributes {lapis.algebraic_kernel,
                  lapis.algebraic_fusion_group = 8 : i64} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    scf.parallel (%batch, %k) = (%c0, %c0) to (%c2, %c3)
        step (%c1, %c1) {
      %value = memref.load %input[%batch, %k] : memref<2x3xf32>
      memref.store %value, %tmp[%batch, %k] : memref<2x3xf32>
    }
    scf.parallel (%batch, %i) = (%c0, %c0) to (%c2, %c2)
        step (%c1, %c1) {
      %value = memref.load %tmp[%batch, %i] : memref<2x3xf32>
      memref.store %value, %out[%batch, %i] : memref<2x2xf32>
    }
    return
  }

  func.func private @unsafe(%input: memref<2x3xf32>,
                            %tmp: memref<2x3xf32>,
                            %out: memref<2x2xf32>)
      attributes {lapis.algebraic_kernel,
                  lapis.algebraic_fusion_group = 9 : i64} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    scf.parallel (%batch, %k) = (%c0, %c0) to (%c2, %c3)
        step (%c1, %c1) {
      %value = memref.load %input[%batch, %k] : memref<2x3xf32>
      memref.store %value, %tmp[%batch, %k] : memref<2x3xf32>
    }
    scf.parallel (%batch, %i) = (%c0, %c0) to (%c2, %c2)
        step (%c1, %c1) {
      %value = memref.load %tmp[%c0, %i] : memref<2x3xf32>
      memref.store %value, %out[%batch, %i] : memref<2x2xf32>
    }
    return
  }

  func.func private @reduction_consumer(%input: memref<2xf32>,
                                        %tmp: memref<2xf32>,
                                        %result: memref<f32>)
      attributes {lapis.algebraic_kernel,
                  lapis.algebraic_fusion_group = 10 : i64} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.parallel (%i) = (%c0) to (%c2) step (%c1) {
      %value = memref.load %input[%i] : memref<2xf32>
      memref.store %value, %tmp[%i] : memref<2xf32>
    }
    %initial = memref.load %result[] : memref<f32>
    %sum = scf.parallel (%i) = (%c0) to (%c2) step (%c1)
        init (%initial) -> f32 {
      %value = memref.load %tmp[%i] : memref<2xf32>
      scf.reduce(%value : f32) {
      ^bb0(%lhs: f32, %rhs: f32):
        %next = arith.addf %lhs, %rhs : f32
        scf.reduce.return %next : f32
      }
    }
    memref.store %sum, %result[] : memref<f32>
    return
  }

  func.func private @eliminate(%input: memref<4xf32>,
                               %output: memref<4xf32>)
      attributes {lapis.algebraic_kernel,
                  lapis.algebraic_fusion_group = 12 : i64} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %tmp = memref.alloc() : memref<4xf32>
    scf.parallel (%i) = (%c0) to (%c4) step (%c1) {
      %value = memref.load %input[%i] : memref<4xf32>
      %doubled = arith.addf %value, %value : f32
      memref.store %doubled, %tmp[%i] : memref<4xf32>
      %forwarded = memref.load %tmp[%i] : memref<4xf32>
      %result = arith.addf %forwarded, %value : f32
      memref.store %result, %output[%i] : memref<4xf32>
    }
    memref.dealloc %tmp : memref<4xf32>
    return
  }

  func.func @main(%input: memref<2x3xf32>) {
    %tmp0 = memref.alloc() : memref<2x3xf32>
    %out0 = memref.alloc() : memref<2x2xf32>
    call @fusable(%input, %tmp0, %out0)
        : (memref<2x3xf32>, memref<2x3xf32>, memref<2x2xf32>) -> ()
    %tmp1 = memref.alloc() : memref<2x3xf32>
    %out1 = memref.alloc() : memref<2x2xf32>
    call @unsafe(%input, %tmp1, %out1)
        : (memref<2x3xf32>, memref<2x3xf32>, memref<2x2xf32>) -> ()
    %reduction_input = memref.alloc() : memref<2xf32>
    %reduction_tmp = memref.alloc() : memref<2xf32>
    %reduction_result = memref.alloc() : memref<f32>
    call @reduction_consumer(%reduction_input, %reduction_tmp,
                             %reduction_result)
        : (memref<2xf32>, memref<2xf32>, memref<f32>) -> ()
    %elimination_input = memref.alloc() : memref<4xf32>
    %elimination_output = memref.alloc() : memref<4xf32>
    call @eliminate(%elimination_input, %elimination_output)
        : (memref<4xf32>, memref<4xf32>) -> ()
    return
  }
}
)mlir",
                                            &context);
  ASSERT_TRUE(module);

  PassManager passManager(&context);
  passManager.addPass(createOutlineAlgebraicKernelsPass());
  passManager.addPass(createFuseAlgebraicKernelLoopsPass());
  ASSERT_TRUE(succeeded(passManager.run(*module)));
  ASSERT_TRUE(succeeded(verify(*module)));

  auto outlined =
      module->lookupSymbol<func::FuncOp>("outline__lapis_algebraic_kernel_7");
  ASSERT_TRUE(outlined);
  EXPECT_TRUE(outlined.isPrivate());
  EXPECT_TRUE(outlined->hasAttr(kAlgebraicKernelAttr));
  EXPECT_EQ(llvm::range_size(outlined.getOps<linalg::GenericOp>()), 2u);

  auto outlineParent = module->lookupSymbol<func::FuncOp>("outline");
  ASSERT_TRUE(outlineParent);
  EXPECT_TRUE(outlineParent.getOps<linalg::GenericOp>().empty());
  EXPECT_EQ(llvm::range_size(outlineParent.getOps<func::CallOp>()), 1u);

  auto internalized = module->lookupSymbol<func::FuncOp>(
      "internalize__lapis_algebraic_kernel_11");
  ASSERT_TRUE(internalized);
  EXPECT_EQ(internalized.getNumArguments(), 3u);
  EXPECT_EQ(llvm::range_size(internalized.getOps<memref::AllocOp>()), 1u);
  EXPECT_EQ(llvm::range_size(internalized.getOps<linalg::FillOp>()), 1u);
  EXPECT_EQ(llvm::range_size(internalized.getOps<linalg::GenericOp>()), 2u);

  auto internalizeParent = module->lookupSymbol<func::FuncOp>("internalize");
  ASSERT_TRUE(internalizeParent);
  EXPECT_TRUE(internalizeParent.getOps<memref::AllocOp>().empty());
  EXPECT_TRUE(internalizeParent.getOps<linalg::FillOp>().empty());
  EXPECT_EQ(llvm::range_size(internalizeParent.getOps<func::CallOp>()), 1u);

  auto fusable = module->lookupSymbol<func::FuncOp>("fusable");
  ASSERT_TRUE(fusable);
  auto fusedLoops = llvm::to_vector(fusable.getOps<scf::ParallelOp>());
  ASSERT_EQ(fusedLoops.size(), 1u);
  EXPECT_EQ(fusedLoops.front().getNumLoops(), 1u);
  EXPECT_EQ(
      llvm::range_size(fusedLoops.front().getBody()->getOps<scf::ParallelOp>()),
      2u);

  auto unsafe = module->lookupSymbol<func::FuncOp>("unsafe");
  ASSERT_TRUE(unsafe);
  EXPECT_EQ(llvm::range_size(unsafe.getOps<scf::ParallelOp>()), 2u);

  auto reductionConsumer =
      module->lookupSymbol<func::FuncOp>("reduction_consumer");
  ASSERT_TRUE(reductionConsumer);
  auto reductionLoops =
      llvm::to_vector(reductionConsumer.getOps<scf::ParallelOp>());
  ASSERT_EQ(reductionLoops.size(), 1u);
  EXPECT_EQ(reductionLoops.front().getNumLoops(), 1u);
  EXPECT_EQ(reductionLoops.front().getNumResults(), 1u);

  auto eliminated = module->lookupSymbol<func::FuncOp>("eliminate");
  ASSERT_TRUE(eliminated);
  EXPECT_TRUE(eliminated.getOps<memref::AllocOp>().empty());
  EXPECT_TRUE(eliminated.getOps<memref::DeallocOp>().empty());
  auto eliminatedLoops = llvm::to_vector(eliminated.getOps<scf::ParallelOp>());
  ASSERT_EQ(eliminatedLoops.size(), 1u);
  EXPECT_EQ(llvm::range_size(eliminatedLoops.front().getOps<memref::LoadOp>()),
            1u);
  EXPECT_EQ(llvm::range_size(eliminatedLoops.front().getOps<memref::StoreOp>()),
            1u);
}

} // namespace
} // namespace mlir::lapis::kernel_pipeline_tests

//===- MemrefResultsToParams.cpp -
// Patterns to move memref return values to parameters
//--------------------===//

#include "lapis/Dialect/Kokkos/IR/KokkosDialect.h"
#include "lapis/Dialect/Kokkos/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"

namespace mlir {
#define GEN_PASS_DEF_MEMREFRESULTSTOPARAMS
#include "lapis/Dialect/Kokkos/Transforms/Passes.h.inc"
}

using namespace mlir;

namespace {

struct MemrefResultsToParamsRewriter : public OpRewritePattern<func::FuncOp> {
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;

  MemrefResultsToParamsRewriter(MLIRContext *context) : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(func::FuncOp op,
                                PatternRewriter &rewriter) const override {
    return failure();
  }
};

}

void mlir::populateMemrefResultsToParamsPatterns(RewritePatternSet &patterns) {
  patterns.add<MemrefResultsToParamsRewriter>(patterns.getContext());
}

struct MemrefResultsToParamsPass
    : public impl::MemrefResultsToParamsBase<MemrefResultsToParamsPass> {

  MemrefResultsToParamsPass() = default;
  MemrefResultsToParamsPass(const MemrefResultsToParamsPass& pass) = default;

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    populateMemrefResultsToParamsPatterns(patterns);
    (void) applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

std::unique_ptr<Pass> mlir::createMemrefResultsToParamsPass()
{
  return std::make_unique<MemrefResultsToParamsPass>();
}


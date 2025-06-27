//===- MemrefsToKokkosScratch.cpp -
// Patterns to place memref allocations in Kokkos scratch
//--------------------===//

#include "lapis/Dialect/Kokkos/IR/KokkosDialect.h"
#include "lapis/Dialect/Kokkos/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_MEMREFTOKOKKOSSCRATCH
#include "lapis/Dialect/Kokkos/Transforms/Passes.h.inc"
}

using namespace mlir;

namespace {

struct MemrefToKokkosScratchRewriter : public OpRewritePattern<func::FuncOp> {
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;

  MemrefToKokkosScratchRewriter(MLIRContext *context) : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(func::FuncOp op, PatternRewriter &rewriter) const override {
    return failure();
  }
};

}

struct MemrefToKokkosScratchPass 
    : public impl::MemrefToKokkosScratchBase<MemrefToKokkosScratchPass> {

  MemrefToKokkosScratchPass() = default;
  MemrefToKokkosScratchPass(const MemrefToKokkosScratchPass& pass) = default;

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<MemrefToKokkosScratchRewriter>(patterns.getContext());
    (void) applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

std::unique_ptr<Pass> mlir::createMemrefToKokkosScratchPass()
{
  return std::make_unique<MemrefToKokkosScratchPass>();
}


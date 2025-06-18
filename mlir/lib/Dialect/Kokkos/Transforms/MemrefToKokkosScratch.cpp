//===- MemrefsToKokkosScratch.cpp -
// Patterns to place memref allocations in Kokkos scratch
//--------------------===//

#include "lapis/Dialect/Kokkos/IR/KokkosDialect.h"
#include "lapis/Dialect/Kokkos/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"

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

void mlir::populateMemrefToKokkosScratchPatterns(RewritePatternSet &patterns) {
  patterns.add<MemrefToKokkosScratchRewriter>(patterns.getContext());
}


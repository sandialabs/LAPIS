//===- MemrefCopyToParallel.cpp -
// Pattern to rewrite memref.copy as an
// scf.parallel with element load/store in the body.
//--------------------===//

#include "lapis/Dialect/Kokkos/IR/KokkosDialect.h"
#include "lapis/Dialect/Kokkos/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <vector>

namespace mlir {
#define GEN_PASS_DEF_MEMREFCOPYTOPARALLEL
#include "lapis/Dialect/Kokkos/Transforms/Passes.h.inc"
}

using namespace mlir;

struct MemrefCopyRewriter : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  MemrefCopyRewriter(MLIRContext *context) : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(memref::CopyOp op, PatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    Value src = op.getSource();
    Value dst = op.getTarget();
    // Get the shape of src and dst
    MemRefType mrtSrc = cast<MemRefType>(src.getType());
    // If only one memref has static shape, prefer that one for getting shape
    // (memref.dim can fold to constant)
    Value shapeVal = mrtSrc.hasStaticShape() ? src : dst;
    int rank = mrtSrc.getRank();
    SmallVector<Value> lb, ub, step;
    for(int i = 0; i < rank; i++) {
      // Define iteration space from 0...dim, with step 1
      step.push_back(rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 1));
      lb.push_back(rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 0));
      ub.push_back(rewriter.create<memref::DimOp>(op->getLoc(), shapeVal, i).getResult());
    }
    rewriter.create<scf::ParallelOp>(
        op->getLoc(), lb, ub, step,
        [&](OpBuilder& nestedBuilder, Location, ValueRange ivs) {
          // Load fro
          Value elem = nestedBuilder.create<memref::LoadOp>(op->getLoc(), src, ivs);
          nestedBuilder.create<memref::StoreOp>(op->getLoc(), elem, dst, ivs);
        });
    rewriter.eraseOp(op);
    return success();
  }
};

struct MemrefCopyToParallelPass 
    : public impl::MemrefCopyToParallelBase<MemrefCopyToParallelPass> {

  MemrefCopyToParallelPass() = default;
  MemrefCopyToParallelPass(const MemrefCopyToParallelPass& pass) = default;

  void runOnOperation() override {
    MLIRContext* ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<MemrefCopyRewriter>(patterns.getContext());
    (void) applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

std::unique_ptr<Pass> mlir::createMemrefCopyToParallelPass()
{
  return std::make_unique<MemrefCopyToParallelPass>();
}


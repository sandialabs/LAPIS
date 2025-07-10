//===- TransposeConstantFold.cpp -===//
// Pattern to fold linalg.transpose on constant tensors       //
// This is just a driver for the pattern that lives in Linalg //

#include "lapis/Dialect/Kokkos/IR/KokkosDialect.h"
#include "lapis/Dialect/Kokkos/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::kokkos;

namespace mlir {
#define GEN_PASS_DEF_TRANSPOSECONSTANTFOLDPASS
#include "lapis/Dialect/Kokkos/Transforms/Passes.h.inc"
}

static bool alwaysFold(OpOperand*) {return true;}

struct TransposeConstantFoldPass
    : public impl::TransposeConstantFoldPassBase <TransposeConstantFoldPass> {

  using impl::TransposeConstantFoldPassBase<TransposeConstantFoldPass>::TransposeConstantFoldPassBase;

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = op->getContext();
    RewritePatternSet patterns(context);

    linalg::populateConstantFoldLinalgOperations(patterns, alwaysFold);

    (void) applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

std::unique_ptr<Pass> mlir::createTransposeConstantFoldPass() {
  return std::make_unique<TransposeConstantFoldPass>();
}


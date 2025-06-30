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

struct MemrefToKokkosScratchPass 
    : public impl::MemrefToKokkosScratchBase<MemrefToKokkosScratchPass> {

  MemrefToKokkosScratchPass() = default;
  MemrefToKokkosScratchPass(const MemrefToKokkosScratchPass& pass) = default;

  void runOnOperation() override {
    MLIRContext* ctx = &getContext();
    IRRewriter rewriter(ctx);
    func::FuncOp func = getOperation();
  }
};

std::unique_ptr<Pass> mlir::createMemrefToKokkosScratchPass()
{
  return std::make_unique<MemrefToKokkosScratchPass>();
}


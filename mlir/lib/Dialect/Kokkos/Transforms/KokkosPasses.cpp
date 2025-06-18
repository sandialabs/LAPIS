//===- KokkosPasses.cpp - Passes for lowering to Kokkos dialect -------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h" //for SparseParallelizationStrategy
#include "lapis/Dialect/Kokkos/IR/KokkosDialect.h"
#include "lapis/Dialect/Kokkos/Transforms/Passes.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_MEMREFRESULTSTOPARAMS
#define GEN_PASS_DEF_MEMREFTOKOKKOSSCRATCH
#define GEN_PASS_DEF_PARALLELUNITSTEP
#define GEN_PASS_DEF_KOKKOSLOOPMAPPING
#define GEN_PASS_DEF_KOKKOSMEMORYSPACEASSIGNMENT

#include "lapis/Dialect/Kokkos/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::kokkos;

namespace {

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

struct MemrefToKokkosScratchPass 
    : public impl::MemrefToKokkosScratchBase<MemrefToKokkosScratchPass> {

  MemrefToKokkosScratchPass() = default;
  MemrefToKokkosScratchPass(const MemrefToKokkosScratchPass& pass) = default;

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    populateMemrefToKokkosScratchPatterns(patterns);
    (void) applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

struct ParallelUnitStepPass
    : public impl::ParallelUnitStepBase<ParallelUnitStepPass> {

  ParallelUnitStepPass() = default;
  ParallelUnitStepPass(const ParallelUnitStepPass& pass) = default;

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    populateParallelUnitStepPatterns(patterns);
    (void) applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

struct KokkosLoopMappingPass
    : public impl::KokkosLoopMappingBase<KokkosLoopMappingPass> {

  KokkosLoopMappingPass() = default;
  KokkosLoopMappingPass(const KokkosLoopMappingPass& pass) = default;
  KokkosLoopMappingPass(const KokkosLoopMappingOptions& options) : impl::KokkosLoopMappingBase<KokkosLoopMappingPass>(options) {}

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    populateKokkosLoopMappingPatterns(patterns, this->teamLevel);
    (void) applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

struct KokkosMemorySpaceAssignmentPass
    : public impl::KokkosMemorySpaceAssignmentBase<KokkosMemorySpaceAssignmentPass> {

  KokkosMemorySpaceAssignmentPass() = default;
  KokkosMemorySpaceAssignmentPass(const KokkosMemorySpaceAssignmentPass& pass) = default;

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    populateKokkosMemorySpaceAssignmentPatterns(patterns);
    (void) applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};
}

std::unique_ptr<Pass> mlir::createMemrefResultsToParamsPass()
{
  return std::make_unique<MemrefResultsToParamsPass>();
}

std::unique_ptr<Pass> mlir::createMemrefToKokkosScratchPass()
{
  return std::make_unique<MemrefToKokkosScratchPass>();
}

std::unique_ptr<Pass> mlir::createParallelUnitStepPass()
{
  return std::make_unique<ParallelUnitStepPass>();
}

std::unique_ptr<Pass> mlir::createKokkosLoopMappingPass(bool teamLevel)
{
  KokkosLoopMappingOptions klmo;
  klmo.teamLevel = teamLevel;
  return std::make_unique<KokkosLoopMappingPass>(klmo);
}

std::unique_ptr<Pass> mlir::createKokkosMemorySpaceAssignmentPass()
{
  return std::make_unique<KokkosMemorySpaceAssignmentPass>();
}


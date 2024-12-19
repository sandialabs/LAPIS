//===- KokkosPasses.cpp - Passes for lowering to Kokkos dialect -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "lapis/Dialect/Kokkos/IR/KokkosDialect.h"
#include "lapis/Dialect/Kokkos/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h" //for SparseParallelizationStrategy

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_PARALLELUNITSTEP
#define GEN_PASS_DEF_KOKKOSLOOPMAPPING
#define GEN_PASS_DEF_KOKKOSMEMORYSPACEASSIGNMENT

#define GEN_PASS_DEF_SPARSEKOKKOSCODEGEN

#include "lapis/Dialect/Kokkos/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::kokkos;

namespace {

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

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    populateKokkosLoopMappingPatterns(patterns);
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

std::unique_ptr<Pass> mlir::createParallelUnitStepPass()
{
  return std::make_unique<ParallelUnitStepPass>();
}

std::unique_ptr<Pass> mlir::createKokkosLoopMappingPass()
{
  return std::make_unique<KokkosLoopMappingPass>();
}

std::unique_ptr<Pass> mlir::createKokkosMemorySpaceAssignmentPass()
{
  return std::make_unique<KokkosMemorySpaceAssignmentPass>();
}

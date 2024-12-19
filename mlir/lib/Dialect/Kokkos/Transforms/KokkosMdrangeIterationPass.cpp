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

#include <iostream> // is there an LLVM way to do this?

namespace mlir {
#define GEN_PASS_DEF_KOKKOSMDRANGEITERATION

#include "lapis/Dialect/Kokkos/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::kokkos;

namespace {

struct KokkosMdrangeIterationPass
    : public impl::KokkosMdrangeIterationBase<KokkosMdrangeIterationPass> {

  KokkosMdrangeIterationPass() = default;
  KokkosMdrangeIterationPass(const KokkosMdrangeIterationPass& pass) = default;

  void runOnOperation() override {
    // do nothing
    std::cerr << __FILE__ << ":" << __LINE__ << "\n";
  }
};

} // anonymous namespace

std::unique_ptr<Pass> mlir::createKokkosMdrangeIterationPass() {
  return std::make_unique<KokkosMdrangeIterationPass>();
}

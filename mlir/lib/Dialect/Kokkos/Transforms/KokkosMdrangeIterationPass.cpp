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


  static void dump_ops(ModuleOp &mod) {
    mod.walk([&](Operation *op) {
      if (auto parallelOp = dyn_cast<scf::ParallelOp>(op)) {
        llvm::outs() << "Found scf.parallel operation:\n";
        llvm::outs() << "Induction variables and strides:\n";
        for (auto iv : llvm::zip(parallelOp.getInductionVars(), parallelOp.getStep())) {
          std::get<0>(iv).print(llvm::outs());
          llvm::outs() << " with stride ";
          std::get<1>(iv).print(llvm::outs());
          llvm::outs() << "\n";
        }
        llvm::outs() << "\n\n";
      }

      if (auto memrefOp = dyn_cast<memref::LoadOp>(op)) {
        llvm::outs() << "Found memref.load operation:\n";
        llvm::outs() << "MemRef: ";
        memrefOp.getMemRef().print(llvm::outs());
        llvm::outs() << "\nIndex variables:\n";
        for (Value index : memrefOp.getIndices()) {
          index.print(llvm::outs());
          llvm::outs() << "\n";
        }
        if (auto memrefType = memrefOp.getMemRef().getType().dyn_cast<MemRefType>()) {
          llvm::outs() << "MemRef extents:\n";
          for (int64_t dim : memrefType.getShape()) {
            llvm::outs() << dim << "\n";
          }
        }
        llvm::outs() << "\n\n";
      }

      if (auto memrefOp = dyn_cast<memref::StoreOp>(op)) {
        llvm::outs() << "Found memref.store operation:\n";
        llvm::outs() << "MemRef: ";
        memrefOp.getMemRef().print(llvm::outs());
        llvm::outs() << "\nIndex variables:\n";
        for (Value index : memrefOp.getIndices()) {
          index.print(llvm::outs());
          llvm::outs() << "\n";
        }
        if (auto memrefType = memrefOp.getMemRef().getType().dyn_cast<MemRefType>()) {
          llvm::outs() << "MemRef extents:\n";
          for (int64_t dim : memrefType.getShape()) {
            llvm::outs() << dim << "\n";
          }
        }
        llvm::outs() << "\n\n";
      }
    });
  }


  void runOnOperation() override {
    ModuleOp module = getOperation();
    dump_ops(module);
  }
};

} // anonymous namespace

std::unique_ptr<Pass> mlir::createKokkosMdrangeIterationPass() {
  return std::make_unique<KokkosMdrangeIterationPass>();
}

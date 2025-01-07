//===- Passes.h - Kokkos pipeline entry points -----------*- C++ -*-===//
//
// **** This file has been modified from its original in llvm-project ****
// Original file was mlir/include/mlir/Dialect/SparseTensor/Pipelines/Passes.h
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes of all sparse tensor pipelines.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_KOKKOS_PIPELINES_PASSES_H_
#define MLIR_KOKKOS_PIPELINES_PASSES_H_

#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "lapis/Dialect/Kokkos/Transforms/Passes.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Pass/PassOptions.h"
#if defined(ENABLE_PART_TENSOR)
#include "lapis/Dialect/PartTensor/Transforms/Passes.h"
#endif

using namespace mlir::detail;
using namespace llvm::cl;

namespace mlir {
namespace kokkos {

struct LapisCompilerOptions
    : public PassPipelineOptions<LapisCompilerOptions> {
  PassOptions::Option<mlir::SparseParallelizationStrategy> parallelization{
      *this, "parallelization-strategy",
      ::llvm::cl::desc("Set the parallelization strategy (default: parallelize every possible loop)"),
      ::llvm::cl::init(mlir::SparseParallelizationStrategy::kAnyStorageAnyLoop),
      llvm::cl::values(
          clEnumValN(mlir::SparseParallelizationStrategy::kNone, "none",
                     "Turn off sparse parallelization."),
          clEnumValN(mlir::SparseParallelizationStrategy::kDenseOuterLoop,
                     "dense-outer-loop",
                     "Enable dense outer loop sparse parallelization."),
          clEnumValN(mlir::SparseParallelizationStrategy::kAnyStorageOuterLoop,
                     "any-storage-outer-loop",
                     "Enable sparse parallelization regardless of storage for "
                     "the outer loop."),
          clEnumValN(mlir::SparseParallelizationStrategy::kDenseAnyLoop,
                     "dense-any-loop",
                     "Enable dense parallelization for any loop."),
          clEnumValN(
              mlir::SparseParallelizationStrategy::kAnyStorageAnyLoop,
              "any-storage-any-loop",
              "Enable sparse parallelization for any storage and loop."))};

#if defined(ENABLE_PART_TENSOR)
  PassOptions::Option<mlir::PartTensorDistBackend> partTensorBackend{
      *this, "pt-backend",
      ::llvm::cl::desc("Backend to use for part tensor communication"),
      ::llvm::cl::init(mlir::PartTensorDistBackend::kNone),
      llvm::cl::values(
          clEnumValN(mlir::PartTensorDistBackend::kNone, "none",
                     "Turn off part tensor distribution."),
          clEnumValN(mlir::PartTensorDistBackend::kMPI, "mpi", "Use Mpi."),
          clEnumValN(mlir::PartTensorDistBackend::kKRS, "krs",
                     "Use Kokkos Remote Spaces."))};
#endif
};

//===----------------------------------------------------------------------===//
// Building and Registering.
//===----------------------------------------------------------------------===//

/// Adds the "sparse-compiler-kokkos" pipeline to the `OpPassManager`.  This
/// is the standard pipeline for taking sparsity-agnostic IR using
/// the sparse-tensor type and lowering it to the Kokkos dialect with concrete
/// representations and algorithms for sparse tensors.
void buildSparseKokkosCompiler(OpPassManager &pm, const LapisCompilerOptions &options);

/// Registers all pipelines for the `kokkos` dialect.
void registerKokkosPipelines();

} // namespace kokkos
} // namespace mlir

#endif // MLIR_KOKKOS_PIPELINES_PASSES_H_

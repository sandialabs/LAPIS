//===- Passes.h - Kokkos pipeline entry points -*-===//
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
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Dialect/Kokkos/Transforms/Passes.h"
#include "mlir/Pass/PassOptions.h"

using namespace mlir::detail;
using namespace llvm::cl;

namespace mlir {
namespace kokkos {

/// Options for the "sparse-compiler" pipeline.  So far this only contains
/// a subset of the options that can be set for the underlying passes,
/// because it must be manually kept in sync with the tablegen files
/// for those passes.
struct SparseCompilerOptions
    : public PassPipelineOptions<SparseCompilerOptions> {
  // These options must be kept in sync with `SparsificationBase`.
  // TODO(57514): These options are duplicated in Passes.td.
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
  PassOptions::Option<mlir::GPUDataTransferStrategy> gpuDataTransfer{
      *this, "gpu-data-transfer-strategy",
      ::llvm::cl::desc(
          "Set the data transfer strategy between the host and the GPUs"),
      ::llvm::cl::init(mlir::GPUDataTransferStrategy::kRegularDMA),
      llvm::cl::values(
          clEnumValN(mlir::GPUDataTransferStrategy::kRegularDMA, "regular-dma",
                     "Default option: malloc on host without additional "
                     "options or care and then use DMA to copy the data"),
          clEnumValN(mlir::GPUDataTransferStrategy::kPinnedDMA, "pinned-dma",
                     "Based on the default option, pin the host memory to "
                     "accelerate the data transfer"),
          clEnumValN(mlir::GPUDataTransferStrategy::kZeroCopy, "zero-copy",
                     "Use zero-copy to perform the data transfer from the host "
                     "to the GPU"))};

  PassOptions::Option<bool> enableIndexReduction{
      *this, "enable-index-reduction",
      desc("Enable dependent index reduction based algorithm to handle "
           "non-trivial index expressions on sparse inputs (experimental "
           "features)"),
      init(false)};

  PassOptions::Option<bool> enableRuntimeLibrary{
      *this, "enable-runtime-library",
      desc("Enable runtime library for manipulating sparse tensors"),
      init(true)};

  PassOptions::Option<bool> testBufferizationAnalysisOnly{
      *this, "test-bufferization-analysis-only",
      desc("Run only the inplacability analysis"), init(false)};

  PassOptions::Option<bool> enableBufferInitialization{
      *this, "enable-buffer-initialization",
      desc("Enable zero-initialization of memory buffers"), init(false)};

  PassOptions::Option<bool> createSparseDeallocs{
      *this, "create-sparse-deallocs",
      desc("Specify if the temporary buffers created by the sparse "
           "compiler should be deallocated. For compatibility with core "
           "bufferization passes. "
           "This option is only used when enable-runtime-library=false. "
           "See also create-deallocs for BufferizationOption."),
      init(true)};

  PassOptions::Option<int32_t> vectorLength{
      *this, "vl", desc("Set the vector length (0 disables vectorization)"),
      init(0)};

  // These options must be kept in sync with `SparseTensorConversionBase`.
  PassOptions::Option<int32_t> sparseToSparse{
      *this, "s2s-strategy",
      desc("Set the strategy for sparse-to-sparse conversion"), init(0)};

  // These options must be kept in sync with the `ConvertVectorToLLVM`
  // (defined in include/mlir/Dialect/SparseTensor/Pipelines/Passes.h).
  PassOptions::Option<bool> reassociateFPReductions{
      *this, "reassociate-fp-reductions",
      desc("Allows llvm to reassociate floating-point reductions for speed"),
      init(false)};
  PassOptions::Option<bool> force32BitVectorIndices{
      *this, "enable-index-optimizations",
      desc("Allows compiler to assume indices fit in 32-bit if that yields "
           "faster code"),
      init(true)};
  PassOptions::Option<bool> amx{
      *this, "enable-amx",
      desc("Enables the use of AMX dialect while lowering the vector dialect"),
      init(false)};
  PassOptions::Option<bool> armNeon{
      *this, "enable-arm-neon",
      desc("Enables the use of ArmNeon dialect while lowering the vector "
           "dialect"),
      init(false)};
  PassOptions::Option<bool> armSVE{
      *this, "enable-arm-sve",
      desc("Enables the use of ArmSVE dialect while lowering the vector "
           "dialect"),
      init(false)};
  PassOptions::Option<bool> x86Vector{
      *this, "enable-x86vector",
      desc("Enables the use of X86Vector dialect while lowering the vector "
           "dialect"),
      init(false)};

  /// These options are used to enable GPU code generation.
  PassOptions::Option<std::string> gpuTriple{*this, "gpu-triple",
                                             desc("GPU target triple")};
  PassOptions::Option<std::string> gpuChip{*this, "gpu-chip",
                                           desc("GPU target architecture")};
  PassOptions::Option<std::string> gpuFeatures{*this, "gpu-features",
                                               desc("GPU target features")};

  /// This option is used to enable GPU library generation.
  PassOptions::Option<bool> enableGPULibgen{
      *this, "enable-gpu-libgen",
      desc("Enables GPU acceleration by means of direct library calls (like "
           "cuSPARSE)")};

  /// Projects out the options for `createSparsificationPass`.
  SparsificationOptions sparsificationOptions() const {
    return SparsificationOptions(parallelization, gpuDataTransfer,
                                 enableIndexReduction, enableGPULibgen,
                                 enableRuntimeLibrary);
  }

  /// Projects out the options for `createSparseTensorConversionPass`.
  SparseTensorConversionOptions sparseTensorConversionOptions() const {
    return SparseTensorConversionOptions(
        sparseToSparseConversionStrategy(sparseToSparse));
  }

  /// Projects out the options for `createConvertVectorToLLVMPass`.
  ConvertVectorToLLVMPassOptions lowerVectorToLLVMOptions() const {
    ConvertVectorToLLVMPassOptions opts{};
    opts.reassociateFPReductions = reassociateFPReductions;
    opts.force32BitVectorIndices = force32BitVectorIndices;
    opts.armNeon = armNeon;
    opts.armSVE = armSVE;
    opts.amx = amx;
    opts.x86Vector = x86Vector;
    return opts;
  }
};

//===----------------------------------------------------------------------===//
// Building and Registering.
//===----------------------------------------------------------------------===//

/// Adds the "sparse-compiler-kokkos" pipeline to the `OpPassManager`.  This
/// is the standard pipeline for taking sparsity-agnostic IR using
/// the sparse-tensor type and lowering it to the Kokkos dialect with concrete
/// representations and algorithms for sparse tensors.
void buildSparseKokkosCompiler(OpPassManager &pm,
                         const SparseCompilerOptions &options);

/// Registers all pipelines for the `kokkos` dialect.
void registerKokkosPipelines();

} // namespace kokkos
} // namespace mlir

#endif // MLIR_KOKKOS_PIPELINES_PASSES_H_
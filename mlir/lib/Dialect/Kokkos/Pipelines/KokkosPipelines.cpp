//===- KokkosPipelines.cpp - Pipelines using the Kokkos dialect for sparse and dense tensors) -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lapis/Dialect/Kokkos/Pipelines/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "lapis/Dialect/Kokkos/IR/KokkosDialect.h"
#include "lapis/Dialect/Kokkos/Transforms/Passes.h"
#ifdef ENABLE_PART_TENSOR
#include "lapis/Dialect/PartTensor/Transforms/Passes.h"
#endif
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::kokkos;

//===----------------------------------------------------------------------===//
// Pipeline implementation.
//===----------------------------------------------------------------------===//

void mlir::kokkos::buildSparseKokkosCompiler(
    OpPassManager &pm, const LapisCompilerOptions& options) {
#ifdef ENABLE_PART_TENSOR
  pm.addPass(::mlir::createPartTensorConversionPass(options.partTensorBackend));
#endif
  // Rewrite named linalg ops into generic ops and apply fusion.
  pm.addNestedPass<func::FuncOp>(createLinalgGeneralizeNamedOpsPass());
  pm.addPass(createPreSparsificationRewritePass());
  pm.addNestedPass<func::FuncOp>(createLinalgElementwiseOpFusionPass());
  pm.addPass(createConvertShapeToStandardPass());
  pm.addPass(createSparseAssembler());

  // Set up options for sparsification.
  // The only option exposed by LapisCompilerOptions is the parallelization strategy.
  SparsificationOptions sparseOptions(
      options.parallelization,
      mlir::SparseEmitStrategy::kFunctional,
      /* enableRuntimeLibrary*/ true);

  // Sparsification and bufferization mini-pipeline.
  pm.addPass(createSparsificationAndBufferizationPass(
      getBufferizationOptionsForSparsification(false),
      sparseOptions,
      /* createSparseDeallocs */ true,
      /* enableRuntimeLibrary */ true,
      /* enableBufferInitialization */ false,
      /* vectorLength */ 0,
      /* enableVLAVectorization */ false,
      /* enableSIMDIndex32 */ false,
      /* enableGPULibgen */ false,
      sparseOptions.sparseEmitStrategy));

  /*
"func.func(sparse-encoding-propagation)",
# MLIR Sparsifier mini-pipeline:
#   use the PyTorch assembler conventions
#   enable vectorization with VL=16 (more or less assumes AVX512 for float)
#   allow 32-bit index optimizations (unsafe for very large dimensions)
"sparse-assembler{{direct-out}}",
"sparsification-and-bufferization{{{sp_options}}}",
"sparse-storage-specifier-to-llvm",
# Buffer deallocation pass does not know how to handle realloc.
"func.func(expand-realloc)",
# Generalize pad and concat after sparse compiler, as they are handled
# differently when the operations involve sparse operands.
"func.func(refback-generalize-tensor-pad)",
"func.func(refback-generalize-tensor-concat)",
# Bufferize.
"func.func(tm-tensor-bufferize)",
"one-shot-bufferize{{copy-before-write bufferize-function-boundaries function-boundary-type-conver    sion=identity-layout-map}}",
"refback-mlprogram-bufferize",
"func.func(finalizing-bufferize)",
"func.func(buffer-deallocation)",
# Inline sparse helper methods where useful (but after dealloc).
"inline",
"refback-munge-calling-conventions",
"func.func(tm-tensor-to-loops)",
"func.func(refback-munge-memref-copy)",
"func.func(convert-linalg-to-loops)",
"func.func(lower-affine)",

   */

  // Storage specifier lowering and bufferization wrap-up.
  pm.addPass(createStorageSpecifierToLLVMPass());
  pm.addNestedPass<func::FuncOp>(memref::createExpandReallocPass());

  // Note: these options are taken from the MPACTBackend pipeline
  bufferization::OneShotBufferizationOptions buffOptions;
  buffOptions.copyBeforeWrite = true;
  buffOptions.bufferizeFunctionBoundaries = true;
  buffOptions.setFunctionBoundaryTypeConversion(bufferization::LayoutMapOption::IdentityLayoutMap);
  pm.addPass(createOneShotBufferizePass(buffOptions));

  pm.addNestedPass<func::FuncOp>(bufferization::createFinalizingBufferizePass());
  //pm.addPass(bufferization::createBufferDeallocationPass());

  pm.addPass(createInlinerPass());

  //pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());

  // Progressively lower to LLVM. Note that the convert-vector-to-llvm
  // pass is repeated on purpose.
  // TODO(springerm): Add sparse support to the BufferDeallocation pass and add
  // it to this pipeline.
  pm.addNestedPass<func::FuncOp>(createConvertLinalgToParallelLoopsPass());
  pm.addNestedPass<func::FuncOp>(createConvertVectorToSCFPass());
  pm.addPass(memref::createExpandStridedMetadataPass());

  pm.addPass(createLowerAffinePass());
  pm.addNestedPass<func::FuncOp>(createConvertComplexToStandardPass());
  pm.addNestedPass<func::FuncOp>(arith::createArithExpandOpsPass());
  // Lower SCF to Kokkos dialect
  pm.addPass(createParallelUnitStepPass());
  pm.addPass(createKokkosLoopMappingPass());
  //pm.addPass(createKokkosMemorySpaceAssignmentPass());
  pm.addPass(createKokkosDualViewManagementPass());

  // Ensure all casts are realized.
  pm.addPass(createReconcileUnrealizedCastsPass());
}

//===----------------------------------------------------------------------===//
// Pipeline registration.
//===----------------------------------------------------------------------===//

void mlir::kokkos::registerKokkosPipelines() {
  PassPipelineRegistration<LapisCompilerOptions>(
      "sparse-compiler-kokkos",
      "The standard pipeline for taking sparsity-agnostic IR using the"
      " sparse-tensor type, and lowering it to dialects compatible with the Kokkos emitter",
      buildSparseKokkosCompiler);
}


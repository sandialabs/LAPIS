//===- KokkosPipelines.cpp - Pipelines using the Kokkos dialect for sparse and dense tensors) -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lapis/LAPIS_config.h"
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
#ifdef LAPIS_ENABLE_PART_TENSOR
#include "lapis/Dialect/PartTensor/Transforms/Passes.h"
#endif
#ifdef LAPIS_HAS_TORCH_MLIR
#include "torch-mlir-dialects/Dialect/TMTensor/Transforms/Passes.h"
#include "torch-mlir/RefBackend/Passes.h"
#endif
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include <sstream>

using namespace mlir;
using namespace mlir::kokkos;

//===----------------------------------------------------------------------===//
// Pipeline implementation.
//===----------------------------------------------------------------------===//

void mlir::kokkos::buildSparseKokkosCompiler(
    OpPassManager &pm, const LapisCompilerOptions& options) {
  bool enableRuntimeLib = !options.decompose;
#ifdef LAPIS_ENABLE_PART_TENSOR
  pm.addPass(::mlir::createPartTensorConversionPass(options.partTensorBackend));
#endif

  // Rewrite named linalg ops into generic ops and apply fusion.
  pm.addNestedPass<func::FuncOp>(createLinalgGeneralizeNamedOpsPass());

  // Remove compile-time unit extent dimensions from linalg ops.
  // For example, a 3D loop over (N, M, 1) will be rewritten to 2D loop over (N, M).
  // This does not affect tensor types, at least in function parameter/return types,
  // so it is transparent to any caller.
  pm.addPass(createLinalgFoldUnitExtentDimsPass());

  if(options.decompose) {
    pm.addPass(createPreSparsificationRewritePass());
  }

  pm.addNestedPass<func::FuncOp>(createLinalgElementwiseOpFusionPass());
  pm.addPass(createConvertShapeToStandardPass());

  // Set up options for sparsification.
  // The only option exposed by LapisCompilerOptions is the parallelization strategy.
  // TODO: enableRuntimeLibrary = false when decompose = true?
  SparsificationOptions sparseOptions(
      options.parallelization,
      mlir::SparseEmitStrategy::kFunctional,
      /* enableRuntimeLibrary*/ enableRuntimeLib);

  // Sparsification and bufferization mini-pipeline.
  pm.addPass(createSparsificationAndBufferizationPass(
        getBufferizationOptionsForSparsification(false),
        sparseOptions,
        /* createSparseDeallocs */ false,
        /* enableRuntimeLibrary */ enableRuntimeLib,
        /* enableBufferInitialization */ false,
        /* vectorLength */ 0,
        /* enableVLAVectorization */ false,
        /* enableSIMDIndex32 */ false,
        /* enableGPULibgen */ false,
        sparseOptions.sparseEmitStrategy));

  // Storage specifier lowering and bufferization wrap-up.
  pm.addPass(createStorageSpecifierToLLVMPass());

  pm.addNestedPass<func::FuncOp>(memref::createExpandReallocPass());

#ifdef LAPIS_HAS_TORCH_MLIR
  pm.addNestedPass<func::FuncOp>(torch::RefBackend::createGeneralizeTensorPadPass());
  pm.addNestedPass<func::FuncOp>(torch::RefBackend::createGeneralizeTensorConcatPass());
  pm.addNestedPass<func::FuncOp>(torch::TMTensor::createTMTensorBufferizePass());
#endif

  // All one-shot bufferization options are copied from the TM linalg-on-tensors pipeline
  bufferization::OneShotBufferizationOptions oneShotBuffOptions;
  oneShotBuffOptions.copyBeforeWrite = true;
  oneShotBuffOptions.bufferizeFunctionBoundaries = true;
  oneShotBuffOptions.setFunctionBoundaryTypeConversion(bufferization::LayoutMapOption::IdentityLayoutMap);
  pm.addPass(createOneShotBufferizePass(oneShotBuffOptions));

#ifdef LAPIS_HAS_TORCH_MLIR
  pm.addPass(torch::RefBackend::createMLProgramBufferizePass());
#endif

  pm.addNestedPass<func::FuncOp>(bufferization::createFinalizingBufferizePass());
  //pm.addNestedPass<func::FuncOp>(bufferization::createBufferDeallocationPass());

  pm.addPass(createInlinerPass());

#ifdef LAPIS_HAS_TORCH_MLIR
  pm.addNestedPass<func::FuncOp>(torch::TMTensor::createTMTensorToLoopsPass());
#endif

  pm.addNestedPass<func::FuncOp>(createConvertLinalgToParallelLoopsPass());

// Not needed anymore?
//  pm.addNestedPass<func::FuncOp>(createConvertVectorToSCFPass());

  pm.addNestedPass<func::FuncOp>(arith::createArithExpandOpsPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(createLowerAffinePass());

  pm.addNestedPass<func::FuncOp>(createConvertComplexToStandardPass());
  // Ensure all casts are realized.
  pm.addPass(createReconcileUnrealizedCastsPass());

  // Finally, lower scf/memref to kokkos
  pm.addPass(createParallelUnitStepPass());
  pm.addPass(createKokkosLoopMappingPass());
  //pm.addPass(createKokkosMemorySpaceAssignmentPass());
  pm.addPass(createKokkosDualViewManagementPass());

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


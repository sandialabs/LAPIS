//===- KokkosPipelines.cpp - Pipelines using the Kokkos dialect for sparse and dense tensors) -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Kokkos/Pipelines/Passes.h"

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
#include "mlir/Dialect/Kokkos/IR/KokkosDialect.h"
#include "mlir/Dialect/Kokkos/Transforms/Passes.h"
#ifdef ENABLE_PART_TENSOR
#include "mlir/Dialect/PartTensor/Transforms/Passes.h"
#endif
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::kokkos;

//===----------------------------------------------------------------------===//
// Pipeline implementation.
//===----------------------------------------------------------------------===//

// Not working yet: new Kokkos dialect based pipeline.
void mlir::kokkos::buildSparseKokkosCompiler(
    OpPassManager &pm, const SparseCompilerOptions &options) {
#ifdef ENABLE_PART_TENSOR
  pm.addPass(::mlir::createPartTensorConversionPass());
#endif
  pm.addNestedPass<func::FuncOp>(createLinalgGeneralizationPass());
  pm.addPass(createSparsificationAndBufferizationPass(
      getBufferizationOptionsForSparsification(
          options.testBufferizationAnalysisOnly),
      options.sparsificationOptions(), options.sparseTensorConversionOptions(),
      options.createSparseDeallocs, options.enableRuntimeLibrary,
      options.enableBufferInitialization, options.vectorLength,
      options.armSVE,
      options.force32BitVectorIndices));
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(
      mlir::bufferization::createFinalizingBufferizePass());
  pm.addPass(createLinalgFoldUnitExtentDimsPass());
  pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
  pm.addNestedPass<func::FuncOp>(createConvertVectorToSCFPass());
  pm.addNestedPass<func::FuncOp>(memref::createExpandReallocPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(createLowerAffinePass());
  // Lower SCF to Kokkos dialect
  pm.addPass(createParallelUnitStepPass());
  pm.addPass(createKokkosLoopMappingPass());
  //pm.addPass(createKokkosMemorySpaceAssignmentPass());
  pm.addPass(createKokkosDualViewManagementPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  // Apply CSE (common subexpression elimination) now, since the
  // output of this pipeline gets fed directly into the Kokkos C++ emitter.
  pm.addPass(createCSEPass());
}

//===----------------------------------------------------------------------===//
// Pipeline registration.
//===----------------------------------------------------------------------===//

void mlir::kokkos::registerKokkosPipelines() {
  PassPipelineRegistration<SparseCompilerOptions>(
      "sparse-compiler-kokkos",
      "The standard pipeline for taking sparsity-agnostic IR using the"
      " sparse-tensor type, and lowering it to dialects compatible with the Kokkos emitter",
      buildSparseKokkosCompiler);
}


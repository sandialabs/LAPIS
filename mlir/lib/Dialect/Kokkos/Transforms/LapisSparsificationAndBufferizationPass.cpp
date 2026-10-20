//===- LapisSparsificationAndBufferizationPass.cpp -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "lapis/Dialect/Kokkos/Transforms/Passes.h"

using namespace mlir;

namespace mlir {

#define GEN_PASS_DEF_LAPISSPARSIFICATIONANDBUFFERIZATION
#include "lapis/Dialect/Kokkos/Transforms/Passes.h.inc"

namespace kokkos {

/// Return `true` if one of the given types is a sparse tensor type.
static bool containsSparseTensor(TypeRange types) {
  for (Type t : types)
    if (isa<TensorType>(t) && mlir::sparse_tensor::getSparseTensorEncoding(t))
      return true;
  return false;
}

/// A pass that lowers tensor ops to memref ops, regardless of whether they are
/// dense or sparse.
///
/// One-Shot Analysis is used to detect RaW conflicts and to insert buffer
/// copies of the tensor level (`insertTensorCopies`). Afterwards, the lowering
/// of tensor ops to memref ops follows a different code path depending on
/// whether the op is sparse or dense:
///
/// * Sparse tensor ops are lowered through Sparsification and follow-up pass
///   that lowers sparse_tensor dialect ops.
/// * Dense tensor ops are lowered through BufferizableOpInterface
///   implementations.
class LapisSparsificationAndBufferizationPass
    : public impl::LapisSparsificationAndBufferizationBase<
          LapisSparsificationAndBufferizationPass> {
public:
  // Private pass options only.
  LapisSparsificationAndBufferizationPass(
      const bufferization::OneShotBufferizationOptions &bufferizationOptions,
      const SparsificationOptions &sparsificationOptions,
      bool createSparseDeallocs, bool enableRuntimeLibrary,
      bool enableBufferInitialization)
      : bufferizationOptions(bufferizationOptions),
        sparsificationOptions(sparsificationOptions),
        createSparseDeallocs(createSparseDeallocs),
        enableRuntimeLibrary(enableRuntimeLibrary),
        enableBufferInitialization(enableBufferInitialization) {}
  // Private pass options and visible pass options.
  LapisSparsificationAndBufferizationPass(
      const bufferization::OneShotBufferizationOptions &bufferizationOptions,
      const SparsificationOptions &sparsificationOptions,
      bool createSparseDeallocs, bool enableRuntimeLibrary,
      bool enableBufferInitialization, unsigned vl, bool vla, bool index32,
      bool gpu, SparseEmitStrategy emitStrategy,
      SparseParallelizationStrategy parallelizationStrategy)
      : bufferizationOptions(bufferizationOptions),
        sparsificationOptions(sparsificationOptions),
        createSparseDeallocs(createSparseDeallocs),
        enableRuntimeLibrary(enableRuntimeLibrary),
        enableBufferInitialization(enableBufferInitialization) {
    // Set the visible pass options explicitly.
    vectorLength = vl;
    enableVLAVectorization = vla;
    enableSIMDIndex32 = index32;
    enableGPULibgen = gpu;
    sparseEmitStrategy = emitStrategy;
    parallelization = parallelizationStrategy;
  }

  /// Bufferize all dense ops. This assumes that no further analysis is needed
  /// and that all required buffer copies were already inserted by
  /// `insertTensorCopies` in the form of `bufferization.alloc_tensor` ops.
  LogicalResult runDenseBufferization() {
    bufferization::OneShotBufferizationOptions updatedOptions =
        bufferizationOptions;
    // Skip all sparse ops.
    updatedOptions.opFilter.denyOperation([&](Operation *op) {
      if (containsSparseTensor(TypeRange(op->getResults())) ||
          containsSparseTensor(TypeRange(op->getOperands())))
        return true;
      if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
        FunctionType funcType = funcOp.getFunctionType();
        if (containsSparseTensor(funcType.getInputs()) ||
            containsSparseTensor(funcType.getResults()))
          return true;
      }
      return false;
    });

    bufferization::BufferizationState bufferizationState;

    if (failed(bufferization::bufferizeModuleOp(getOperation(), updatedOptions,
                                                 bufferizationState)))
      return failure();

    bufferization::removeBufferizationAttributesInModule(getOperation());
    return success();
  }

  void runOnOperation() override {
    // Overrides the default emit strategy using user-provided value.
    this->sparsificationOptions.sparseEmitStrategy = sparseEmitStrategy;

    // Overrides the default parallelization strategy using user-provided value.
    this->sparsificationOptions.parallelizationStrategy = parallelization;

    // Run enabling transformations.
    {
      OpPassManager pm("builtin.module");
      pm.addPass(createPreSparsificationRewritePass());
      pm.addNestedPass<func::FuncOp>(
          bufferization::createEmptyTensorToAllocTensorPass());
      if (failed(runPipeline(pm, getOperation())))
        return signalPassFailure();
    }

    // Insert tensor copies. This step runs One-Shot Analysis (which analyzes
    // SSA use-def chains of tensor IR) and decides where buffer copies are
    // needed and where buffers can be written to in-place. These decisions are
    // materialized in the IR in the form of `bufferization.alloc_tensor` ops.
    //
    // Note: All following steps in this pass must be careful not to modify the
    // structure of the IR (i.e., tensor use-def chains), as that could
    // invalidate the results of the analysis. From now on, only small and
    // localized rewrites are allowed, such as replacing a tensor op with its
    // memref equivalent.
    bufferization::BufferizationState bufferizationState;

    if (failed(bufferization::insertTensorCopies(
            getOperation(), bufferizationOptions, bufferizationState)))
      return signalPassFailure();

    // Option `testAnalysisOnly` is a debug/testing flag. If set, the results of
    // OneShotAnalysis are added to the IR via attributes. In that case, do not
    // continue with the remaining pipeline.
    if (bufferizationOptions.testAnalysisOnly)
      return;

    // Bufferize all sparse ops. No further analysis is needed. All required
    // buffer copies were already inserted by `insertTensorCopies` in the form
    // of `bufferization.alloc_tensor` ops.
    {
      OpPassManager pm("builtin.module");
      if (enableGPULibgen)
        pm.addPass(createSparseGPUCodegenPass(0, enableRuntimeLibrary));
      pm.addPass(createSparseReinterpretMapPass(ReinterpretMapScope::kAll));
      pm.addPass(createSparsificationPass(sparsificationOptions));
      if (sparsificationOptions.sparseEmitStrategy ==
          SparseEmitStrategy::kSparseIterator) {
        pm.addNestedPass<func::FuncOp>(createSparseSpaceCollapsePass());
        pm.addNestedPass<func::FuncOp>(createLowerSparseIterationToSCFPass());
      }

      pm.addNestedPass<func::FuncOp>(createStageSparseOperationsPass());
      pm.addPass(createLowerSparseOpsToForeachPass(enableRuntimeLibrary,
                                                   /*enableConvert=*/true));
      pm.addPass(
          createSparseReinterpretMapPass(ReinterpretMapScope::kExceptGeneric));
      pm.addNestedPass<func::FuncOp>(createLowerForeachToSCFPass());
      // Note: LoopInvariantCodeMotion is not run when LAPIS pipeline decomposes tensors
      // (!enableRuntimeLibrary) because it incorrectly hoists
      // linalg.fill operations out of loops, corrupting buffers that are
      // written inside the loop. This caused LAPIS pcg_solve and hitting_times tests to fail
      // with the upstream SparsificationAndBufferizationPass.
      if (enableRuntimeLibrary) {
        pm.addPass(mlir::createLoopInvariantCodeMotionPass());
      }
      if (vectorLength > 0) {
        pm.addPass(createSparseVectorizationPass(
            vectorLength, enableVLAVectorization, enableSIMDIndex32));
      }
      if (enableRuntimeLibrary) {
        pm.addPass(createSparseTensorConversionPass());
      } else {
        pm.addPass(createSparseTensorCodegenPass(createSparseDeallocs,
                                                 enableBufferInitialization));
        pm.addPass(createSparseBufferRewritePass(enableBufferInitialization));
      }
      if (failed(runPipeline(pm, getOperation())))
        return signalPassFailure();
    }

    // Bufferize all dense ops.
    if (failed(runDenseBufferization()))
      signalPassFailure();
  }

private:
  bufferization::OneShotBufferizationOptions bufferizationOptions;
  SparsificationOptions sparsificationOptions;
  bool createSparseDeallocs;
  bool enableRuntimeLibrary;
  bool enableBufferInitialization;
};

} // namespace kokkos
} // namespace mlir

std::unique_ptr<mlir::Pass> mlir::createLapisSparsificationAndBufferizationPass() {
  SparsificationOptions sparseOptions;
  return std::make_unique<
      mlir::kokkos::LapisSparsificationAndBufferizationPass>(
      getBufferizationOptionsForSparsification(/*analysisOnly=*/false),
      sparseOptions,
      /*createSparseDeallocs=*/false,
      /*enableRuntimeLibrary=*/false,
      /*enableBufferInitialization=*/false);
}

std::unique_ptr<mlir::Pass> mlir::createLapisSparsificationAndBufferizationPass(
    const bufferization::OneShotBufferizationOptions &bufferizationOptions,
    const SparsificationOptions &sparsificationOptions,
    bool createSparseDeallocs, bool enableRuntimeLibrary,
    bool enableBufferInitialization, unsigned vectorLength,
    bool enableVLAVectorization, bool enableSIMDIndex32, bool enableGPULibgen,
    SparseEmitStrategy emitStrategy,
    SparseParallelizationStrategy parallelizationStrategy) {
  return std::make_unique<
      mlir::kokkos::LapisSparsificationAndBufferizationPass>(
      bufferizationOptions, sparsificationOptions, createSparseDeallocs,
      enableRuntimeLibrary, enableBufferInitialization, vectorLength,
      enableVLAVectorization, enableSIMDIndex32, enableGPULibgen, emitStrategy,
      parallelizationStrategy);
}

//===- Passes.h - Kokkos passes ----------------===//
//
// This header file defines prototypes of all kokkos passes.
//
// In general, this file takes the approach of keeping "mechanism" (the
// actual steps of applying a transformation) completely separate from
// "policy" (heuristics for when and where to apply transformations).
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_KOKKOS_PASSES_H_
#define MLIR_DIALECT_KOKKOS_PASSES_H_

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"

namespace mlir {

#define GEN_PASS_DECL
#include "lapis/Dialect/Kokkos/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createTransposeConstantFoldPass();

std::unique_ptr<Pass> createDenseLinalgToParallelLoopsPass();

std::unique_ptr<Pass> createMemrefResultsToParamsPass();

std::unique_ptr<Pass> createMemrefCopyToParallelPass();

std::unique_ptr<Pass> createMemrefToKokkosScratchPass();

std::unique_ptr<Pass> createParallelUnitStepPass();

std::unique_ptr<Pass> createKokkosLoopMappingPass(bool teamLevel = false);

std::unique_ptr<Pass> createKokkosDualViewManagementPass();

std::unique_ptr<Pass> createKokkosMdrangeIterationPass();

std::unique_ptr<Pass> createLapisSparsificationAndBufferizationPass();

std::unique_ptr<Pass> createLapisSparsificationAndBufferizationPass(
    const bufferization::OneShotBufferizationOptions &bufferizationOptions,
    const SparsificationOptions &sparsificationOptions,
    bool createSparseDeallocs, bool enableRuntimeLibrary,
    bool enableBufferInitialization, unsigned vectorLength,
    bool enableVLAVectorization, bool enableSIMDIndex32, bool enableGPULibgen,
    SparseEmitStrategy emitStrategy,
    SparseParallelizationStrategy parallelizationStrategy);

//===----------------------------------------------------------------------===//
// Registration.
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "lapis/Dialect/Kokkos/Transforms/Passes.h.inc"

} // namespace mlir

#endif // MLIR_DIALECT_KOKKOS_PASSES_H_

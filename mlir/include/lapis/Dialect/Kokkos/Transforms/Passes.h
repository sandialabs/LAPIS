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

namespace mlir {

#define GEN_PASS_DECL
#include "lapis/Dialect/Kokkos/Transforms/Passes.h.inc"

void populateParallelUnitStepPatterns(RewritePatternSet &patterns);
std::unique_ptr<Pass> createParallelUnitStepPass();

void populateKokkosLoopMappingPatterns(RewritePatternSet &patterns);
std::unique_ptr<Pass> createKokkosLoopMappingPass();

void populateKokkosMemorySpaceAssignmentPatterns(RewritePatternSet &patterns);
std::unique_ptr<Pass> createKokkosMemorySpaceAssignmentPass();

std::unique_ptr<Pass> createKokkosDualViewManagementPass();

std::unique_ptr<Pass> createKokkosMdrangeIterationPass();

//===----------------------------------------------------------------------===//
// Registration.
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "lapis/Dialect/Kokkos/Transforms/Passes.h.inc"

} // namespace mlir

#endif // MLIR_DIALECT_KOKKOS_PASSES_H_

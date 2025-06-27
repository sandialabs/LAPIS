//===-- Passes.td - Kokkos pass definition file -----*- tablegen -*-===//

#ifndef MLIR_DIALECT_KOKKOS_TRANSFORMS_PASSES
#define MLIR_DIALECT_KOKKOS_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def MemrefResultsToParams : Pass<"memref-results-to-params"> {
  let summary = "Moves memref-type return values to be function parameters instead";
  let description = [{
    For each non-private function with at least one memref-typed return value, remove the return value and add
    a parameter with the same type. Each return value of the original function must be the result of a memref.alloc operation.
    Replace all usages of this allocation with the new parameter, and delete the allocation.
    The user/caller is then responsible for passing in a view with the correct size.
  }];
  let constructor = "mlir::createMemrefResultsToParamsPass()";
  let dependentDialects = [
    "memref::MemRefDialect",
    "func::FuncDialect"
  ];
}

def MemrefToKokkosScratch : Pass<"memref-to-kokkos-scratch"> {
  let summary = "Moves memref-type return values to be function parameters instead";
  let description = [{
    For each non-private function with at least one memref-typed return value, remove the return value and add
    a parameter with the same type. Each return value of the original function must be the result of a memref.alloc operation.
    Replace all usages of this allocation with the new parameter, and delete the allocation.
    The user/caller is then responsible for passing in a view with the correct size.
  }];
  let constructor = "mlir::createMemrefToKokkosScratchPass()";
  let dependentDialects = [
    "memref::MemRefDialect",
    "func::FuncDialect",
    "kokkos::KokkosDialect"
  ];
}

def ParallelUnitStep : Pass<"parallel-unit-step"> {
  let summary = "Rewrites scf.parallel ops to have lower bound 0 and step size 1 in all dimensions";
  let description = [{
  }];
  let constructor = "mlir::createParallelUnitStepPass()";
  let dependentDialects = [
    "arith::ArithDialect",
    "scf::SCFDialect"
  ];
}

def KokkosLoopMapping : Pass<"kokkos-loop-mapping"> {
  let summary = "Maps nested SCF parallel loops to Kokkos parallel loops";
  let description = [{
  }];
  let options = [
    Option<"teamLevel", "team-level", "bool", /*default=*/"false",
           "Whether to generate code to be called within a TeamPolicy">
  ];
  let dependentDialects = [
    "arith::ArithDialect",
    "kokkos::KokkosDialect",
    "func::FuncDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect"
  ];
}

def KokkosDualViewManagement : Pass<"kokkos-dualview-management", "func::FuncOp"> {
  let summary = "Insert sync and modify operations for DualView-typed memrefs";
  let description = [{
  }];
  let constructor = "mlir::createKokkosDualViewManagementPass()";
  let dependentDialects = [
    "arith::ArithDialect",
    "func::FuncDialect",
    "kokkos::KokkosDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect"
  ];
}

#endif // MLIR_DIALECT_KOKKOS_TRANSFORMS_PASSES

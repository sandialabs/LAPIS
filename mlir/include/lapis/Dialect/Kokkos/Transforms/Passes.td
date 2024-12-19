//===-- Passes.td - Kokkos pass definition file -----*- tablegen -*-===//

#ifndef MLIR_DIALECT_KOKKOS_TRANSFORMS_PASSES
#define MLIR_DIALECT_KOKKOS_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

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
  let constructor = "mlir::createKokkosLoopMappingPass()";
  let dependentDialects = [
    "arith::ArithDialect",
    "kokkos::KokkosDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect"
  ];
}

def KokkosMemorySpaceAssignment : Pass<"kokkos-assign-memory-spaces"> {
  let summary = "Determines the MemorySpace attributes for each memref value";
  let description = [{
  }];
  let constructor = "mlir::createKokkosMemorySpaceAssignmentPass()";
  let dependentDialects = [
    "arith::ArithDialect",
    "kokkos::KokkosDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect"
  ];
}

def KokkosDualViewManagement : Pass<"kokkos-dualview-management", "ModuleOp"> {
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

def KokkosMdrangeIteration : Pass<"kokkos-mdrange-iteration", "ModuleOp"> {
  let summary = "Rearange MDRange to improve memory access patterns on GPU";
  let description = [{
  }];
  let constructor = "mlir::createKokkosMdrangeIterationPass()";
  let dependentDialects = [
    "arith::ArithDialect",
    "func::FuncDialect",
    "kokkos::KokkosDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect"
  ];
}

#endif // MLIR_DIALECT_KOKKOS_TRANSFORMS_PASSES

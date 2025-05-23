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

def SparseAssemblerDirectOut : Pass<"sparse-assembler-direct-out", "ModuleOp"> {
  let summary = "Add [dis]assemble operations on external sparse tensors";
  let description = [{
    Unlike dense tensors, MLIR does **not** provide a direct `_mlir_ciface_`
    ABI for passing sparse tensors as arguments from and to external methods
    (within MLIR-generated methods, sparse tensors can be freely passed
    around, but this eventually uses a bespoke parameter passing format
    that is subject to change; like opaque pointers when the sparse runtime
    support library is used or the constituent arrays and structs for 
    direct IR codegen). The sparse assembler pass, however, can be used
    to obtain a stable `_mlir_ciface_` API for passing sparse tensors
    from and to an external environment, such as Python, PyTorch, or JAX.

    The pass converts public entry methods that use sparse tensors as
    input parameters and/or output return values into wrapper methods
    that [dis]assemble the individual tensors that constitute the actual
    storage used externally into MLIR sparse tensors. This pass can be used
    to prepare the public entry methods of a program that is compiled by the 
    MLIR sparsifier to interface with an external runtime, e.g., when passing
    sparse tensors as numpy arrays from and to Python. Note that eventual
    bufferization decisions (e.g. who [de]allocates the underlying memory)
    should be resolved in agreement with the external runtime.

    By default, the pass uses the [dis]assemble operations to input and output
    sparse tensors. When the direct-out option is set, however, the output
    directly returns the MLIR allocated buffers to the external runtime.

    The pass should always run before the actual sparsification passes.
  }]; 
  let constructor = "mlir::createSparseAssemblerDirectOutPass()";
  let dependentDialects = [ 
    "bufferization::BufferizationDialect",
    "sparse_tensor::SparseTensorDialect",
    "tensor::TensorDialect",
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

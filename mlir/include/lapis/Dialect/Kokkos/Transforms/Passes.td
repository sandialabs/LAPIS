//===-- Passes.td - Kokkos pass definition file -----*- tablegen -*-===//

#ifndef MLIR_DIALECT_KOKKOS_TRANSFORMS_PASSES
#define MLIR_DIALECT_KOKKOS_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def TransposeConstantFoldPass : Pass<"fold-constant-transpose"> {
  let summary = "Fold linalg.transpose ops with constant operands";
  let dependentDialects = [
    "affine::AffineDialect", "linalg::LinalgDialect", "memref::MemRefDialect"
  ];
  let constructor = "mlir::createTransposeConstantFoldPass()";
}

def DenseLinalgToParallelLoops : Pass<"dense-linalg-to-parallel-loops"> {
  let summary = "Rewrite dense linalg.generic operations to scf.parallel loop nests";
  let dependentDialects = [
    "scf::SCFDialect", "linalg::LinalgDialect", "memref::MemRefDialect"
  ];
  let constructor = "mlir::createDenseLinalgToParallelLoopsPass()";
}

def MemrefResultsToParams : Pass<"memref-results-to-params", "ModuleOp"> {
  let summary = "Moves memref-type return values to be function parameters instead";
  let description = [{
    For each non-private function with at least one memref-typed return value, remove the return value and add
    a parameter with the same type. Each return value of the original function must be the result of a memref.alloc operation,
    or an alias of an allocation.
    Replace all usages of this allocation with the new parameter, and delete the allocation.
    Also remove any return values which are equivalent to a newly added parameter.
    The user/caller is then responsible for passing in a view with the correct size.
    This is similar but not equivalent to the built-in passes ``--buffer-results-to-out-params --drop-equivalent-buffer-results``.

    That pass updates call sites for the new interface, but does not eliminate memref allocations for
    results (the whole point of this pass) and deep-copies from the allocated buffer to the argument.
    There does not seem to be a built-in pass to eliminate allocations that are used in this way.
    This pass requires that no public function has a call site in the module (it is intended to be run
    after inlining, and only in modules without recursion).
  }];
  let constructor = "mlir::createMemrefResultsToParamsPass()";
  let dependentDialects = [
    "memref::MemRefDialect",
    "func::FuncDialect"
  ];
}

def MemrefCopyToParallel : Pass<"memref-copy-to-parallel"> {
  let summary = "Lowers memref.copy ops to parallel loops and load/store";
  let constructor = "mlir::createMemrefCopyToParallelPass()";
  let dependentDialects = [
    "memref::MemRefDialect",
    "scf::SCFDialect"
  ];
}

def MemrefToKokkosScratch : Pass<"memref-to-kokkos-scratch", "func::FuncOp"> {
  let summary = "Replaces memref-type return values to be function parameters instead";
  let description = [{
    Replace each memref.alloc op with kokkos.alloc_scratch. This analyzes lifetimes of each
    allocation (including aliasing memrefs) to determine which memrefs can occupy the same
    scratch, and then greedily places the scratch allocations. Each allocation's address is
    determined at compile-time (as an attribute) and will be aligned properly for the memref element type.
    This pass makes simplifying assumptions about the input program: function arguments are assumed to be "restrict"
    and it is assumed that this pass is run after ``memref-results-to-params`` so that all allocations inside the
    function have a lifetime that ends inside the function. This is why the pass has its own memref lifetime determination,
    instead of using bufferization's passes for inserting deallocations.
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

def LapisSparsificationAndBufferization : Pass<"lapis-sparsification-and-bufferization", "ModuleOp"> {
  let summary = "LAPIS-patched mini-pipeline that combines bufferization and sparsification";
  let description = [{
    LAPIS-maintained copy of LLVM's SparsificationAndBufferization pass with
    a patched LICM behavior to avoid incorrect hoisting of linalg.fill
    operations (see LAPIS pcg_solve issue).
  }];
  let constructor = "mlir::createLapisSparsificationAndBufferizationPass()";
  let dependentDialects = [
    "affine::AffineDialect",
    "arith::ArithDialect",
    "bufferization::BufferizationDialect",
    "gpu::GPUDialect",
    "LLVM::LLVMDialect",
    "linalg::LinalgDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect",
    "sparse_tensor::SparseTensorDialect",
    "vector::VectorDialect"
  ];
  let options = [
    Option<"vectorLength", "vl", "int32_t", "0",
           "Set the vector length (use 0 to disable vectorization)">,
    Option<"enableVLAVectorization", "enable-vla-vectorization", "bool", "false",
           "Enable vector length agnostic vectorization">,
    Option<"enableSIMDIndex32", "enable-simd-index32", "bool", "false",
           "Enable i32 indexing into vectors (for efficient gather/scatter)">,
    Option<"enableGPULibgen", "enable-gpu-libgen", "bool", "false",
           "Enable GPU acceleration by means of direct library calls">,
    Option<"sparseEmitStrategy", "sparse-emit-strategy", "mlir::SparseEmitStrategy",
           "mlir::SparseEmitStrategy::kFunctional",
           "Emit functional code or interfaces (to debug) for sparse loops", [{llvm::cl::values(
             clEnumValN(mlir::SparseEmitStrategy::kFunctional, "functional",
                        "Emit functional code (with scf.for/while)."),
             clEnumValN(mlir::SparseEmitStrategy::kSparseIterator, "sparse-iterator",
                        "Emit (experimental) loops (with sparse.iterate)."),
             clEnumValN(mlir::SparseEmitStrategy::kDebugInterface, "debug-interface",
                        "Emit non-functional but easy-to-read interfaces to debug."))}]>,
    Option<"parallelization", "parallelization-strategy", "mlir::SparseParallelizationStrategy",
           "mlir::SparseParallelizationStrategy::kNone",
           "Set the parallelization strategy", [{llvm::cl::values(
             clEnumValN(mlir::SparseParallelizationStrategy::kNone, "none",
                        "Turn off sparse parallelization."),
             clEnumValN(mlir::SparseParallelizationStrategy::kDenseOuterLoop,
                        "dense-outer-loop",
                        "Enable dense outer loop sparse parallelization."),
             clEnumValN(mlir::SparseParallelizationStrategy::kAnyStorageOuterLoop,
                        "any-storage-outer-loop",
                        "Enable sparse parallelization regardless of storage for the outer loop."),
             clEnumValN(mlir::SparseParallelizationStrategy::kDenseAnyLoop,
                        "dense-any-loop",
                        "Enable dense parallelization for any loop."),
             clEnumValN(mlir::SparseParallelizationStrategy::kAnyStorageAnyLoop,
                        "any-storage-any-loop",
                        "Enable sparse parallelization for any storage and loop."))}]>,
  ];
}

#endif // MLIR_DIALECT_KOKKOS_TRANSFORMS_PASSES

//===-- Passes.td - LAPIS transform pass definitions -------*- tablegen -*-===//

#ifndef LAPIS_TRANSFORM_PASSES
#define LAPIS_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def AlgebraicKernelFusionPass
    : Pass<"algebraic-kernel-fusion", "ModuleOp"> {
  let summary = "Fuse and reorder compatible linear algebra kernels";
  let constructor = "mlir::createAlgebraicKernelFusionPass()";
  let dependentDialects = [
    "arith::ArithDialect",
    "func::FuncDialect",
    "linalg::LinalgDialect",
    "tensor::TensorDialect"
  ];
}

def OutlineAlgebraicKernelsPass
    : Pass<"outline-algebraic-kernels", "ModuleOp"> {
  let summary = "Outline selected algebraic contraction regions";
  let constructor = "mlir::createOutlineAlgebraicKernelsPass()";
  let dependentDialects = [
    "func::FuncDialect",
    "linalg::LinalgDialect",
    "memref::MemRefDialect"
  ];
}

def FuseAlgebraicKernelLoopsPass
    : Pass<"fuse-algebraic-kernel-loops", "ModuleOp"> {
  let summary = "Fuse compatible parallel domains in algebraic kernels";
  let constructor = "mlir::createFuseAlgebraicKernelLoopsPass()";
  let dependentDialects = [
    "func::FuncDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect"
  ];
}

#endif // LAPIS_TRANSFORM_PASSES

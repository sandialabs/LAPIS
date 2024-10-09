//===-- Passes.td - Part tensor pass definition file -----*- tablegen -*-===//

#ifndef MLIR_DIALECT_PARTTENSOR_TRANSFORMS_PASSES
#define MLIR_DIALECT_PARTTENSOR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def PartTensorConversionPass : Pass<"part-tensor-conversion", "ModuleOp"> {
  let summary = "Convert part tensors and primitives to library calls";
  let description = [{
    A pass that converts part tensor primitives into calls into a runtime
    support library. Part tensor types are converted into opaque pointers
    to the underlying part storage schemes.
  }];
  let constructor = "mlir::createPartTensorConversionPass()";
  let dependentDialects = [
    "arith::ArithDialect",
    "bufferization::BufferizationDialect",
    "LLVM::LLVMDialect",
    "linalg::LinalgDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect",
    "part_tensor::PartTensorDialect",
  ];
  let options = [
    Option<"partTensorDistBackend", "pt-backend", "mlir::PartTensorDistBackend",
           "mlir::PartTensorDistBackend::kNone",
           "Backend to use for part tensor communication",
            [{llvm::cl::values(
                clEnumValN(mlir::PartTensorDistBackend::kNone,
                         "none", "Turn off part tensor distribution."),
                clEnumValN(mlir::PartTensorDistBackend::kMPI, "mpi",
                        "Use Mpi."),
                clEnumValN(mlir::PartTensorDistBackend::kKRS, "krs",
                         "Use Kokkos Remote Spaces.")
             )}]>
  ];
}

#endif // MLIR_DIALECT_PARTTENSOR_TRANSFORMS_PASSES

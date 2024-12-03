#include "lapis-c/EmitKokkos.h"
#include "lapis/Target/KokkosCpp/KokkosCppEmitter.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>

#include "lapis/Dialect/Kokkos/IR/KokkosDialect.h"
#include "lapis/Dialect/Kokkos/Pipelines/Passes.h"
#include "lapis/Dialect/Kokkos/Transforms/Passes.h"
#ifdef ENABLE_PART_TENSOR
#include "lapis/Dialect/PartTensor/IR/PartTensor.h"
#include "lapis/Dialect/PartTensor/Pipelines/Passes.h"
#include "lapis/Dialect/PartTensor/Transforms/Passes.h"
#endif
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/AllInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/RuntimeOpVerification.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/MLProgram/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/Pipelines/Passes.h"
#include "mlir/Dialect/SparseTensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/SubsetInsertionOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"

using namespace mlir;

// Create a fresh MLIR context with dialects required by LAPIS for end-to-end
// lowering
static MLIRContext getLAPISContext() {
  DialectRegistry registry;
  registry.insert<
#ifdef ENABLE_PART_TENSOR
      mlir::part_tensor::PartTensorDialect,
#endif
      mlir::LLVM::LLVMDialect, mlir::vector::VectorDialect,
      mlir::bufferization::BufferizationDialect, mlir::linalg::LinalgDialect,
      mlir::sparse_tensor::SparseTensorDialect, mlir::tensor::TensorDialect,
      mlir::arith::ArithDialect, mlir::scf::SCFDialect,
      mlir::memref::MemRefDialect, mlir::func::FuncDialect,
      mlir::ml_program::MLProgramDialect, mlir::kokkos::KokkosDialect>();

  // Have to also register dialect extensions.
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  arith::registerValueBoundsOpInterfaceExternalModels(registry);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  builtin::registerCastOpInterfaceExternalModels(registry);
  linalg::registerAllDialectInterfaceImplementations(registry);
  linalg::registerRuntimeVerifiableOpInterfaceExternalModels(registry);
  ml_program::registerBufferizableOpInterfaceExternalModels(registry);
  scf::registerBufferDeallocationOpInterfaceExternalModels(registry);
  scf::registerBufferizableOpInterfaceExternalModels(registry);
  sparse_tensor::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerFindPayloadReplacementOpInterfaceExternalModels(registry);
  tensor::registerInferTypeOpInterfaceExternalModels(registry);
  tensor::registerSubsetOpInterfaceExternalModels(registry);
  tensor::registerTilingInterfaceExternalModels(registry);
  tensor::registerValueBoundsOpInterfaceExternalModels(registry);
  vector::registerBufferizableOpInterfaceExternalModels(registry);
  return MLIRContext(registry, MLIRContext::Threading::DISABLED);
}

// Given the source code (ASCII text) for a linalg-level MLIR module,
// lower to Kokkos dialect and emit Kokkos source code.
// cxxSourceFile: path to C++ file to emit
// pySourceFIle: path to python file for ctypes wrapper
MlirLogicalResult lapisLowerAndEmitKokkos(const char *moduleText,
                                          const char *cxxSourceFile,
                                          const char *pySourceFile,
                                          bool isLastKernel) {
  // Parse the high-level module
  MLIRContext context = getLAPISContext();
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(moduleText, &context);
  if (!module) {
    std::cerr << "Failed to parse module\n";
    return wrap(failure());
  }
  // Lower the module
  PassManager pm(&context);
  kokkos::LapisCompilerOptions options;
  kokkos::buildSparseKokkosCompiler(pm, options);
  if (failed(pm.run(*module))) {
    std::cerr << "Failed to lower module\n";
    return wrap(failure());
  }
  std::error_code ec;
  llvm::raw_fd_ostream cxxFileHandle(StringRef(cxxSourceFile), ec);
  llvm::raw_fd_ostream pyFileHandle(StringRef(pySourceFile), ec);
  LogicalResult result = kokkos::translateToKokkosCpp(
      *module, cxxFileHandle, pyFileHandle, /* enableSparseSupport */ true,
      /* useHierarchical */ true, isLastKernel);
  pyFileHandle.close();
  cxxFileHandle.close();
  return wrap(result);
}

MlirLogicalResult lapisEmitKokkos(MlirModule module, const char *cxxSourceFile,
                                  const char *pySourceFile, bool isLastKernel) {
  ModuleOp op = unwrap(module);
  std::error_code ec;
  llvm::raw_fd_ostream cxxFileHandle(StringRef(cxxSourceFile), ec);
  llvm::raw_fd_ostream pyFileHandle(StringRef(pySourceFile), ec);
  LogicalResult result = kokkos::translateToKokkosCpp(
      op, cxxFileHandle, pyFileHandle, /* enableSparseSupport */ true,
      /* useHierarchical */ true, isLastKernel);
  pyFileHandle.close();
  cxxFileHandle.close();
  return wrap(result);
}

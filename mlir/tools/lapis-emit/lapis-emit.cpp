//===- lapis-emit.cpp MLIR->Kokkos end-to-end driver -------------------------===//

#include <iostream>

// Dialects, passes and extensions that must be registered
#include "lapis/InitAllKokkosTranslations.h"
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
#include "mlir/Dialect/Math/IR/Math.h"
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

// Kokkos emitter
#include "lapis/Target/KokkosCpp/KokkosCppEmitter.h"

// MLIR utilities
#include "mlir/Support/FileUtilities.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

// LLVM utilities
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace llvm;

// Default intput is stdin
cl::opt<std::string> inputFilename("i", cl::desc("MLIR input file (linalg on tensors)"), cl::init("-"));
cl::opt<std::string> cxxFilename("cxx", cl::desc("Specify filename for C++ source code output"));
cl::opt<std::string> pyFilename("py", cl::desc("Specify filename for Python wrapper module output"));
cl::opt<bool> finalModule("final", cl::desc("Whether this module should finalize Kokkos"));
cl::opt<bool> dump("dump", cl::desc("Whether to dump the lowered MLIR to file"), cl::init(false));
cl::opt<bool> skipLowering("skipLowering", cl::desc("Whether to skip lowering passes (because input is already lowered)"), cl::init(false));

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv);
  //Register everything
  DialectRegistry registry;
  registry.insert<
#ifdef ENABLE_PART_TENSOR
      mlir::part_tensor::PartTensorDialect,
#endif
      mlir::LLVM::LLVMDialect, mlir::vector::VectorDialect,
      mlir::bufferization::BufferizationDialect, mlir::linalg::LinalgDialect,
      mlir::sparse_tensor::SparseTensorDialect, mlir::tensor::TensorDialect,
      mlir::math::MathDialect, mlir::arith::ArithDialect, mlir::scf::SCFDialect,
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
  MLIRContext context(registry, MLIRContext::Threading::DISABLED);

  std::error_code ec;

  // Open input file, or read from stdin
  std::string errorMessage;
  std::unique_ptr<MemoryBuffer> inputFile = openInputFile(inputFilename, &errorMessage);
  if (!inputFile) {
    std::cerr << "Unable to read MLIR input: " << errorMessage << '\n';
    return 1;
  }
  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(inputFile), SMLoc());
  SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);

  // Parse the input file.
  OwningOpRef<ModuleOp> module(parseSourceFile<ModuleOp>(sourceMgr, &context));
  if (!module) {
    return 1;
  }

  if(!skipLowering) {
    if(dump) {
      std::string dumpPath = cxxFilename + ".hi.mlir";
      std::cout << "Dumping high-level MLIR to \"" << dumpPath << "\"\n";
      llvm::raw_fd_ostream mlirDump(StringRef(dumpPath), ec);
      if(ec) {
        std::cerr << "Failed to open MLIR dump file\n";
        return 1;
      }
      module->print(mlirDump);
      mlirDump.close();
    }

    PassManager pm(&context);
    kokkos::LapisCompilerOptions options;
    kokkos::buildSparseKokkosCompiler(pm, options);
    if (failed(pm.run(*module))) {
      std::cerr << "Failed to lower module\n";
      return 1;
    }
    if(dump) {
      std::string dumpPath = cxxFilename + ".lo.mlir";
      std::cout << "Dumping lowered MLIR to \"" << dumpPath << "\"\n";
      llvm::raw_fd_ostream mlirDump(StringRef(dumpPath), ec);
      if(ec) {
        std::cerr << "Failed to open MLIR dump file\n";
        return 1;
      }
      module->print(mlirDump);
      mlirDump.close();
    }
  }

  llvm::raw_fd_ostream cxxFileHandle(StringRef(cxxFilename), ec);
  if(ec) {
    std::cerr << "Failed to open C++ output file \"" << cxxFilename << "\"\n";
    return 1;
  }
  llvm::raw_fd_ostream pyFileHandle(StringRef(pyFilename), ec);
  if(ec) {
    std::cerr << "Failed to open Python output file \"" << pyFilename << "\"\n";
    return 1;
  }
  //printf("Dump of MLIR before final emitting.");
  //module->dump();
  if(failed(kokkos::translateToKokkosCpp(*module, cxxFileHandle, pyFileHandle, finalModule)))
  {
    std::cerr << "Failed to emit Kokkos\n";
    return 1;
  }
  pyFileHandle.close();
  cxxFileHandle.close();
  return 0;
}


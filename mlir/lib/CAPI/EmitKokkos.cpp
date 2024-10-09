#include "lapis-c/EmitKokkos.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "lapis/Target/KokkosCpp/KokkosCppEmitter.h"

using namespace mlir;

MlirLogicalResult lapisEmitKokkos(MlirModule module, const char* cxxSourceFile,
                                  const char* pySourceFile) {
  ModuleOp op = unwrap(module);
  std::error_code ec;
  llvm::raw_fd_ostream cxxFileHandle(StringRef(cxxSourceFile), ec);
  llvm::raw_fd_ostream pyFileHandle(StringRef(pySourceFile), ec);
  LogicalResult result = kokkos::translateToKokkosCpp(
      op, cxxFileHandle, pyFileHandle, /* enableSparseSupport */ false);
  pyFileHandle.close();
  cxxFileHandle.close();
  return wrap(result);
}

MlirLogicalResult lapisEmitKokkosSparse(MlirModule module,
                                        const char* cxxSourceFile,
                                        const char* pySourceFile,
                                        bool useHierarchical,
                                        bool isLastKernel) {
  ModuleOp op = unwrap(module);
  std::error_code ec;
  llvm::raw_fd_ostream cxxFileHandle(StringRef(cxxSourceFile), ec);
  llvm::raw_fd_ostream pyFileHandle(StringRef(pySourceFile), ec);
  LogicalResult result = kokkos::translateToKokkosCpp(
      op, cxxFileHandle, pyFileHandle, /* enableSparseSupport */ true,
      useHierarchical, isLastKernel);
  pyFileHandle.close();
  cxxFileHandle.close();
  return wrap(result);
}

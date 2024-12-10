//===- TranslateRegistration.cpp - Register translation -------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "lapis/Dialect/Kokkos/IR/KokkosDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "lapis/Target/KokkosCpp/KokkosCppEmitter.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;

namespace mlir {

//===----------------------------------------------------------------------===//
// KokkosCpp registration
//===----------------------------------------------------------------------===//

void registerToKokkosTranslation() {
  TranslateFromMLIRRegistration reg1(
      "mlir-to-kokkos", "translate from mlir to Kokkos",
      [](Operation *op, raw_ostream &output) {
        return kokkos::translateToKokkosCpp(op, output);
      },
      [](DialectRegistry &registry) {
        // clang-format off
        registry.insert<arith::ArithDialect,
                        cf::ControlFlowDialect,
                        emitc::EmitCDialect,
                        func::FuncDialect,
                        kokkos::KokkosDialect,
                        LLVM::LLVMDialect,
                        math::MathDialect,
                        memref::MemRefDialect,
                        scf::SCFDialect>();
        // clang-format on
      });
}

} // namespace mlir

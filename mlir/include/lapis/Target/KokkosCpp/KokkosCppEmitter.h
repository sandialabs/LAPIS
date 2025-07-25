//===- KokkosCppEmitter.h - Helpers to create Kokkos emitter -------------*- C++ -*-===//

#ifndef MLIR_TARGET_KOKKOSCPP_KOKKOSCPPEMITTER_H
#define MLIR_TARGET_KOKKOSCPP_KOKKOSCPPEMITTER_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"
#include <stack>

namespace mlir {
namespace kokkos {

/// Translates the given operation to Kokkos C++ code.
LogicalResult translateToKokkosCpp(Operation *op, raw_ostream* os, raw_ostream* header_os, llvm::StringRef header_path);

/// Translates the given operation to Kokkos C++ code, with a Python wrapper module written to py_os.
LogicalResult translateToKokkosCpp(Operation *op, raw_ostream* os, raw_ostream* header_os, llvm::StringRef header_path, raw_ostream* py_os, bool isLastKernel = true);

/// Translates the given operation to Kokkos team-level functions.
/// This requires that the module was lowered through the team-compiler-kokkos pipeline.
LogicalResult translateToKokkosCppTeamLevel(Operation *op, raw_ostream* os, raw_ostream* header_os, llvm::StringRef header_path);

} // namespace kokkos
} // namespace mlir

#endif // MLIR_TARGET_KOKKOSCPP_KOKKOSCPPEMITTER_H

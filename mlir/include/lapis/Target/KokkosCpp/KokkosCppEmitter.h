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
LogicalResult translateToKokkosCpp(Operation *op, raw_ostream &os);

/// Translates the given operation to Kokkos C++ code, with a Python wrapper module written to py_os.
LogicalResult translateToKokkosCpp(Operation *op, raw_ostream &os, raw_ostream &py_os, bool isLastKernel = true);

} // namespace kokkos
} // namespace mlir

#endif // MLIR_TARGET_KOKKOSCPP_KOKKOSCPPEMITTER_H

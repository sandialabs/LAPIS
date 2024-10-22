#ifndef MLIR_DIALECT_KOKKOS_DIALECT_H
#define MLIR_DIALECT_KOKKOS_DIALECT_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <optional>

#include "lapis/Dialect/Kokkos/IR/KokkosDialect.h.inc"
#include "lapis/Dialect/Kokkos/IR/KokkosEnums.h.inc"

#define GET_OP_CLASSES
#include "lapis/Dialect/Kokkos/IR/Kokkos.h.inc"

namespace mlir {
namespace kokkos {

// Given a CallOp, find the FuncOp corresponding to the callee.
// Since CallOp can only do direct calls, this should always succeed.
func::FuncOp getCalledFunction(func::CallOp callOp);

// Get the top-level "parent" memref of v.
// If v is a block argument or result of an allocation,
// it is its own parent.
// But if it's the result of a view-like op
// (casting, slicing, reshaping) then the memref operand
// of that op is the parent of v.
Value getParentMemref(Value v);

// Determine the correct memory space (Host, Device or DualView)
// for v based on where it gets accessed.
MemorySpace getMemSpace(Value v);

// Get the parallel nesting depth of the given Op
// - If Op itself is a kokkos.parallel or scf.parallel, then that counts as 1
// - Otherwise, Op counts for 0
// - Each enclosing parallel counts for 1 more
int getOpParallelDepth(Operation *op);

// Determine which execution space (Host or Device) executes the given op.
// Note that op may contain parallel kernels that execute on device,
// but in that case op itself still counts as Host.
kokkos::ExecutionSpace getOpExecutionSpace(Operation *op);

// Get a list of the memrefs whose data is read by op, while running on the
// provided exec space. This does not include memrefs where op only uses
// metadata (shape, type, layout).
DenseSet<Value> getMemrefsRead(Operation *op, kokkos::ExecutionSpace space);

// Get a list of the memrefs (possibly) whose data is written to by op.
DenseSet<Value> getMemrefsWritten(Operation *op, kokkos::ExecutionSpace space);

bool valueIsIntegerConstantZero(Value v);
bool valueIsIntegerConstantOne(Value v);

} // namespace kokkos
} // namespace mlir

#endif // MLIR_DIALECT_KOKKOS_DIALECT_H

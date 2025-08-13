#ifndef MLIR_DIALECT_KOKKOS_DIALECT_H
#define MLIR_DIALECT_KOKKOS_DIALECT_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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

// Is op a memref/view alias?
// For example, shallow-copy, subview and cast ops
bool isViewAliasingOp(Operation* op);

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

// Return the function that has v as a parameter, if it is a parameter.
// Otherwise return null.
func::FuncOp getFuncWithParameter(Value v);

// Does this function have a body/definition?
// (i.e. it's not just a declaration for an extern function)
bool funcHasBody(func::FuncOp op);

// Determine the correct memory space (Host, Device or DualView)
// for v based on where it gets accessed, and whether we are generating
// team-level code.
//
// Arguments and results are always DualView if !teamLevel,
// and Device if teamLevel.
MemorySpace getMemSpace(Value v, bool teamLevel);

// Determine the correct memory space (Host, Device or DualView)
// for the global view.
//
// If teamLevel, then all code is assumed to be device code so globals are Device space.
MemorySpace getMemSpace(memref::GlobalOp global, bool teamLevel);

// Is the global memref used by at least one op?
bool isGlobalUsed(memref::GlobalOp global);

// Get the parallel nesting depth of the given Op
// - If Op itself is a kokkos.parallel or scf.parallel, then that counts as 1
// - Otherwise, Op counts for 0
// - Each enclosing parallel counts for 1 more
int getOpParallelDepth(Operation *op);

// Determine which execution space (Host or Device) executes the given op.
// Note that op may contain parallel kernels that execute on device,
// but in that case op itself still counts as Host.
kokkos::ExecutionSpace getOpExecutionSpace(Operation *op);

// Get a list of the memrefs whose data may be read by op, while running on the
// provided exec space. This does not include memrefs whose metadata (shape, type, layout
// is used but data is not.
DenseSet<Value> getMemrefsRead(Operation *op, kokkos::ExecutionSpace space);

// Get a list of the memrefs that may be written to by op.
DenseSet<Value> getMemrefsWritten(Operation *op, kokkos::ExecutionSpace space);

bool valueIsIntegerConstantZero(Value v);
bool valueIsIntegerConstantOne(Value v);

// If struct type contains only one type (as either single members or LLVM arrays)
// then return that type.
// Otherwise, return null.
// Also returns null if st has no members.
Type getStructElementType(LLVM::LLVMStructType st);

// If the above function returns true, count the total number of elements in the struct:
// sizeof(structType) / sizeof(elemType)
int getStructElementCount(LLVM::LLVMStructType st);

// Get the size in bytes to represent a built-in (primitive) type t.
// This includes integers and floats but also (TODO: check) structs and complex types.
// Returns 0 if a size is variable or otherwise not known.
size_t getBuiltinTypeSize(Type t, Operation* op);

// Returns true if the non-padded size in bytes for this memref
// type is known statically. Will return false if mrt has dynamic dimensions, or uses
// platform-specific vector types.
// 
// If true, also sets size to this size.
bool memrefSizeInBytesKnown(MemRefType mrt, size_t& size, Operation* op);

} // namespace kokkos
} // namespace mlir

#endif // MLIR_DIALECT_KOKKOS_DIALECT_H

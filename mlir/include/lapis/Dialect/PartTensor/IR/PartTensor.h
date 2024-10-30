#ifndef MLIR_DIALECT_PARTITION_IR_PARTITION_H
#define MLIR_DIALECT_PARTITION_IR_PARTITION_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
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

namespace mlir {
namespace arith {
enum class AtomicRMWKind : uint64_t;
class AtomicRMWKindAttr;
} // namespace arith
} // namespace mlir

#define GET_ATTRDEF_CLASSES
#include "lapis/Dialect/PartTensor/IR/PartTensorAttrDefs.h.inc"

#define GET_OP_CLASSES
#include "lapis/Dialect/PartTensor/IR/PartTensorOps.h.inc"

#include "lapis/Dialect/PartTensor/IR/PartTensorOpsDialect.h.inc"

namespace mlir {
namespace part_tensor {
/// Convenience method to get a sparse encoding attribute from a type.
/// Returns null-attribute for any type without an encoding.
PartTensorEncodingAttr getPartTensorEncoding(Type type);
} // namespace part_tensor
} // namespace mlir

#endif // MLIR_DIALECT_PARTITION_IR_PARTITION_H

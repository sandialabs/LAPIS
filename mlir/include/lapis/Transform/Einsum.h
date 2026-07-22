#ifndef LAPIS_TRANSFORM_EINSUM_H
#define LAPIS_TRANSFORM_EINSUM_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

namespace mlir::lapis {

using IndexId = unsigned;

struct TensorAccess {
  llvm::SmallVector<IndexId, 4> indices;
};

struct IndexExtent {
  IndexId index;
  int64_t extent;
};

/// A platform-independent sum-product einsum expression.
///
/// The result indices are the free indices; every other operand index is a
/// reduction index. Extents must be positive, static, and exactly cover the
/// indices used by the operands. Index order is deterministic and follows the
/// first occurrence in the operand list.
///
/// This class validates structural einsum invariants only. Repeated input
/// indices and unary reductions are valid expressions even if a particular
/// planner or MLIR lowering does not support them yet.
class EinsumExpression {
public:
  static llvm::Expected<EinsumExpression>
  create(llvm::ArrayRef<TensorAccess> operands, TensorAccess result,
         llvm::ArrayRef<IndexExtent> extents);

  llvm::ArrayRef<TensorAccess> getOperands() const { return operands; }
  const TensorAccess &getResult() const { return result; }
  llvm::ArrayRef<IndexId> getIndexOrder() const { return indexOrder; }

  std::optional<int64_t> getExtent(IndexId index) const;
  llvm::SmallVector<IndexId, 4> getReductionIndices() const;

private:
  EinsumExpression(llvm::SmallVector<TensorAccess, 4> operands,
                   TensorAccess result,
                   llvm::DenseMap<IndexId, int64_t> extents,
                   llvm::SmallVector<IndexId, 4> indexOrder);

  llvm::SmallVector<TensorAccess, 4> operands;
  TensorAccess result;
  llvm::DenseMap<IndexId, int64_t> extents;
  llvm::SmallVector<IndexId, 4> indexOrder;
};

/// The composed expression together with the producer-to-composed index map.
///
/// The map carries algebraic metadata associated with values inside the
/// producer into the composed expression's index namespace.
class EinsumComposition {
public:
  const EinsumExpression &getExpression() const { return expression; }
  EinsumExpression takeExpression() { return std::move(expression); }

  TensorAccess remapProducerAccess(const TensorAccess &access) const;

private:
  EinsumComposition(EinsumExpression expression,
                    llvm::DenseMap<IndexId, IndexId> producerIndexMap);

  EinsumExpression expression;
  llvm::DenseMap<IndexId, IndexId> producerIndexMap;

  friend llvm::Expected<EinsumComposition>
  composeEinsumExpressionsWithProvenance(const EinsumExpression &producer,
                                         const EinsumExpression &consumer,
                                         std::size_t consumerOperand);
};

/// Substitutes the producer result for one consumer operand.
///
/// Producer result indices are matched positionally with the selected consumer
/// operand. Producer reduction indices are alpha-renamed when they collide with
/// consumer indices, and the producer operands replace the selected consumer
/// operand in the returned expression.
llvm::Expected<EinsumExpression>
composeEinsumExpressions(const EinsumExpression &producer,
                         const EinsumExpression &consumer,
                         std::size_t consumerOperand);

/// Composes two expressions and retains the producer index renaming.
llvm::Expected<EinsumComposition>
composeEinsumExpressionsWithProvenance(const EinsumExpression &producer,
                                       const EinsumExpression &consumer,
                                       std::size_t consumerOperand);

/// An algebraic expression together with the MLIR values from which it was
/// extracted.
///
/// `operands` and the operands of `expression` have the same order. `init` is
/// the proven-zero output initializer, and `result` is the single tensor result
/// produced by `source`.
class ExtractedEinsumExpression {
public:
  ExtractedEinsumExpression(linalg::GenericOp source,
                            EinsumExpression expression,
                            llvm::ArrayRef<Value> operands, Value init,
                            Value result);

  linalg::GenericOp getSource() const { return source; }
  const EinsumExpression &getExpression() const { return expression; }
  llvm::ArrayRef<Value> getOperands() const { return operands; }
  Value getInit() const { return init; }
  Value getResult() const { return result; }

private:
  linalg::GenericOp source;
  EinsumExpression expression;
  llvm::SmallVector<Value, 4> operands;
  Value init;
  Value result;
};

/// Extracts the supported sum-product subset of tensor-form linalg.generic.
///
/// The current contract requires two static, positive-shape tensor inputs, one
/// tensor result with a provably zero initializer, dimension-only indexing
/// maps, iterator types consistent with Einstein summation, and a body
/// equivalent to:
///
///   product = arith.mulf(input0, input1)
///   sum = arith.addf(accumulator, product)
///   linalg.yield sum
///
/// Operand order and result-index order follow the linalg.generic indexing
/// maps. Unsupported operations produce an Error and leave the IR unchanged.
///
/// TODO: Require an explicit reassociation policy before using extraction to
/// authorize transformations that change floating-point evaluation order.
llvm::Expected<ExtractedEinsumExpression>
extractEinsumExpression(linalg::GenericOp generic);

} // namespace mlir::lapis

#endif // LAPIS_TRANSFORM_EINSUM_H

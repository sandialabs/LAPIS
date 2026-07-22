#ifndef LAPIS_TRANSFORM_CONTRACTIONPLANNER_H
#define LAPIS_TRANSFORM_CONTRACTIONPLANNER_H

#include "lapis/Transform/Einsum.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace mlir::lapis {

using PlanValueId = unsigned;
using OperandSubset = uint64_t;

inline constexpr unsigned kMaxTrackedOperands =
    std::numeric_limits<OperandSubset>::digits;

struct ContractionStep {
  PlanValueId lhs;
  PlanValueId rhs;
  llvm::SmallVector<IndexId, 4> resultIndices;
};

/// An immutable, topologically ordered sequence of binary contractions.
///
/// Values [0, operandCount) identify original expression operands. Step N
/// produces value operandCount + N. The verifier, rather than this container,
/// establishes that the sequence is a complete and legal contraction tree.
class ContractionPlan {
public:
  ContractionPlan(unsigned operandCount, llvm::ArrayRef<ContractionStep> steps,
                  PlanValueId result);

  unsigned getOperandCount() const { return operandCount; }
  llvm::ArrayRef<ContractionStep> getSteps() const { return steps; }
  PlanValueId getResult() const { return result; }

  PlanValueId getStepResult(std::size_t step) const {
    return operandCount + step;
  }

private:
  unsigned operandCount;
  llvm::SmallVector<ContractionStep, 4> steps;
  PlanValueId result;
};

/// Algebraic index-boundary analysis for subsets of expression operands.
class IndexLiveness {
public:
  static llvm::Expected<IndexLiveness>
  create(const EinsumExpression &expression);

  unsigned getOperandCount() const { return operandCount; }
  OperandSubset getAllOperands() const { return allOperands; }
  OperandSubset getUsers(IndexId index) const;

  /// Returns the canonical result indices for materializing `operands`.
  ///
  /// Final-result indices appear in final-result order. Other indices survive
  /// only when used by an operand outside the subset and appear in the
  /// expression's first-occurrence order.
  llvm::Expected<llvm::SmallVector<IndexId, 4>>
  getLiveIndices(OperandSubset operands) const;

private:
  IndexLiveness(unsigned operandCount, OperandSubset allOperands,
                llvm::DenseMap<IndexId, OperandSubset> indexUsers,
                llvm::SmallVector<IndexId, 4> indexOrder,
                llvm::SmallVector<IndexId, 4> resultOrder,
                llvm::DenseSet<IndexId> resultIndices);

  unsigned operandCount;
  OperandSubset allOperands;
  llvm::DenseMap<IndexId, OperandSubset> indexUsers;
  llvm::SmallVector<IndexId, 4> indexOrder;
  llvm::SmallVector<IndexId, 4> resultOrder;
  llvm::DenseSet<IndexId> resultIndices;
};

/// Reconstructs the original-operand subset represented by every plan value.
/// The returned vector is indexed by PlanValueId.
llvm::Expected<llvm::SmallVector<OperandSubset, 8>>
computePlanOperandSubsets(const ContractionPlan &plan);

/// Finds the plan value that represents exactly `operands`.
llvm::Expected<PlanValueId>
findPlanValueForOperandSubset(const ContractionPlan &plan,
                              OperandSubset operands);

/// Independently verifies plan structure, operand coverage, and index
/// liveness against an expression.
llvm::Error verifyContractionPlan(const EinsumExpression &expression,
                                  const ContractionPlan &plan);

/// A platform-independent count of scalar contraction work.
using WorkCost = uint64_t;

struct ContractionPlanCost {
  llvm::SmallVector<WorkCost, 4> stepCosts;
  WorkCost totalWork = 0;
};

/// Computes algebraic work from operand subsets and index extents.
///
/// One work unit is one point in a binary contraction's iteration space,
/// equivalently one scalar multiplication. This deliberately excludes
/// backend-dependent execution and storage characteristics.
class ContractionCostModel {
public:
  static llvm::Expected<ContractionCostModel>
  create(const EinsumExpression &expression);

  const EinsumExpression &getExpression() const { return expression; }
  const IndexLiveness &getIndexLiveness() const { return liveness; }

  llvm::Expected<WorkCost> getContractionWork(OperandSubset lhs,
                                              OperandSubset rhs) const;

  llvm::Expected<ContractionPlanCost>
  evaluatePlan(const ContractionPlan &plan) const;

private:
  ContractionCostModel(EinsumExpression expression, IndexLiveness liveness);

  llvm::Expected<llvm::SmallVector<IndexId, 4>>
  getValueIndices(OperandSubset operands) const;

  EinsumExpression expression;
  IndexLiveness liveness;
};

/// Deliberate guardrail for the exponential exact planner.
///
/// OperandSubset can represent more operands, but the planner requires O(3^N)
/// time and O(2^N) memory. Callers must choose a different planning strategy
/// rather than accidentally invoking exact planning above this limit.
inline constexpr unsigned kMaxExactPlannerOperands = 16;

/// An intermediate value that must remain available after reassociation.
struct ContractionResultConstraint {
  OperandSubset operands;
  TensorAccess access;
};

/// Finds a minimum-work binary contraction tree by subset dynamic programming.
///
/// The objective is ContractionCostModel::totalWork. Equal-cost alternatives
/// are resolved deterministically by lexicographic operand-subset order.
class ExactContractionPlanner {
public:
  static llvm::Expected<ExactContractionPlanner>
  create(const EinsumExpression &expression);

  /// Plans a contraction tree containing every requested result.
  ///
  /// Required subsets model externally visible intermediate results. They
  /// must be pairwise nested or disjoint because a binary contraction tree
  /// cannot represent partially overlapping intermediates without duplicating
  /// work.
  llvm::Expected<ContractionPlan>
  plan(llvm::ArrayRef<ContractionResultConstraint> requiredResults = {}) const;

private:
  explicit ExactContractionPlanner(ContractionCostModel costModel);

  ContractionCostModel costModel;
};

} // namespace mlir::lapis

#endif // LAPIS_TRANSFORM_CONTRACTIONPLANNER_H

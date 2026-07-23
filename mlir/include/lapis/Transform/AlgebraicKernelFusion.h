//===- AlgebraicKernelFusion.h - Algebraic kernel fusion -------*- C++ -*-===//

#ifndef LAPIS_TRANSFORM_ALGEBRAICKERNELFUSION_H
#define LAPIS_TRANSFORM_ALGEBRAICKERNELFUSION_H

#include "lapis/Transform/ContractionPlanner.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <optional>

namespace mlir {

#define GEN_PASS_DECL
#include "lapis/Transform/Passes.h.inc"

std::unique_ptr<Pass> createAlgebraicKernelFusionPass();

/// Outlines each marked, bufferized contraction group into one private
/// function. Marked contractions must have pure buffer semantics, be in one
/// block, and be movable to the final contraction without crossing an
/// interfering side effect. Allocation-backed intermediates used only for
/// communication inside one group are moved into the outlined function along
/// with their initialization and optional deallocation.
std::unique_ptr<Pass> createOutlineAlgebraicKernelsPass();

/// Fuses adjacent top-level scf.parallel operations in outlined algebraic
/// kernels when they have an equal, non-empty leading domain and every
/// cross-loop memory dependence is partitioned by that domain. Selected
/// Linalg groups align corresponding parallel iterators before loop lowering,
/// without moving or reordering reduction iterators, so equivalent permuted
/// domains can satisfy this requirement.
/// Allocation-backed intermediates represented by a single dominating store
/// and same-index loads in the resulting loop are forwarded as SSA scalars and
/// removed.
std::unique_ptr<Pass> createFuseAlgebraicKernelLoopsPass();

#define GEN_PASS_REGISTRATION
#include "lapis/Transform/Passes.h.inc"

namespace lapis {

/// Internal handoff attributes between algebraic selection,
/// post-bufferization outlining, and loop fusion. They deliberately describe a
/// proven candidate; they are not a user-facing request to fuse arbitrary
/// operations.
inline constexpr llvm::StringLiteral kAlgebraicFusionGroupAttr =
    "lapis.algebraic_fusion_group";
inline constexpr llvm::StringLiteral kAlgebraicKernelAttr =
    "lapis.algebraic_kernel";

/// An externally visible tensor produced by an algebraic region.
struct LinalgEinsumRegionOutput {
  linalg::GenericOp source;
  Value value;
  OperandSubset operandSubset;
  TensorAccess access;
};

/// A maximal tree of composable linalg.generic operations.
///
/// Operations are stored in topological order with the root last. `operands`
/// contains the external tensor values corresponding positionally to the
/// operands of `expression`. Internal producer results do not appear there.
/// `outputs` contains the root result and any internal results used outside the
/// region. Their operand subsets constrain materialization to preserve those
/// values without recomputation.
class LinalgEinsumRegion {
public:
  llvm::ArrayRef<linalg::GenericOp> getOperations() const { return operations; }
  linalg::GenericOp getRoot() const { return operations.back(); }
  const EinsumExpression &getExpression() const { return expression; }
  llvm::ArrayRef<Value> getOperands() const { return operands; }
  llvm::ArrayRef<LinalgEinsumRegionOutput> getOutputs() const {
    return outputs;
  }
  Value getResult() const { return outputs.back().value; }

private:
  LinalgEinsumRegion(llvm::ArrayRef<linalg::GenericOp> operations,
                     EinsumExpression expression,
                     llvm::ArrayRef<Value> operands,
                     llvm::ArrayRef<LinalgEinsumRegionOutput> outputs);

  llvm::SmallVector<linalg::GenericOp, 4> operations;
  EinsumExpression expression;
  llvm::SmallVector<Value, 4> operands;
  llvm::SmallVector<LinalgEinsumRegionOutput, 2> outputs;

  friend llvm::Expected<llvm::SmallVector<LinalgEinsumRegion, 4>>
  discoverLinalgEinsumRegions(func::FuncOp function);
};

/// Discovers maximal algebraic producer-consumer regions in `function`.
///
/// A producer is internalized only when both operations satisfy
/// extractEinsumExpression, occur in the same block, and the producer result
/// has exactly one composable consumer use. Other uses become preserved region
/// outputs. Multiple composable uses, unsupported operations, and block
/// boundaries terminate a region. Nested control-flow blocks are considered
/// independently. These conservative rules form disjoint trees and avoid
/// duplicating computation. Singleton regions are omitted because they contain
/// no removable producer-consumer boundary.
///
/// This discovers legal algebraic candidates only. It does not decide that a
/// region is profitable or authorize a rewrite.
llvm::Expected<llvm::SmallVector<LinalgEinsumRegion, 4>>
discoverLinalgEinsumRegions(func::FuncOp function);

/// Platform-independent comparison for an algebraic fusion candidate.
struct FusionCandidateEvaluation {
  WorkCost originalWork;
  WorkCost optimizedWork;
  unsigned originalKernelCount;
  unsigned fusedKernelCount;

  /// Multiplication work is the primary objective. When work is equal, prefer
  /// placing the region in fewer algebraic kernels. No backend characteristics
  /// are part of either objective.
  bool shouldRewrite() const {
    return optimizedWork < originalWork ||
           (optimizedWork == originalWork &&
            fusedKernelCount < originalKernelCount);
  }
};

/// Compares the work performed by the source operations in `region` with the
/// work performed by `optimizedPlan`.
///
/// One work unit is one scalar multiplication. The secondary objective counts
/// algebraic kernel boundaries only. Storage volume, locality, launch cost, and
/// backend characteristics are intentionally excluded.
llvm::Expected<FusionCandidateEvaluation>
evaluateFusionCandidate(const LinalgEinsumRegion &region,
                        const ContractionPlan &optimizedPlan);

/// Replaces `region` with the binary linalg.generic contractions in `plan`.
///
/// Every plan step is materialized as a sum-product contraction. Intermediate
/// results use fresh, statically shaped zero-filled tensors; externally visible
/// results reuse their original proven-zero initializers. The plan and all
/// operand/result types are validated before the IR is modified.
///
/// TODO: Require an explicit reassociation policy before authorizing this
/// floating-point rewrite.
llvm::Expected<llvm::SmallVector<Value, 2>> materializeContractionPlan(
    RewriterBase &rewriter, const LinalgEinsumRegion &region,
    const ContractionPlan &plan,
    std::optional<std::uint64_t> fusionGroup = std::nullopt);

} // namespace lapis
} // namespace mlir

#endif // LAPIS_TRANSFORM_ALGEBRAICKERNELFUSION_H

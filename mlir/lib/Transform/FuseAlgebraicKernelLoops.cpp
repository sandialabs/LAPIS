//===- FuseAlgebraicKernelLoops.cpp --------------------------------------===//

#include "lapis/Transform/AlgebraicKernelFusion.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

//===----------------------------------------------------------------------===//
// Partial parallel-domain fusion
//===----------------------------------------------------------------------===//

namespace mlir::lapis {
namespace {

struct MemoryAccess {
  Value buffer;
  llvm::SmallVector<Value, 4> indices;
  bool writes;
};

bool collectMemoryAccesses(Operation *root,
                           llvm::SmallVectorImpl<MemoryAccess> &accesses) {
  bool supported = true;
  root->walk([&](Operation *operation) {
    if (auto load = dyn_cast<memref::LoadOp>(operation)) {
      accesses.push_back(
          MemoryAccess{load.getMemRef(),
                       {load.getIndices().begin(), load.getIndices().end()},
                       /*writes=*/false});
      return;
    }
    if (auto store = dyn_cast<memref::StoreOp>(operation)) {
      accesses.push_back(
          MemoryAccess{store.getMemRef(),
                       {store.getIndices().begin(), store.getIndices().end()},
                       /*writes=*/true});
      return;
    }
    if (isa<scf::ParallelOp, scf::ReduceOp, scf::ReduceReturnOp>(operation) ||
        isMemoryEffectFree(operation) ||
        operation->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
      return;
    supported = false;
  });
  return supported;
}

bool equivalentBound(Value first, Value second) {
  if (first == second)
    return true;
  std::optional<std::int64_t> firstConstant = getConstantIntValue(first);
  std::optional<std::int64_t> secondConstant = getConstantIntValue(second);
  return firstConstant && secondConstant && *firstConstant == *secondConstant;
}

unsigned getCommonPrefix(scf::ParallelOp first, scf::ParallelOp second) {
  unsigned common = 0;
  unsigned maximum = std::min(first.getNumLoops(), second.getNumLoops());
  while (common < maximum &&
         equivalentBound(first.getLowerBound()[common],
                         second.getLowerBound()[common]) &&
         equivalentBound(first.getUpperBound()[common],
                         second.getUpperBound()[common]) &&
         equivalentBound(first.getStep()[common], second.getStep()[common]))
    ++common;
  return common;
}

bool isPartitionedByCommonPrefix(const MemoryAccess &first,
                                 const MemoryAccess &second,
                                 scf::ParallelOp firstLoop,
                                 scf::ParallelOp secondLoop,
                                 unsigned commonPrefix) {
  if (first.indices.size() != second.indices.size())
    return false;
  for (unsigned dimension = 0; dimension < commonPrefix; ++dimension) {
    bool foundPartitionIndex = false;
    for (const auto &[firstIndex, secondIndex] :
         llvm::zip(first.indices, second.indices)) {
      if (firstIndex == firstLoop.getInductionVars()[dimension] &&
          secondIndex == secondLoop.getInductionVars()[dimension]) {
        foundPartitionIndex = true;
        break;
      }
    }
    if (!foundPartitionIndex)
      return false;
  }
  return true;
}

Value resolveCallSiteBuffer(Value buffer, func::FuncOp kernel,
                            func::CallOp call) {
  auto argument = dyn_cast<BlockArgument>(buffer);
  if (!argument || argument.getOwner() != &kernel.getBody().front())
    return buffer;
  return call.getOperand(argument.getArgNumber());
}

bool isFreshAllocation(Value value) {
  return isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(
      value.getDefiningOp());
}

bool areProvablyDistinctFreshBuffers(Value first, Value second) {
  if (first == second)
    return false;
  bool firstFresh = isFreshAllocation(first);
  bool secondFresh = isFreshAllocation(second);
  return (firstFresh && (secondFresh || isa<BlockArgument>(second))) ||
         (secondFresh && isa<BlockArgument>(first));
}

bool buffersCannotAlias(Value first, Value second,
                        AliasAnalysis &aliasAnalysis) {
  return aliasAnalysis.alias(first, second).isNo() ||
         areProvablyDistinctFreshBuffers(first, second);
}

bool dependenciesPermitFusion(scf::ParallelOp firstLoop,
                              scf::ParallelOp secondLoop, unsigned commonPrefix,
                              func::FuncOp kernel, func::CallOp call,
                              AliasAnalysis &aliasAnalysis) {
  llvm::SmallVector<MemoryAccess, 16> firstAccesses;
  llvm::SmallVector<MemoryAccess, 16> secondAccesses;
  if (!collectMemoryAccesses(firstLoop.getOperation(), firstAccesses) ||
      !collectMemoryAccesses(secondLoop.getOperation(), secondAccesses))
    return false;

  for (const MemoryAccess &first : firstAccesses) {
    for (const MemoryAccess &second : secondAccesses) {
      if (!first.writes && !second.writes)
        continue;
      Value firstBuffer = resolveCallSiteBuffer(first.buffer, kernel, call);
      Value secondBuffer = resolveCallSiteBuffer(second.buffer, kernel, call);
      if (buffersCannotAlias(firstBuffer, secondBuffer, aliasAnalysis))
        continue;
      if (first.buffer != second.buffer ||
          !isPartitionedByCommonPrefix(first, second, firstLoop, secondLoop,
                                       commonPrefix))
        return false;
    }
  }
  return true;
}

bool operationsBetweenAreMovable(scf::ParallelOp first, scf::ParallelOp second,
                                 func::FuncOp kernel, func::CallOp call,
                                 AliasAnalysis &aliasAnalysis) {
  llvm::SmallVector<MemoryAccess, 16> firstAccesses;
  if (!collectMemoryAccesses(first.getOperation(), firstAccesses))
    return false;

  for (Operation *operation = first->getNextNode(); operation != second;
       operation = operation->getNextNode()) {
    llvm::SmallVector<MemoryAccess, 4> interveningAccesses;
    if (!collectMemoryAccesses(operation, interveningAccesses))
      return false;
    for (const MemoryAccess &firstAccess : firstAccesses) {
      for (const MemoryAccess &interveningAccess : interveningAccesses) {
        if (!firstAccess.writes && !interveningAccess.writes)
          continue;
        Value firstBuffer =
            resolveCallSiteBuffer(firstAccess.buffer, kernel, call);
        Value interveningBuffer =
            resolveCallSiteBuffer(interveningAccess.buffer, kernel, call);
        if (!buffersCannotAlias(firstBuffer, interveningBuffer, aliasAnalysis))
          return false;
      }
    }
  }
  return true;
}

void cloneParallelBody(OpBuilder &rewriter, scf::ParallelOp source,
                       ValueRange commonInductionVars, unsigned commonPrefix,
                       bool cloneTerminator = false) {
  IRMapping mapping;
  llvm::SmallVector<Value> sourceInductionVars = source.getInductionVars();
  mapping.map(ValueRange(sourceInductionVars).take_front(commonPrefix),
              commonInductionVars);

  auto cloneOperations = [&](OpBuilder &builder, ValueRange remainingIvs) {
    mapping.map(ValueRange(sourceInductionVars).drop_front(commonPrefix),
                remainingIvs);
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(builder.getInsertionBlock());
    for (Operation &operation : *source.getBody()) {
      if (!cloneTerminator && operation.hasTrait<OpTrait::IsTerminator>())
        continue;
      builder.clone(operation, mapping);
    }
  };

  unsigned remaining = source.getNumLoops() - commonPrefix;
  if (remaining == 0) {
    cloneOperations(rewriter, {});
    return;
  }

  rewriter.create<scf::ParallelOp>(
      source.getLoc(), source.getLowerBound().drop_front(commonPrefix),
      source.getUpperBound().drop_front(commonPrefix),
      source.getStep().drop_front(commonPrefix),
      [&](OpBuilder &builder, Location, ValueRange inductionVars) {
        cloneOperations(builder, inductionVars);
      });
}

bool fuseParallelPair(scf::ParallelOp first, scf::ParallelOp second,
                      func::FuncOp kernel, func::CallOp call,
                      AliasAnalysis &aliasAnalysis) {
  if (!first.getInitVals().empty() || first->getBlock() != second->getBlock() ||
      !operationsBetweenAreMovable(first, second, kernel, call, aliasAnalysis))
    return false;

  unsigned commonPrefix = getCommonPrefix(first, second);
  bool secondReduces = !second.getInitVals().empty();
  // A reduction consumer can be fused when the shared domain is its complete
  // reduction domain. Partial reduction-domain fusion would require carrying
  // reduction values through another nested parallel operation.
  if (secondReduces && commonPrefix != second.getNumLoops())
    return false;
  if (commonPrefix == 0 ||
      !dependenciesPermitFusion(first, second, commonPrefix, kernel, call,
                                aliasAnalysis))
    return false;

  IRRewriter rewriter(first.getContext());
  rewriter.setInsertionPoint(second);
  scf::ParallelOp fused = rewriter.create<scf::ParallelOp>(
      second.getLoc(), second.getLowerBound().take_front(commonPrefix),
      second.getUpperBound().take_front(commonPrefix),
      second.getStep().take_front(commonPrefix), second.getInitVals(),
      [&](OpBuilder &builder, Location, ValueRange inductionVars, ValueRange) {
        cloneParallelBody(builder, first, inductionVars, commonPrefix);
        cloneParallelBody(builder, second, inductionVars, commonPrefix,
                          /*cloneTerminator=*/secondReduces);
      });
  for (auto [oldResult, newResult] :
       llvm::zip(second.getResults(), fused.getResults()))
    oldResult.replaceAllUsesWith(newResult);
  rewriter.eraseOp(first);
  rewriter.eraseOp(second);
  return true;
}

func::CallOp findSoleKernelCall(ModuleOp module, func::FuncOp kernel) {
  func::CallOp result;
  bool multipleCalls = false;
  module.walk([&](func::CallOp call) {
    if (call.getCallee() != kernel.getSymName())
      return;
    if (result)
      multipleCalls = true;
    else
      result = call;
  });
  return multipleCalls ? func::CallOp{} : result;
}

void fuseKernelLoops(func::FuncOp kernel, func::CallOp call,
                     AliasAnalysis &aliasAnalysis) {
  bool changed = true;
  while (changed) {
    changed = false;
    for (Block &block : kernel.getBody()) {
      llvm::SmallVector<scf::ParallelOp, 4> loops =
          llvm::to_vector(block.getOps<scf::ParallelOp>());
      for (auto [first, second] : llvm::zip(loops, llvm::drop_begin(loops))) {
        if (fuseParallelPair(first, second, kernel, call, aliasAnalysis)) {
          changed = true;
          break;
        }
      }
      if (changed)
        break;
    }
  }
}

struct ScalarForwardingCandidate {
  Operation *allocation;
  memref::StoreOp store;
  llvm::SmallVector<memref::LoadOp, 2> loads;
  llvm::SmallVector<memref::DeallocOp, 1> deallocations;
};

/// Recognizes the deliberately narrow case where a local buffer is merely an
/// SSA edge obscured by loop lowering: one store operation, one or more later
/// loads in the same parallel loop, and exactly equal index Values. Buffers
/// with initialization, multiple stores, subviews, or cross-loop reuse remain
/// materialized.
std::optional<ScalarForwardingCandidate>
getScalarForwardingCandidate(Operation *allocation, DominanceInfo &dominance) {
  if (!isa<memref::AllocOp, memref::AllocaOp>(allocation) ||
      allocation->getNumResults() != 1)
    return std::nullopt;

  Value buffer = allocation->getResult(0);
  ScalarForwardingCandidate candidate{allocation, {}, {}, {}};
  for (OpOperand &use : buffer.getUses()) {
    Operation *owner = use.getOwner();
    if (auto candidateStore = dyn_cast<memref::StoreOp>(owner)) {
      if (candidateStore.getMemRef() != buffer || candidate.store)
        return std::nullopt;
      candidate.store = candidateStore;
      continue;
    }
    if (auto load = dyn_cast<memref::LoadOp>(owner)) {
      if (load.getMemRef() != buffer)
        return std::nullopt;
      candidate.loads.push_back(load);
      continue;
    }
    if (auto deallocation = dyn_cast<memref::DeallocOp>(owner)) {
      candidate.deallocations.push_back(deallocation);
      continue;
    }
    return std::nullopt;
  }
  if (!candidate.store || candidate.loads.empty())
    return std::nullopt;

  scf::ParallelOp parallel =
      candidate.store->getParentOfType<scf::ParallelOp>();
  if (!parallel)
    return std::nullopt;
  for (memref::LoadOp load : candidate.loads) {
    if (load->getParentOfType<scf::ParallelOp>() != parallel ||
        !dominance.dominates(candidate.store, load) ||
        !llvm::equal(candidate.store.getIndices(), load.getIndices()))
      return std::nullopt;
  }
  return candidate;
}

void eliminateScalarIntermediates(func::FuncOp kernel) {
  DominanceInfo dominance(kernel);
  llvm::SmallVector<ScalarForwardingCandidate, 4> candidates;
  kernel.walk([&](Operation *operation) {
    auto candidate = getScalarForwardingCandidate(operation, dominance);
    if (candidate)
      candidates.push_back(std::move(*candidate));
  });

  for (ScalarForwardingCandidate &candidate : candidates) {
    Value forwarded = candidate.store.getValue();
    for (memref::LoadOp load : candidate.loads) {
      load.getResult().replaceAllUsesWith(forwarded);
      load.erase();
    }
    candidate.store.erase();
    for (memref::DeallocOp deallocation : candidate.deallocations)
      deallocation.erase();
    candidate.allocation->erase();
  }
}

} // namespace

LogicalResult fuseAlgebraicKernelLoops(ModuleOp module) {
  AliasAnalysis aliasAnalysis(module);
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    if (!function->hasAttr(kAlgebraicKernelAttr))
      continue;
    func::CallOp call = findSoleKernelCall(module, function);
    if (!call) {
      function.emitRemark(
          "loop fusion requires an outlined kernel with exactly one call");
      continue;
    }
    fuseKernelLoops(function, call, aliasAnalysis);
    eliminateScalarIntermediates(function);
  }
  return success();
}

} // namespace mlir::lapis

namespace mlir {
#define GEN_PASS_DEF_FUSEALGEBRAICKERNELLOOPSPASS
#include "lapis/Transform/Passes.h.inc"
} // namespace mlir

namespace {
struct FuseAlgebraicKernelLoopsPass
    : public mlir::impl::FuseAlgebraicKernelLoopsPassBase<
          FuseAlgebraicKernelLoopsPass> {
  using Base::Base;

  void runOnOperation() override {
    if (failed(mlir::lapis::fuseAlgebraicKernelLoops(getOperation())))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> mlir::createFuseAlgebraicKernelLoopsPass() {
  return std::make_unique<FuseAlgebraicKernelLoopsPass>();
}

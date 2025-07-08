/* An adjusted version of SCF parallel loop fusion. 
 *
 * The largest difference is that this version *does not* check for aliasing.
 * Additionally, the input expected is different from what is expected in
 * vanilla SCF parallel loop fusion.
 */

#include "Transform/Kernel/KernelPasses.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/Passes.h"
#include <algorithm>

namespace mlir {

#define GEN_PASS_DEF_SCFPARALLELLOOPFUSION
#include "mlir/Dialect/SCF/Transforms/Passes.h.inc"
namespace kernel {

using ParallelOp = scf::ParallelOp;
using ReduceOp = scf::ReduceOp;

//===----------------------------------------------------------------------===//
// ParallelLoopFusion (modified version of upstream SCF parallel loop fusion) 
// NOTE: most of the functions here are from SCF parallel loop fusion and are
// either slightly modified or left entirely the same as they appear in that
// file.
//===----------------------------------------------------------------------===//

/// Verify there are no nested ParallelOps.
static bool hasNestedParallelOp(ParallelOp ploop) {
  auto walkResult =
      ploop.getBody()->walk([](ParallelOp) { return WalkResult::interrupt(); });
  return walkResult.wasInterrupted();
}

/// Retrieve loop bounds/steps as constants
std::optional<SmallVector<int64_t>> getConstBounds(OperandRange bounds) {
  return getConstantIntValues(getAsOpFoldResult(SmallVector<Value>(bounds)));
}

/// Checks if the parallel loops have mixed access to the same buffers. Returns
/// `true` if the first parallel loop writes to the same indices that the second
/// loop reads.
static bool haveNoReadsAfterWriteExceptSameIndex(
    ParallelOp firstPloop, ParallelOp secondPloop,
    const IRMapping &firstToSecondPloopIndices) {
  DenseMap<Value, SmallVector<ValueRange, 1>> bufferStores;
  SmallVector<Value> bufferStoresVec;
  firstPloop.getBody()->walk([&](memref::StoreOp store) {
    bufferStores[store.getMemRef()].push_back(store.getIndices());
    bufferStoresVec.emplace_back(store.getMemRef());
  });
  auto walkResult = secondPloop.getBody()->walk([&](memref::LoadOp load) {
    Value loadMem = load.getMemRef();
    // Stop if the memref is defined in secondPloop body. Careful alias analysis
    // is needed.
    auto *memrefDef = loadMem.getDefiningOp();
    if (memrefDef && memrefDef->getBlock() == load->getBlock())
      return WalkResult::interrupt();

    auto write = bufferStores.find(loadMem);
    if (write == bufferStores.end())
      return WalkResult::advance();

    // Check that at last one store was retrieved
    if (!write->second.size())
      return WalkResult::interrupt();

    auto storeIndices = write->second.front();

    // Multiple writes to the same memref are allowed only on the same indices
    for (const auto &othStoreIndices : write->second) {
      if (othStoreIndices != storeIndices)
        return WalkResult::interrupt();
    }

    // Check that the load indices of secondPloop coincide with store indices of
    // firstPloop for the same memrefs.
    auto loadIndices = load.getIndices();
    if (storeIndices.size() != loadIndices.size())
      return WalkResult::interrupt();
    for (int i = 0, e = storeIndices.size(); i < e; ++i) {
      if (firstToSecondPloopIndices.lookupOrDefault(storeIndices[i]) !=
          loadIndices[i]) {
        auto *storeIndexDefOp = storeIndices[i].getDefiningOp();
        auto *loadIndexDefOp = loadIndices[i].getDefiningOp();
        if (storeIndexDefOp && loadIndexDefOp) {
          if (!isMemoryEffectFree(storeIndexDefOp))
            return WalkResult::interrupt();
          if (!isMemoryEffectFree(loadIndexDefOp))
            return WalkResult::interrupt();
          if (!OperationEquivalence::isEquivalentTo(
                  storeIndexDefOp, loadIndexDefOp,
                  [&](Value storeIndex, Value loadIndex) {
                    if (firstToSecondPloopIndices.lookupOrDefault(storeIndex) !=
                        firstToSecondPloopIndices.lookupOrDefault(loadIndex))
                      return failure();
                    else
                      return success();
                  },
                  /*markEquivalent=*/nullptr,
                  OperationEquivalence::Flags::IgnoreLocations)) {
            return WalkResult::interrupt();
          }
        } else
          return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return !walkResult.wasInterrupted();
}

/// Analyzes dependencies in the most primitive way by checking simple read and
/// write patterns.
static LogicalResult verifyDependencies(ParallelOp firstPloop,
                                        ParallelOp secondPloop) {
  IRMapping firstToSecondPloopIndices;
  firstToSecondPloopIndices.map(firstPloop.getBody()->getArguments(),
                                secondPloop.getBody()->getArguments());
  if (!haveNoReadsAfterWriteExceptSameIndex(firstPloop, secondPloop,
                                            firstToSecondPloopIndices))
    return failure();

  IRMapping secondToFirstPloopIndices;
  secondToFirstPloopIndices.map(secondPloop.getBody()->getArguments(),
                                firstPloop.getBody()->getArguments());
  return success(haveNoReadsAfterWriteExceptSameIndex(
      secondPloop, firstPloop, secondToFirstPloopIndices));
}

static bool isFusionLegal(ParallelOp firstPloop, ParallelOp secondPloop) {
  return !hasNestedParallelOp(firstPloop) &&
         !hasNestedParallelOp(secondPloop) &&
         succeeded(verifyDependencies(firstPloop, secondPloop));
}

/// Prepends operations of firstPloop's body into secondPloop's body.
/// Updates secondPloop with new loop.
static bool fuseIfLegal(ParallelOp firstPloop, ParallelOp &secondPloop,
                        OpBuilder builder) {

  if (!isFusionLegal(firstPloop, secondPloop))
    return false;

  DominanceInfo dom;
  // We are fusing first loop into second, make sure there are no users of the
  // first loop results between loops.
  for (Operation *user : firstPloop->getUsers())
    if (!dom.properlyDominates(secondPloop, user, /*enclosingOpOk*/ false))
      return false;

  ValueRange inits1 = firstPloop.getInitVals();
  ValueRange inits2 = secondPloop.getInitVals();

  SmallVector<Value> newInitVars(inits1.begin(), inits1.end());
  newInitVars.append(inits2.begin(), inits2.end());

  IRRewriter b(builder);
  b.setInsertionPoint(secondPloop);
  auto newSecondPloop = b.create<ParallelOp>(
      secondPloop.getLoc(), secondPloop.getLowerBound(),
      secondPloop.getUpperBound(), secondPloop.getStep(), newInitVars);

  Block *block1 = firstPloop.getBody();
  Block *block2 = secondPloop.getBody();
  Block *newBlock = newSecondPloop.getBody();
  auto term1 = cast<ReduceOp>(block1->getTerminator());
  auto term2 = cast<ReduceOp>(block2->getTerminator());

  b.inlineBlockBefore(block2, newBlock, newBlock->begin(),
                      newBlock->getArguments());
  b.inlineBlockBefore(block1, newBlock, newBlock->begin(),
                      newBlock->getArguments());

  ValueRange results = newSecondPloop.getResults();
  if (!results.empty()) {
    b.setInsertionPointToEnd(newBlock);

    ValueRange reduceArgs1 = term1.getOperands();
    ValueRange reduceArgs2 = term2.getOperands();
    SmallVector<Value> newReduceArgs(reduceArgs1.begin(), reduceArgs1.end());
    newReduceArgs.append(reduceArgs2.begin(), reduceArgs2.end());

    auto newReduceOp = b.create<scf::ReduceOp>(term2.getLoc(), newReduceArgs);

    for (auto &&[i, reg] : llvm::enumerate(llvm::concat<Region>(
             term1.getReductions(), term2.getReductions()))) {
      Block &oldRedBlock = reg.front();
      Block &newRedBlock = newReduceOp.getReductions()[i].front();
      b.inlineBlockBefore(&oldRedBlock, &newRedBlock, newRedBlock.begin(),
                          newRedBlock.getArguments());
    }

    firstPloop.replaceAllUsesWith(results.take_front(inits1.size()));
    secondPloop.replaceAllUsesWith(results.take_back(inits2.size()));
  }
  term1->erase();
  term2->erase();
  firstPloop.erase();
  secondPloop.erase();
  secondPloop = newSecondPloop;

  return true;
}

void naivelyFuseParallelOps(Region &region) {
  OpBuilder b(region);
  // Consider every single block and attempt to fuse adjacent loops.
  SmallVector<SmallVector<ParallelOp>, 1> ploopChains;
  for (auto &block : region) {
    ploopChains.clear();
    ploopChains.push_back({});

    // Not using `walk()` to traverse only top-level parallel loops and also
    // make sure that there are no side-effecting ops between the parallel
    // loops.
    bool noSideEffects = true;
    for (auto &op : block) {
      if (auto ploop = dyn_cast<ParallelOp>(op)) {
        if (noSideEffects) {
          ploopChains.back().push_back(ploop);
        } else {
          ploopChains.push_back({ploop});
          noSideEffects = true;
        }
        continue;
      }
      // TODO: Handle region side effects properly.
      noSideEffects &= isMemoryEffectFree(&op) && op.getNumRegions() == 0;
    }
    for (MutableArrayRef<ParallelOp> ploops : ploopChains) {
      for (int i = 0, e = ploops.size(); i + 1 < e; ++i)
        fuseIfLegal(ploops[i], ploops[i + 1], b);
    }
  }
}

#define GEN_PASS_DEF_KERNELDOMAINFUSION
#include "Transform/Kernel/KernelPasses.h.inc"

struct KernelDomainFusion
    : public mlir::impl::SCFParallelLoopFusionBase<KernelDomainFusion> {

  void runOnOperation() override {
    getOperation()->walk([&](Operation *child) {
      for (Region &region : child->getRegions())
        naivelyFuseParallelOps(region);
    });

    return;
  }
};

std::unique_ptr<Pass> createKernelDomainFusionPass() {
  return std::make_unique<KernelDomainFusion>();
}

}
}

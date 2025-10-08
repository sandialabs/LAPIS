/* An adjusted version of SCF parallel loop fusion. 
 *
 * The largest difference is that this version *does not* check for aliasing.
 * Additionally, the input expected is different from what is expected in
 * vanilla SCF parallel loop fusion.
 */

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"

#include "Transform/Kernel/KernelPasses.h"

namespace mlir {

#define GEN_PASS_DEF_SCFPARALLELLOOPFUSION
#include "mlir/Dialect/SCF/Transforms/Passes.h.inc"
namespace kernel {

using ParallelOp = scf::ParallelOp;
using ReduceOp = scf::ReduceOp;

std::optional<SmallVector<int64_t>> getConstBounds(OperandRange bounds) {
  return getConstantIntValues(getAsOpFoldResult(SmallVector<Value>(bounds)));
}

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
    auto *memrefDef = loadMem.getDefiningOp();
    if (memrefDef && memrefDef->getBlock() == load->getBlock())
      return WalkResult::interrupt();

    auto write = bufferStores.find(loadMem);
    if (write == bufferStores.end())
      return WalkResult::advance();

    if (!write->second.size())
      return WalkResult::interrupt();

    auto storeIndices = write->second.front();

    for (const auto &othStoreIndices : write->second) {
      if (othStoreIndices != storeIndices)
        return WalkResult::interrupt();
    }

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
  return succeeded(verifyDependencies(firstPloop, secondPloop));
}

static bool fuseIfLegal(ParallelOp firstPloop, ParallelOp &secondPloop,
                        OpBuilder builder) {

  if (!isFusionLegal(firstPloop, secondPloop))
    return false;

  DominanceInfo dom;
  for (Operation *user : firstPloop->getUsers())
    if (!dom.properlyDominates(secondPloop, user, /*enclosingOpOk*/ true))
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
  SmallVector<SmallVector<ParallelOp>, 1> ploopChains;
  for (auto &block : region) {
    ploopChains.clear();
    ploopChains.push_back({});

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

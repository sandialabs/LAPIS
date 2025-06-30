//===- MemrefsToKokkosScratch.cpp -
// Patterns to place memref allocations in Kokkos scratch
//--------------------===//

#include "lapis/Dialect/Kokkos/IR/KokkosDialect.h"
#include "lapis/Dialect/Kokkos/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/IR/Dominance.h"
#include <vector>

namespace mlir {
#define GEN_PASS_DEF_MEMREFTOKOKKOSSCRATCH
#include "lapis/Dialect/Kokkos/Transforms/Passes.h.inc"
}

using namespace mlir;

// Uncomment to print out detailed debug info about allocation scheduling
// #define SCRATCH_ALLOCATION_DEBUG

struct UndirectedGraph
{
  UndirectedGraph() = delete;
  UndirectedGraph(int n_) {
    n = n_;
    data = std::vector<bool>(n * n, false);
  }

  // only use the upper-triangular part of the adjacency matrix
  void set(int i, int j) {
    if(i < j)
      data[i * n + j] = true;
    else if(j < i)
      data[j * n + i] = true;
  }

  bool get(int i, int j) const {
    if(i < j)
      return data[i * n + j];
    else if(j < i)
      return data[j * n + i];
    return true; // assume self-loops exist (should never affect conflict analysis)
  }

  SmallVector<int> adj(int i) const {
    SmallVector<int> adj;
    for(int j = 0; j < n; j++) {
      if(get(i, j))
        adj.push_back(j);
    }
    return adj;
  }

  void print() const {
    llvm::outs() << " ";
    // header row to help readability
    for(int i = 0; i < n; i++)
      llvm::outs() << i % 10;
    llvm::outs() << '\n';
    for(int i = 0; i < n; i++) {
      // header column too
      llvm::outs() << i % 10;
      for(int j = 0; j < n; j++) {
        if(get(i, j))
          llvm::outs() << "X";
        else
          llvm::outs() << " ";
      }
      llvm::outs() << "\n";
    }
  }

  std::vector<bool> data;
  int n;
};

// Given memref v, recursively find other memrefs which alias it
static void findAliasingMemrefs(Value v, SmallVector<Value>& aliasing) {
  for(OpOperand& use : v.getUses()) {
    Operation *usingOp = use.getOwner();
    if(kokkos::isViewAliasingOp(usingOp)) {
      Value a = usingOp->getResult(0);
      aliasing.push_back(a);
      findAliasingMemrefs(a, aliasing);
    }
  }
}

struct Allocation
{
  int id;
  size_t addr;
  size_t size;
};

static bool intersect(size_t begin1, size_t end1, size_t begin2, size_t end2) {
  return begin1 < end2 && begin2 < end1;
}

struct AllocationSet
{
  SmallVector<Allocation> allocs;

  // Greedily find non-conflicting location for a new allocation and add it to list
  void place(const UndirectedGraph& conflicts, int id, size_t size, size_t alignment)
  {
    size_t begin = 0;
    while(true) {
      // Ensure alignment is correct
      if(begin % alignment)
        begin += (alignment - begin % alignment);
      // Does placement at addr cause a conflict, and if so where does
      // the largest conflicting allocation end?
      bool conf = false;
      size_t confEnd = 0;
      size_t end = begin + size;
      for(auto& alloc : allocs) {
        if(!conflicts.get(id, alloc.id))
          continue;
        size_t allocBegin = alloc.addr;
        size_t allocEnd = alloc.addr + alloc.size;
        if(intersect(begin, end, allocBegin, allocEnd)) {
          conf = true;
          if(allocEnd > confEnd)
            confEnd = allocEnd;
        }
      }
      if(conf) {
        begin = confEnd;
      }
      else {
        allocs.push_back({id, begin, size});
        break;
      }
    }
  }

  size_t totalScratchUsed() const {
    size_t max = 0;
    for(auto& a : allocs) {
      size_t end = a.addr + a.size;
      if(end > max)
        max = end;
    }
    return max;
  }
};

struct MemrefToKokkosScratchPass 
    : public impl::MemrefToKokkosScratchBase<MemrefToKokkosScratchPass> {

  MemrefToKokkosScratchPass() = default;
  MemrefToKokkosScratchPass(const MemrefToKokkosScratchPass& pass) = default;

  void runOnOperation() override {
    // NOTE: This pass assumes that earlier bufferization passes have
    // placed memref.alloc ops as late as possible in the IR. Otherwise the memory placement can be very sub-optimal
    MLIRContext* ctx = &getContext();
    IRRewriter rewriter(ctx);
    func::FuncOp func = getOperation();
    // Walk the function body and find allocations.
    // - check that the size of the allocation in bytes is known at compile time
    // - number each allocation
    // - look for uses of each allocation which alias (shallow copy) it

    // For each alloc result, associate a number starting at 0 (for building conflict graph later)
    SmallVector<Value> allocations;
    DenseMap<Value, int> allocNumbers;
    // For each alloc result, this is a list of aliasing memref values
    DenseMap<Value, SmallVector<Value>> aliases;
    int allocCounter = 0;
    func->walk([&](memref::AllocOp alloc) {
      MemRefType mrt = cast<MemRefType>(alloc.getResult().getType());
      size_t unused;
      if(!kokkos::memrefSizeInBytesKnown(mrt, unused, alloc)) {
        alloc.emitOpError("cannot determine allocation size in bytes");
      }
      Value result = alloc.getResult();
      allocations.push_back(result);
      allocNumbers[result] = allocCounter++;
      SmallVector<Value> aliasing;
      findAliasingMemrefs(result, aliasing);
      aliases[result] = aliasing;
    });
    // Initialize conflict graph
    UndirectedGraph conflictGraph(allocCounter);
    // Create liveness analysis over all values and operations in function body
    Liveness live(func);
    // Create dominance analysis over operations
    DominanceInfo dom(func);
    // Given value V, op O, liveness can efficiently query if V is known to be dead at all ops dominated by O.
    // But it will not tell you if the beginning of V's live range is dominated by O
    // (meaning V has definitely not been generated yet at point O). Use Dominance analysis to check for that.
    auto couldBeAlive = [&](Value V, Operation* O) {
#ifdef SCRATCH_ALLOCATION_DEBUG
      llvm::outs() << "Checking if value V, generated by: ";
      V.getDefiningOp()->dump();
      llvm::outs() << "is possibly alive at op ";
      O->dump();
#endif
      bool knownDead = live.isDeadAfter(V, O);
#ifdef SCRATCH_ALLOCATION_DEBUG
      llvm::outs() << "known dead by then? " << knownDead << '\n';
#endif
      Operation* Vop = V.getDefiningOp();
      if(!Vop) {
        // Can't make any determination about V as it's a block argument
        return true;
      }
      bool knownNotAliveYet = dom.dominates(O, Vop);
#ifdef SCRATCH_ALLOCATION_DEBUG
      llvm::outs() << "known to not yet be alive? " << knownNotAliveYet << '\n';
#endif
      return !(knownDead || knownNotAliveYet);
    };
    // If an allocation A is alive when B is allocated, then we know that A and B cannot overlap in their memory ranges.
    // B's live range begins when it is allocated, so it is sufficient to only check for overlap with A's live at the op
    // where B is allocated.
    func->walk([&](memref::AllocOp alloc) {
      // At the point immediately before alloc, find all currently live allocations.
      // Add these edges to conflict graph.
      int thisAlloc = allocNumbers[alloc.getResult()];
      for(int otherAlloc = 0; otherAlloc < allocCounter; otherAlloc++) {
        if(otherAlloc == thisAlloc)
          continue;
        Value other = allocations[otherAlloc];
        bool conflicts = false;
        do {
          if(couldBeAlive(other, alloc)) {
            conflicts = true;
            break;
          }
          // also check every memref which aliases other
          for(Value alias : aliases[other]) {
            if(couldBeAlive(alias, alloc)) {
              conflicts = true;
              break;
            }
          }
        } while(false);
        if(conflicts) {
          conflictGraph.set(thisAlloc, otherAlloc);
        }
      }
    });
#ifdef SCRATCH_ALLOCATION_DEBUG
    llvm::outs() << "Computed conflict graph:\n";
    conflictGraph.print();
    llvm::outs() << "Size (bytes) for each alloc:\n";
    for(int i = 0; i < allocCounter; i++) {
      size_t size;
      Value a = allocations[i];
      (void) kokkos::memrefSizeInBytesKnown(cast<MemRefType>(a.getType()), size, a.getDefiningOp());
      llvm::outs() << "#" << i << ": " << size << " bytes\n";
    }
    llvm::outs() << "Alignment (bytes) for each alloc:\n";
    for(int i = 0; i < allocCounter; i++) {
      MemRefType mrt = cast<MemRefType>(allocations[i].getType());
      llvm::outs() << "#" << i << ": " << kokkos::getBuiltinTypeSize(mrt.getElementType(), allocations[i].getDefiningOp()) << " bytes\n";
    }
#endif
    // Now decide where to place each allocation using a greedy strategy
    // Initially, memory is empty. Until all allocations are placed:
    // - pick smallest allocation. In case of tie, pick earliest allocation (aka smallest allocNumber)
    // - put it in the lowest non-conflicting address which satisfies alignment
    // For MALA snap model, this strategy gives the optimal placement.
    // TODO: if the number of allocations is small (say, 8 or fewer) then do an exhaustive search on the order in which to place,
    // and pick the one with the lowest total scratch usage
    //
    // Sort the allocations into placement order
    SmallVector<Allocation> allocsToPlace;
    for(int i = 0; i < allocCounter; i++) {
      Value a = allocations[i];
      size_t size;
      (void) kokkos::memrefSizeInBytesKnown(cast<MemRefType>(a.getType()), size, a.getDefiningOp());
      allocsToPlace.push_back({i, 0, size});
    }
    std::stable_sort(allocsToPlace.begin(), allocsToPlace.end(),
      [](const Allocation& a1, const Allocation& a2) -> bool {
        if(a1.size < a2.size)
          return true;
        else
          return false;
      });
    AllocationSet allocSet;
    for(Allocation& a : allocsToPlace) {
      MemRefType mrt = cast<MemRefType>(allocations[a.id].getType());
      int alignment = kokkos::getBuiltinTypeSize(mrt.getElementType(), func);
      allocSet.place(conflictGraph, a.id, a.size, alignment);
    }
#ifdef SCRATCH_ALLOCATION_DEBUG
    llvm::outs() << "Placed all allocations greedily:\n";
    for(auto& alloc : allocSet.allocs) {
      llvm::outs() << "#" << alloc.id << ": [" << alloc.addr << "..." << alloc.addr + alloc.size << ")\n";
    }
    llvm::outs() << "Total scratch requirement: " << allocSet.totalScratchUsed() << '\n';
#endif
    // Now that all allocations have been placed, replace each memref.alloc with kokkos.alloc_scratch
    for(Allocation& alloc : allocSet.allocs) {
      memref::AllocOp oldOp = cast<memref::AllocOp>(allocations[alloc.id].getDefiningOp());
      MemRefType mrt = cast<MemRefType>(oldOp.getResult().getType());
      rewriter.setInsertionPoint(oldOp);
      auto newOp = rewriter.create<kokkos::AllocScratchOp>(oldOp->getLoc(), mrt, rewriter.getIndexAttr(alloc.addr));
      rewriter.replaceOp(oldOp, newOp);
    }
  }
};

std::unique_ptr<Pass> mlir::createMemrefToKokkosScratchPass()
{
  return std::make_unique<MemrefToKokkosScratchPass>();
}


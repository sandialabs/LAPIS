//===- KokkosLoopMapping.cpp - Pattern for kokkos-loop-mapping pass
//--------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lapis/Dialect/Kokkos/IR/KokkosDialect.h"
#include "lapis/Dialect/Kokkos/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;

namespace {

// Get the number of parallel nesting levels for the given ParallelOp
// - The op itself counts as 1
// - Each additional nesting level counts as another
static int getParallelNumLevels(scf::ParallelOp op) {
  // selfDepth will be the nesting depth of op itself
  // (if it has no scf.parallel parents, this is 0)
  int selfDepth = 0;
  int maxDepth = 1;
  // note: walk starting from op visits op itself
  op->walk([&](scf::ParallelOp child) {
    int d = kokkos::getOpParallelDepth(child);
    if (op == child) {
      selfDepth = d - 1;
    } else {
      if (d > maxDepth)
        maxDepth = d;
    }
  });
  return maxDepth - selfDepth;
}

// Within the context of a kokkos.team_parallel: does op imply team-wide
// synchronization? If false, then a kokkos.team_barrier may be required to make
// the side effects/result(s) of op visible to other threads.
//
// - team_barrier syncs
// - a TeamThread or TeamVector parallel reduce syncs, since these
//   behave like all-reduces
// - a per-team Kokkos::single with at least one result is a broadcast, so it
// syncs
//
// - a TeamVector or TeamThread parallel for (no reductions) is async
// - a Kokkos::single with no results is async
// - all other ops have no synchronization behavior and are considered async
/*
bool doesTeamLevelSync(Operation* op)
{
  if(auto rangePar = dyn_cast<kokkos::RangeParallelOp>(op)) {
    auto level = rangePar.getParallelLevel();
    // check if rangePar a) has at least one result and b) is team-level
    return rangePar.getNumResults() &&
      (level == kokkos::ParallelLevel::TeamThread ||
      level == kokkos::ParallelLevel::TeamVector);
  }
  else if(auto single = dyn_cast<kokkos::SingleOp>(op)) {
    return single.getNumResults() && single.getLevel() ==
kokkos::SingleLevel::PerTeam;
  }
  else if(isa<kokkos::TeamBarrierOp>(op))
    return true;
  return false;
}
*/

// Compute the product of some Index-typed values.
// This assumes that overflow can't happen, and that begin != end.
template <typename ValRange>
Value buildIndexProduct(Location loc, RewriterBase &rewriter, ValRange range) {
  auto iter = range.begin();
  Value prod = *iter;
  iter++;
  while (iter != range.end()) {
    prod = rewriter.create<arith::MulIOp>(loc, prod, *iter).getResult();
    iter++;
  }
  return prod;
}

// Return true iff op has side effects that should happen exactly once.
// This means that if it appears in a parallel context, it must be wrapped in a
// kokkos.single.
//
// (!) Note: here, rely on a key difference between SCF and general Kokkos code.
// SCF's nested parallelism acts like a fork-join model, so values scoped
// outside the innermost parallel level cannot differ across threads.
//
// But in Kokkos, values scoped outside the innermost loop are replicated across
// threads/vector lanes, and those replications can have different values (CUDA
// model). The Kokkos dialect is capable of representing this case (e.g. team
// rank is a block argument and other values can be computed based on that), but
// Kokkos dialect code lowered from SCF (that is, all programs this pass is
// applied to) can never do that.
static bool opNeedsSingle(Operation *op) {
  return isa<memref::StoreOp>(op) || isa<memref::AtomicRMWOp>(op) ||
         isa<kokkos::UpdateReductionOp>(op);
}

// If v is the result of a scalar cast op, return the input to cast
// This may trace through multiple different ops
static Value followCast(Value v) {
  auto op = v.getDefiningOp();
  if(op && isa<arith::IndexCastOp, arith::IndexCastUIOp, arith::ExtUIOp>(op)) {
    return followCast(op->getOperand(0));
  }
  return v;
}

// If possible, compute an estimate for the maximum level of parallelism
// for RangeParallelOp ops nested within topParallel at the given level.
// This may insert ops before topParallel to
// compute the value at runtime. Set parallelism (index) to this and return
// success(). topParallel should either be a TeamParallelOp or ThreadParallelOp.
//
// If this is not possible because the nested loop bounds don't follow a
// known pattern, or they aren't loop-invariant, then return failure().
//
// If there are multiple loops at the given nesting level, then take the maximum
// over all loops for which we can figure out a value.
//
// TODO: should we run an LICM pass before this to maximize the cases where this
// will succeed?
static LogicalResult estimateParallelism(Value &parallelism,
                                         RewriterBase &rewriter,
                                         Operation *topParallel,
                                         kokkos::ParallelLevel level) {
  // Walk op, and make a list of all parallels at the given level
  SmallVector<kokkos::RangeParallelOp> nestedLoops;
  topParallel->walk([&](kokkos::RangeParallelOp nested) {
    if (nested.getParallelLevel() == level)
      nestedLoops.push_back(nested);
  });
  // It's possible that topParallel contains no loops at the given level.
  if (nestedLoops.empty())
    return failure();
  // The total parallelism for a loop is the product of its upper bounds.
  // To make an estimate for any one nested loop, we have to know loop-invariant
  // expressions for _all_ its (average) upper bounds.
  Region *enclosingRegion = topParallel->getParentRegion();
  SmallVector<Value> allEstimates;
  rewriter.setInsertionPoint(topParallel);
  for (kokkos::RangeParallelOp nested : nestedLoops) {
    bool matchesPattern = true;
    for (Value bound : nested.getUpperBound()) {
      bound = followCast(bound);
      // If bound is scoped outside topParallel's body, then we already know
      // it's loop-invariant. (enclosingRegion contains topParallel, and
      // bound.getParentRegion() contains bound)
      if (enclosingRegion->isProperAncestor(bound.getParentRegion())) {
        // This bound is not loop-invariant, but check for the CSR iteration
        // pattern. If all these conditions are met:
        // - nested should be directly under topParallel
        // - The upper bound of nested is the result of an arith.subi B,A
        // - if top is a...
        //   TeamParallel: let i be the league rank.
        //   ThreadParallel: let i be the induction variable.
        // - A is a memref.load R[i]
        // - B is a memref.load R[i + 1]
        // - A,B are nested directly under topParallel (no control flow ops
        // enclosing them)
        //
        // Then the average parallelism of the inner loop is (R[N] - R[0]) / N.
        // Since it's better to over-provision than under-provision
        // threads/lanes, we take the ceiling of this: (R[N] - R[0] + N - 1) / N.
        bool followsCsrPattern = true;
        Value csrPatternOffsets;
        // Check for conditions in a do-while, so that break will exit on the
        // first one that's false
        do {
          // Check that there are no other parallel loops nested in topParallel
          // but enclosing nested
          if (kokkos::getOpParallelDepth(topParallel) + 1 !=
              kokkos::getOpParallelDepth(nested)) {
            followsCsrPattern = false;
            break;
          }
          // Get the value that should be the 1D induction variable of
          // topParallel
          // - TeamParallel -> league rank
          // - ThreadParallel -> induction
          Value induction1D;
          if (auto team = dyn_cast<kokkos::TeamParallelOp>(topParallel)) {
            induction1D = team.getLeagueRank();
          } else if (auto thread =
                         dyn_cast<kokkos::ThreadParallelOp>(topParallel)) {
            induction1D = thread.getInductionVar();
          } else {
            followsCsrPattern = false;
            break;
          }
          arith::SubIOp sub = dyn_cast<arith::SubIOp>(bound.getDefiningOp());
          if (!sub) {
            followsCsrPattern = false;
            break;
          }
          Value B = followCast(sub.getLhs());
          Value A = followCast(sub.getRhs());
          auto Aload = dyn_cast<memref::LoadOp>(A.getDefiningOp());
          auto Bload = dyn_cast<memref::LoadOp>(B.getDefiningOp());
          if (!Aload || !Bload) {
            followsCsrPattern = false;
            break;
          }
          if (Aload->getParentOp() != topParallel ||
              Bload->getParentOp() != topParallel) {
            followsCsrPattern = false;
            break;
          }
          // A, B must both read the same memref
          if (Aload.getMemref() != Bload.getMemref()) {
            followsCsrPattern = false;
            break;
          }
          csrPatternOffsets = Aload.getMemref();
          // Offsets should be a 1D memref
          if (Aload.getIndices().size() != 1) {
            followsCsrPattern = false;
            break;
          }
          Value Aindex = followCast(Aload.getIndices().front());
          Value Bindex = followCast(Bload.getIndices().front());
          if (Aindex != induction1D) {
            followsCsrPattern = false;
            break;
          }
          auto Badd = dyn_cast<arith::AddIOp>(Bindex.getDefiningOp());
          if (!Badd) {
            followsCsrPattern = false;
            break;
          }
          if (!(followCast(Badd.getLhs()) == induction1D &&
                kokkos::valueIsIntegerConstantOne(Badd.getRhs())) &&
              !(followCast(Badd.getRhs()) == induction1D &&
                kokkos::valueIsIntegerConstantOne(Badd.getLhs()))) {
            followsCsrPattern = false;
            break;
          }
        } while (0);
        if (!followsCsrPattern) {
          matchesPattern = false;
          break;
        }
      }
    }
    // Now we know whether this loop matches a known pattern. Can start
    // inserting ops to calculate the total parallelism.
    SmallVector<Value> boundsToMultiply;
    if (matchesPattern) {
      for (Value bound : nested.getUpperBound()) {
        if (!enclosingRegion->isProperAncestor(bound.getParentRegion()))
          boundsToMultiply.push_back(bound);
        else {
          auto indexType = rewriter.getIndexType();
          // We know that we previously matched the CSR iteration pattern,
          // so all these casts will succeed
          arith::SubIOp sub = cast<arith::SubIOp>(followCast(bound).getDefiningOp());
          auto Aload = cast<memref::LoadOp>(followCast(sub.getRhs()).getDefiningOp());
          auto offsets = Aload.getMemref();
          Value zero =
              rewriter.create<arith::ConstantIndexOp>(topParallel->getLoc(), 0);
          Value one =
              rewriter.create<arith::ConstantIndexOp>(topParallel->getLoc(), 1);
          Value upper;
          if (auto team = dyn_cast<kokkos::TeamParallelOp>(topParallel)) {
            upper = team.getLeagueSize();
          } else if (auto thread =
                         dyn_cast<kokkos::ThreadParallelOp>(topParallel)) {
            upper = thread.getNumIters();
          }
          Value offsetUpper =
              rewriter
                  .create<memref::LoadOp>(topParallel->getLoc(), offsets, upper)
                  .getResult();
          offsetUpper = 
              rewriter
                  .create<arith::IndexCastOp>(topParallel->getLoc(), indexType, offsetUpper)
                  .getResult();
          Value offsetLower =
              rewriter
                  .create<memref::LoadOp>(topParallel->getLoc(), offsets, zero)
                  .getResult();
          offsetLower = 
              rewriter
                  .create<arith::IndexCastOp>(topParallel->getLoc(), indexType, offsetLower)
                  .getResult();
          Value diff = rewriter
                           .create<arith::SubIOp>(topParallel->getLoc(),
                                                  offsetUpper, offsetLower)
                           .getResult();
          Value diffPlusUpper =
              rewriter.create<arith::AddIOp>(topParallel->getLoc(), diff, upper)
                  .getResult();
          Value diffPlusUpperMinus1 =
              rewriter
                  .create<arith::SubIOp>(topParallel->getLoc(), diffPlusUpper,
                                         one)
                  .getResult();
          Value estimate =
              rewriter
                  .create<arith::DivUIOp>(topParallel->getLoc(),
                                          diffPlusUpperMinus1, upper)
                  .getResult();
          boundsToMultiply.push_back(estimate);
        }
      }
    }
    Value totalEstimate =
        buildIndexProduct(topParallel->getLoc(), rewriter, boundsToMultiply);
    allEstimates.push_back(totalEstimate);
  }
  if (allEstimates.empty())
    return failure();
  // Compute max over all estimates
  Value maxEstimate = allEstimates.front();
  for (size_t i = 1; i < allEstimates.size(); i++) {
    maxEstimate = rewriter.create<arith::MaxUIOp>(topParallel->getLoc(),
                                                  maxEstimate, allEstimates[i]);
  }
  parallelism = maxEstimate;
  return success();
}

static LogicalResult scfParallelToSequential(scf::ForOp &newOp,
                                             RewriterBase &rewriter,
                                             scf::ParallelOp op) {
  using ValueVector = SmallVector<Value>;
  rewriter.setInsertionPoint(op);
  // TODO: support > 1 reductions here.
  int numReductions = op.getNumReductions();
  if (numReductions > 1)
    return op.emitError(
        "Only support translating scf.parallel with 0 or 1 reductions.");
  scf::LoopNest newLoopNest = scf::buildLoopNest(
      rewriter, op.getLoc(), op.getLowerBound(), op.getUpperBound(),
      op.getStep(), op.getInitVals(),
      [&](OpBuilder &builder, Location loc, ValueRange inductionVars,
          ValueRange args) -> ValueVector {
        ValueVector results;
        IRMapping irMap;
        for (auto p : llvm::zip(op.getInductionVars(), inductionVars)) {
          irMap.map(std::get<0>(p), std::get<1>(p));
        }
        for (Operation &oldOp : op.getBody()->getOperations()) {
          if (auto reduce = dyn_cast<scf::ReduceOp>(oldOp)) {
            // Reduce with 0 operands is just a simple terminator
            if (reduce.getNumOperands() == 0)
              continue;
            // TODO: this should be modified to support an
            // arbitrary number of reductions. The basic structure is the same,
            // but now the reduce can have N operands and N regions, each doing
            // one join
            Value valToJoin = irMap.lookupOrDefault(reduce.getOperand(0));
            Value partialReduction = args[0];
            for (Region &reduceRegion : reduce->getRegions()) {
              Block &reduceBlock = reduceRegion.front();
              irMap.map(reduceBlock.getArguments()[0], partialReduction);
              irMap.map(reduceBlock.getArguments()[1], valToJoin);
              for (Operation &oldReduceOp : reduceBlock.getOperations()) {
                if (auto reduceReturn =
                        dyn_cast<scf::ReduceReturnOp>(oldReduceOp)) {
                  // reduceReturn's operand is the updated partial reduction.
                  // Yield this back to the loop (final result or passed to next
                  // iteration)
                  results.push_back(
                      irMap.lookupOrDefault(reduceReturn.getOperand()));
                } else {
                  // All other ops: just inline into new loop body
                  builder.clone(oldReduceOp, irMap);
                }
              }
            }
          } else {
            // For all ops besides reduce, just inline into the new loop
            builder.clone(oldOp, irMap);
          }
        }
        return results;
      });
  newOp = newLoopNest.loops.front();
  rewriter.replaceOp(op, newOp);
  return success();
}

// Clone the given operation into a new Kokkos parallel loop.
// Assume that the rewriter's insertion position is already in the right place.
// This is mostly a standard clone except for scf.reduce:
// - with 0 operands (simple terminator) is not copied.
//   This is because kokkos.yield is inserted automatically to the end of each
//   loop body.
// - with 1 operand is converted into a kokkos.update_reduction
//   - and inside the reduce, scf.reduce.return is converted to kokkos.yield
// - with >1 operands: TODO, multiple reductions, not supported yet by Kokkos IR
// but will be in future
static LogicalResult inlineLoopBodyOp(RewriterBase &rewriter, Operation *op,
                                      IRMapping &irMap, Value reduceIdentity) {
  if (scf::ReduceOp reduce = dyn_cast<scf::ReduceOp>(op)) {
    int numReductions = reduce->getNumOperands();
    if (numReductions > 1)
      return reduce->emitError(
          "Currently cannot lower scf::ReduceOp with >1 reductions.");
    if (numReductions == 0) {
      // Simple terminator, does nothing
      // The terminator for the new Kokkos loop (kokkos.yield) is automatically
      // inserted already
      return success();
    }
    // TODO: support multiple reductions here!
    kokkos::UpdateReductionOp newReduce =
        rewriter.create<kokkos::UpdateReductionOp>(
            reduce->getLoc(), irMap.lookupOrDefault(reduce.getOperand(0)),
            irMap.lookupOrDefault(reduceIdentity));
    // Map (old to new) the two reduce block arguments
    Block &oldReduceBlock = reduce.getReductions()[0].front();
    Block &newReduceBlock = newReduce.getReductionOperator().front();
    for (auto p : llvm::zip(oldReduceBlock.getArguments(),
                            newReduceBlock.getArguments())) {
      irMap.map(std::get<0>(p), std::get<1>(p));
    }
    auto ip = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(
        &newReduce.getReductionOperator().front());
    for (Operation &oldOp : oldReduceBlock.getOperations()) {
      if (scf::ReduceReturnOp reduceReturn =
              dyn_cast<scf::ReduceReturnOp>(oldOp)) {
        // this is not a result of reduce return; the operand of reduceReturn is
        // actually named "result"
        rewriter.create<kokkos::YieldOp>(
            reduceReturn.getLoc(),
            irMap.lookupOrDefault(reduceReturn.getOperand()));
      } else {
        rewriter.clone(oldOp, irMap);
      }
    }
    rewriter.restoreInsertionPoint(ip);
  } else {
    rewriter.clone(*op, irMap);
  }
  return success();
}

// Rewrite the given scf.parallel as a kokkos.parallel, with the given execution
// space and nesting level (not for TeamPolicy loops) Return the new op, or NULL
// if failed.
static LogicalResult scfParallelToKokkosRange(kokkos::RangeParallelOp &newOp,
                                              RewriterBase &rewriter,
                                              scf::ParallelOp op,
                                              kokkos::ExecutionSpace exec,
                                              kokkos::ParallelLevel level) {
  rewriter.setInsertionPoint(op);
  // If the loop has a reduction, reduceIdentity gives its identity element
  // and resultTypes contains its type. Otherwise, they are nullptr and empty
  // respectively.
  Value reduceIdentity = nullptr;
  SmallVector<Type> resultTypes;
  if (op.getInitVals().size()) {
    reduceIdentity = op.getInitVals().front();
    resultTypes.push_back(reduceIdentity.getType());
  }
  // Create the kokkos.parallel but don't populate the body yet
  auto kokkosRange = rewriter.create<kokkos::RangeParallelOp>(
      op.getLoc(), exec, level, op.getUpperBound(), resultTypes);
  // Now inline the old loop's operations into the new loop (replacing all
  // usages of the induction variables) Need to omit scf.yield, so can't just
  // use rewriter.inlineBlockBefore
  rewriter.setInsertionPointToStart(&kokkosRange.getLoopBody().front());
  IRMapping irMap;
  for (auto p :
       llvm::zip(op.getInductionVars(), kokkosRange.getInductionVars())) {
    irMap.map(std::get<0>(p), std::get<1>(p));
  }
  for (Operation &oldOp : op.getBody()->getOperations()) {
    auto result = inlineLoopBodyOp(rewriter, &oldOp, irMap, reduceIdentity);
    if (failed(result))
      return result;
  }
  // The new loop is fully constructed.
  // Erase the old loop and replace uses of its results with those of the new
  // loop.
  rewriter.replaceOp(op, kokkosRange);
  newOp = kokkosRange;
  return success();
}

// Convert an scf.parallel to a TeamPolicy parallel_for.
// The outer loop always uses a 1D iteration space where 1 team == 1 iteration.
// We can recover the original ND induction variables from the league rank.
static LogicalResult scfParallelToKokkosTeam(kokkos::TeamParallelOp &newOp,
                                             RewriterBase &rewriter,
                                             scf::ParallelOp op,
                                             Value teamSizeHint,
                                             Value vectorLengthHint) {
  rewriter.setInsertionPoint(op);
  // First, compute the league size as the product of iteration bounds of op
  // (this is easy because the bounds are in [0:N:1] form)
  auto origLoopBounds = op.getUpperBound();
  Value leagueSize = buildIndexProduct(op.getLoc(), rewriter, origLoopBounds);
  // If the loop has a reduction, reduceIdentity gives its identity element
  // and resultTypes contains its type. Otherwise, they are nullptr and empty
  // respectively.
  Value reduceIdentity = nullptr;
  SmallVector<Type> resultTypes;
  if (op.getInitVals().size()) {
    reduceIdentity = op.getInitVals().front();
    resultTypes.push_back(reduceIdentity.getType());
  }
  // Create the kokkos.parallel but don't populate the body yet
  auto kokkosTeam = rewriter.create<kokkos::TeamParallelOp>(
      op.getLoc(), leagueSize, teamSizeHint, vectorLengthHint, resultTypes);
  // Now inline the old loop's operations into the new loop.
  rewriter.setInsertionPointToStart(&kokkosTeam.getLoopBody().front());
  IRMapping irMap;
  auto origInductionVars = op.getInductionVars();
  int n = origInductionVars.size();
  Value leagueRank = kokkosTeam.getLeagueRank();
  if (n == 1) {
    irMap.map(origInductionVars[0], leagueRank);
  } else {
    // If the original loop was not 1D, use arith.remui and arith.divui to get
    // the original (n-dimensional) induction variables in terms of the single
    // league rank. Treat the last dimension the fastest varying.
    Value leagueRankRemaining = leagueSize;
    for (int i = n - 1; i >= 0; i--) {
      Value thisLoopSize = origLoopBounds[i];
      // First, use remainder (remui) to get the current induction var
      Value newInductionVar =
          rewriter
              .create<arith::RemUIOp>(op.getLoc(), leagueRankRemaining,
                                      thisLoopSize)
              .getResult();
      // Then (if there are more loops after this), divide the remaining league
      // rank by the loop size, rounding down
      if (i != 0) {
        leagueRankRemaining =
            rewriter
                .create<arith::DivUIOp>(op.getLoc(), leagueRankRemaining,
                                        thisLoopSize)
                .getResult();
      }
      // Map old to new, for when we generate the new body
      irMap.map(origInductionVars[i], newInductionVar);
    }
  }
  for (Operation &oldOp : op.getBody()->getOperations()) {
    auto result = inlineLoopBodyOp(rewriter, &oldOp, irMap, reduceIdentity);
    if (failed(result))
      return result;
  }
  // The new loop is fully constructed. Erase the old loop and replace uses of
  // its results with those of the new loop.
  rewriter.replaceOp(op, kokkosTeam);
  newOp = kokkosTeam;
  return success();
}

// Convert an scf.parallel to a kokkos.thread_parallel.
// This represents a common pattern where the top two levels of a TeamPolicy
// (TeamPolicy and TeamThread) are used together to iterate over one flat range.
// This leaves the last level (ThreadVector) available to iterate over
// additional nested loops.
static LogicalResult scfParallelToKokkosThread(kokkos::ThreadParallelOp &newOp,
                                               RewriterBase &rewriter,
                                               scf::ParallelOp op,
                                               Value vectorLengthHint) {
  rewriter.setInsertionPoint(op);
  // First, compute the league size as the product of iteration bounds of op
  // (this is easy because the bounds are in [0:N:1] form)
  auto origLoopBounds = op.getUpperBound();
  Value numIters = buildIndexProduct(op.getLoc(), rewriter, origLoopBounds);
  // If the loop has a reduction, reduceIdentity gives its identity element
  // and resultTypes contains its type. Otherwise, they are nullptr and empty
  // respectively.
  Value reduceIdentity = nullptr;
  SmallVector<Type> resultTypes;
  if (op.getInitVals().size()) {
    reduceIdentity = op.getInitVals().front();
    resultTypes.push_back(reduceIdentity.getType());
  }
  // Create the kokkos.parallel but don't populate the body yet
  auto kokkosThread = rewriter.create<kokkos::ThreadParallelOp>(
      op.getLoc(), numIters, vectorLengthHint, resultTypes);
  // Now inline the old loop's operations into the new loop.
  rewriter.setInsertionPointToStart(&kokkosThread.getLoopBody().front());
  IRMapping irMap;
  auto origInductionVars = op.getInductionVars();
  int n = origInductionVars.size();
  // Use arith.remui and arith.divui to get the original (n-dimensional)
  // induction variables in terms of the single induction variable passed to the
  // new loop body. Treat the last dimension the fastest varying.
  Value inductionVar = kokkosThread.getInductionVar();
  if (n == 1) {
    irMap.map(origInductionVars[0], inductionVar);
  } else {
    Value iterCountRemaining = numIters;
    for (int i = n - 1; i >= 0; i--) {
      Value thisLoopSize = origLoopBounds[i];
      // First, use remainder (remui) to get the current induction var
      Value newInductionVar =
          rewriter
              .create<arith::RemUIOp>(op.getLoc(), iterCountRemaining,
                                      thisLoopSize)
              .getResult();
      // Then (if there are more loops after this), divide the remaining league
      // rank by the loop size, rounding down
      if (i != 0) {
        iterCountRemaining =
            rewriter
                .create<arith::DivUIOp>(op.getLoc(), iterCountRemaining,
                                        thisLoopSize)
                .getResult();
      }
      // Map old to new, for when we generate the new body
      irMap.map(origInductionVars[i], newInductionVar);
    }
  }
  for (Operation &oldOp : op.getBody()->getOperations()) {
    auto result = inlineLoopBodyOp(rewriter, &oldOp, irMap, reduceIdentity);
    if (failed(result))
      return result;
  }
  // The new loop is fully constructed.
  // As a last step, erase the old loop and replace uses of its results with
  // those of the new loop.
  rewriter.replaceOp(op, kokkosThread);
  newOp = kokkosThread;
  return success();
}

// ** Helpers to determine the parallel context of ops inside parallel loops **
// is op inside a team parallel (kokkos.TeamParallelOp)?
static bool inTeamLoop(Operation *op) {
  return op->getParentOfType<kokkos::TeamParallelOp>();
}

// is op inside a thread parallel loop? (ThreadParallelOp, or RangeParallelOp
// with TeamThreadRange)
static bool inThreadLoop(Operation *op) {
  if (op->getParentOfType<kokkos::ThreadParallelOp>())
    return true;
  kokkos::RangeParallelOp iter = op->getParentOfType<kokkos::RangeParallelOp>();
  while (iter) {
    if (iter.getParallelLevel() == kokkos::ParallelLevel::TeamThread)
      return true;
    iter = iter->getParentOfType<kokkos::RangeParallelOp>();
  }
  return false;
}

// is op inside a vector parallel loop? (ThreadVectorRange or TeamVectorRange)
static bool inVectorLoop(Operation *op) {
  // if we are inside a vector loop, it must be the innermost RangeParallel
  kokkos::RangeParallelOp par = op->getParentOfType<kokkos::RangeParallelOp>();
  if (!par)
    return false;
  return par.getParallelLevel() == kokkos::ParallelLevel::ThreadVector ||
         par.getParallelLevel() == kokkos::ParallelLevel::TeamVector;
}

// Within the given loop, wrap individual ops inside kokkos.single as needed.
static LogicalResult insertSingleWraps(RewriterBase &rewriter,
                                       Operation *loop) {
  loop->walk<WalkOrder::PostOrder>([&](Operation *op) {
    if (opNeedsSingle(op)) {
      rewriter.setInsertionPoint(op);
      if (inTeamLoop(op) && !inThreadLoop(op)) {
        // op needs to be in PerTeam single
        auto single = rewriter.create<kokkos::SingleOp>(
            op->getLoc(), op->getResultTypes(), kokkos::SingleLevel::PerTeam);
        auto singleBody = rewriter.createBlock(&single.getRegion());
        rewriter.setInsertionPointToStart(singleBody);
        auto singleWrappedOp = rewriter.clone(*op);
        rewriter.create<kokkos::YieldOp>(op->getLoc(),
                                         singleWrappedOp->getResults());
        rewriter.replaceOp(op, single);
      } else if (inThreadLoop(op) && !inVectorLoop(op)) {
        // op needs to be in PerThread single
        auto single = rewriter.create<kokkos::SingleOp>(
            op->getLoc(), op->getResultTypes(), kokkos::SingleLevel::PerThread);
        auto singleBody = rewriter.createBlock(&single.getRegion());
        rewriter.setInsertionPointToStart(singleBody);
        auto singleWrappedOp = rewriter.clone(*op);
        rewriter.create<kokkos::YieldOp>(op->getLoc(),
                                         singleWrappedOp->getResults());
        rewriter.replaceOp(op, single);
      }
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });
  return success();
}

// !! TODO: optimal team synchronization requires dataflow analysis within op's
// body to understand what specific memrefs might be in use at each point
// (reading or writing) by other threads executing a team-level loop or single.
//
// But for now, use a conservate version that inserts a barrier after each
// team-wide parallel op and single that does not already synchronize.
static LogicalResult insertTeamSynchronization(RewriterBase &rewriter,
                                               kokkos::TeamParallelOp op) {
  op->walk([&](kokkos::RangeParallelOp nestedOp) {
    auto level = nestedOp.getParallelLevel();
    bool needsPostBarrier = (nestedOp.getNumResults() == 0) &&
                            (level == kokkos::ParallelLevel::TeamThread ||
                             level == kokkos::ParallelLevel::TeamVector);
    if (needsPostBarrier) {
      rewriter.setInsertionPointAfter(nestedOp);
      rewriter.create<kokkos::TeamBarrierOp>(nestedOp.getLoc());
    }
  });
  op->walk([&](kokkos::SingleOp single) {
    bool needsPostBarrier = single.getNumResults() == 0 &&
                            single.getLevel() == kokkos::SingleLevel::PerTeam;
    if (needsPostBarrier) {
      rewriter.setInsertionPointAfter(single);
      rewriter.create<kokkos::TeamBarrierOp>(single.getLoc());
    }
  });
  return success();
}

// Recursively map the scf.parallel children of op.
static LogicalResult mapNestedLoopsImpl(RewriterBase &rewriter, Operation *op,
                                        int parLevelsRemaining) {
  // Make a list of the directly nested scf.parallel ops inside op, and their
  // respective nesting depths
  int opDepth = kokkos::getOpParallelDepth(op);
  LogicalResult result = success();
  op->walk([&](scf::ParallelOp child) {
    int childDepth = kokkos::getOpParallelDepth(child);
    if (childDepth == opDepth + 1) {
      int childNestingLevel = getParallelNumLevels(child);
      // Four cases for how child should be mapped:
      // parLevelsRemaining == 2 and childNestingLevel == 1: TeamVector
      // parLevelsRemaining == 2 and childNestingLevel > 1: TeamThread
      // parLevelsRemaining == 1 and childNestingLevel > 1: scf.for
      // parLevelsRemaining == 1 and childNestingLevel == 1: ThreadVector
      //
      // The children of child must be mapped first, but use the cases above
      // to determine if child will use a parallelism level
      bool childWillBeSequential =
          parLevelsRemaining == 1 && childNestingLevel > 1;
      auto result = mapNestedLoopsImpl(rewriter, child,
                                       parLevelsRemaining -
                                           (childWillBeSequential ? 0 : 1));
      // Now map child itself
      if (childWillBeSequential) {
        scf::ForOp newOp;
        result = scfParallelToSequential(newOp, rewriter, child);
        if (failed(result))
          return WalkResult::interrupt();
      } else {
        kokkos::ParallelLevel level;
        if (parLevelsRemaining == 2) {
          level = childNestingLevel == 1 ? kokkos::ParallelLevel::TeamVector
                                         : kokkos::ParallelLevel::TeamThread;
        } else {
          level = kokkos::ParallelLevel::ThreadVector;
        }
        kokkos::RangeParallelOp newOp;
        result = scfParallelToKokkosRange(
            newOp, rewriter, child, kokkos::ExecutionSpace::TeamHandle, level);
        if (failed(result))
          return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return result;
}

// Map all scf.parallel loops nested within a kokkos.team_parallel.
static LogicalResult mapTeamNestedLoops(RewriterBase &rewriter,
                                        kokkos::TeamParallelOp op) {
  return mapNestedLoopsImpl(rewriter, op, 2);
}

struct KokkosLoopRewriter : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  KokkosLoopRewriter(MLIRContext *context) : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(scf::ParallelOp op,
                                PatternRewriter &rewriter) const override {
    // Only match with top-level ParallelOps (meaning op is not enclosed in
    // another ParallelOp)
    if (op->getParentOfType<scf::ParallelOp>())
      return failure();
    // Determine the maximum depth of parallel nesting (a simple RangePolicy is
    // 1, etc.)
    int nestingLevel = getParallelNumLevels(op);
    // Now decide whether this op should execute on device (offloaded) or host.
    // Operations that are assumed to be host-only:
    // - func.call
    // - any memref allocation or deallocation
    // This is conservative as there are obviously functions that work safely on
    // the device, but an inlining pass could work around that easily
    bool canBeOffloaded = true;
    op->walk([&](Operation *childOp) -> WalkResult {
      if (isa<func::CallOp, memref::AllocOp, memref::AllocaOp,
              memref::DeallocOp, memref::ReallocOp>(childOp)) {
        canBeOffloaded = false;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    kokkos::ExecutionSpace exec = canBeOffloaded
                                      ? kokkos::ExecutionSpace::Device
                                      : kokkos::ExecutionSpace::Host;
    // Cases for exec == Device:
    //
    // - Depth 1:   RangePolicy/MDRangePolicy
    // - Depth 2:   TeamPolicy with one thread (simd) per inner work-item (best
    // for spmv-like patterns)
    //              TODO: Write a heuristic to choose TeamPolicy/TeamVector
    //              instead, for when the inner loop requires more parallelism?
    // - Depth >=3: Use TeamPolicy/TeamThread for outermost two loops, and
    // ThreadVector for innermost loop.
    //              Better coalescing that way, if data layout is correct for
    //              the loop structure. Serialize all other loops by replacing
    //              them with scf.for.
    //
    // For exec == Host, just parallelize the outermost loop with RangePolicy
    // and serialize any inner loops.
    if (exec == kokkos::ExecutionSpace::Host || nestingLevel == 1) {
      kokkos::RangeParallelOp newOp;
      LogicalResult result = scfParallelToKokkosRange(
          newOp, rewriter, op, exec, kokkos::ParallelLevel::RangePolicy);
      if (failed(result))
        return result;
      // NOTE: when walking a tree and replacing ops as they're found, the walk
      // must be PostOrder
      newOp->walk<WalkOrder::PostOrder>([&](scf::ParallelOp innerParOp) {
        scf::ForOp newForOp;
        result = scfParallelToSequential(newForOp, rewriter, innerParOp);
        if (failed(result))
          return WalkResult::interrupt();
        return WalkResult::skip();
      });
      if (failed(result))
        return result;
    } else if (nestingLevel == 2) {
      // Note: initially, zero for vector length hint mean that no hint is given
      Value zero = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
      kokkos::ThreadParallelOp newOp;
      LogicalResult result =
          scfParallelToKokkosThread(newOp, rewriter, op, zero);
      if (failed(result))
        return result;
      // since the nesting level is 2, we know all inner parallel loops should
      // be at ThreadVector level. There won't be any more scf.parallels nested
      // within those.
      newOp->walk<mlir::WalkOrder::PostOrder>([&](scf::ParallelOp innerParOp) {
        kokkos::RangeParallelOp newOp;
        result = scfParallelToKokkosRange(newOp, rewriter, innerParOp,
                                          kokkos::ExecutionSpace::TeamHandle,
                                          kokkos::ParallelLevel::ThreadVector);
        if (failed(result))
          return WalkResult::interrupt();
        return WalkResult::skip();
      });
      if (failed(result))
        return result;
      result = insertSingleWraps(rewriter, newOp);
      if (failed(result))
        return result;
      // Compute an estimate for the max parallelism at ThreadVector level, if
      // possible
      Value vectorLengthHint;
      if (!failed(estimateParallelism(vectorLengthHint, rewriter, newOp,
                                      kokkos::ParallelLevel::ThreadVector))) {
        rewriter.startOpModification(newOp);
        newOp.getVectorLengthHintMutable().assign(vectorLengthHint);
        rewriter.finalizeOpModification(newOp);
      }
    } else if (nestingLevel >= 3) {
      // Generate a team policy for the top level. This leaves 2 more levels of
      // parallelism to use. See mapTeamNestedLoops() above for how these are
      // used.
      //
      // Note: zero for team size hint and vector length hint mean that no hint
      // is given, and Kokkos::AUTO should be used instead
      Value zero = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
      kokkos::TeamParallelOp newOp;
      LogicalResult result =
          scfParallelToKokkosTeam(newOp, rewriter, op, zero, zero);
      if (failed(result))
        return result;
      result = mapTeamNestedLoops(rewriter, newOp);
      if (failed(result))
        return result;
      // The loop structure is now finalized; insert singles and team barriers
      // where needed
      result = insertSingleWraps(rewriter, newOp);
      if (failed(result))
        return result;
      result = insertTeamSynchronization(rewriter, newOp);
      if (failed(result))
        return result;
      // Compute an estimate for the max parallelism at TeamThread and
      // ThreadVector levels separately
      // TODO: also check TeamVector parallelism and combine those somehow
      Value teamSizeHint;
      if (!failed(estimateParallelism(teamSizeHint, rewriter, newOp,
                                      kokkos::ParallelLevel::TeamThread))) {
        rewriter.startOpModification(newOp);
        newOp.getTeamSizeHintMutable().assign(teamSizeHint);
        rewriter.finalizeOpModification(newOp);
      }
      Value vectorLengthHint;
      if (!failed(estimateParallelism(vectorLengthHint, rewriter, newOp,
                                      kokkos::ParallelLevel::ThreadVector))) {
        rewriter.startOpModification(newOp);
        newOp.getVectorLengthHintMutable().assign(vectorLengthHint);
        rewriter.finalizeOpModification(newOp);
      }
    }
    return success();
  }
};

} // namespace

void mlir::populateKokkosLoopMappingPatterns(RewritePatternSet &patterns) {
  patterns.add<KokkosLoopRewriter>(patterns.getContext());
}

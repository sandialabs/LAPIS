//===- DenseLinalgToParallelLoops.cpp
//     Lowering linalg.generic to scf.parallel, with reduction support -===//
//     This must run after sparsification as it works on dense
//     linalg.generic ops only.

#include "lapis/Dialect/Kokkos/IR/KokkosDialect.h"
#include "lapis/Dialect/Kokkos/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/Utils/AffineCanonicalizationUtils.h"

namespace mlir {
#define GEN_PASS_DEF_DENSELINALGTOPARALLELLOOPS
#include "lapis/Dialect/Kokkos/Transforms/Passes.h.inc"
}

using namespace mlir;
using LinalgLoops = SmallVector<Operation *, 4>;
using linalg::LinalgOp;

static bool isAdd(Operation* op) {
  return isa<arith::AddFOp, arith::AddIOp>(op);
}

static bool isAssociativeAndCommutativeArith(Operation* op) {
  return isa<
    arith::AddFOp, arith::AddIOp, arith::AndIOp,
    arith::MaximumFOp, arith::MaxSIOp, arith::MaxUIOp,
    arith::MinimumFOp, arith::MinSIOp, arith::MinUIOp,
    arith::MulFOp, arith::MulIOp, arith::OrIOp, arith::XOrIOp>(op);
}

static bool isParallelIterator(utils::IteratorType iteratorType) {
  return iteratorType == utils::IteratorType::parallel;
}

static int getNumUses(Value v) {
  int count = 0;
  for(auto it = v.use_begin(); it != v.use_end(); it++)
    count++;
  return count;
}

// Can op (including its reductions) be parallelized in the Kokkos model?
// Flow chart:
// - If there are no reductions, then yes.
// - If there are multiple outputs, then no (possible in theory but pipeline/emitter need more work)
// - If the output scalar argument has at exactly one usage inside the body,
//   and that usage is to an arith op, then call that arith op is the 'join'.
// - If the join is non-commutative or non-associative, then no.
// - If the join is an add, then yes (sum-reduction is supported by all Kokkos policies)
// - If there is at least one parallel iterator AND more than one reduction iterator, then no.
//   This is because nested MDRange policies do not support reductions other than add
//   (for example, see https://kokkos.org/kokkos-core-wiki/API/core/policies/TeamVectorMDRange.html#restrictions).
// Otherwise, yes.
static bool canBeParallelized(LinalgOp op) {
  // if op has no reduction iterators, then it can't have complex reductions
  int numParallelIters = 0;
  int numReductionIters = 0;
  {
    auto iteratorTypes = op.getIteratorTypesArray();
    for(auto it : iteratorTypes) {
      if(isParallelIterator(it))
        numParallelIters++;
      else
        numReductionIters++;
    }
  }
  if(numReductionIters == 0)
    return true;
  // Note: after bufferization, linalg ops have zero results since their output is stored into a memref.
  // So count the output tensors based on how many are DPS buffers.
  // For now, don't try to handle multiple results in ops with at least one reduction.
  int numOutputs = op.getDpsInitsMutable().size();
  if(numOutputs != 1) return false;
  // The last body block argument is the scalar from output tensor (we have checked that there is exactly one)
  Block& body = op->getRegion(0).front();
  Value outputArg = *body.args_rbegin();
  int numUses = getNumUses(outputArg);
  if(numUses != 1) {
    return false;
  }
  // Arg is used once, but check if it's in a simple reduction that we know how to parallelize
  // and which Kokkos supports.
  Operation* join = outputArg.use_begin()->getOwner();
  // outputUsage should be the 'join' operation. Make sure its result is directly yielded by the body,
  // and has no other uses.
  auto joinedValue = join->getResult(0);
  if(getNumUses(joinedValue) != 1 || !isa<linalg::YieldOp>(joinedValue.use_begin()->getOwner()))
    return false;
  if(numParallelIters > 0 && numReductionIters > 1)
    return isAdd(join);
  return isAssociativeAndCommutativeArith(join);
}

static void unpackRanges(PatternRewriter &rewriter, Location loc,
                         ArrayRef<Range> ranges, SmallVectorImpl<Value> &lbs,
                         SmallVectorImpl<Value> &ubs,
                         SmallVectorImpl<Value> &steps) {
  for (Range range : ranges) {
    lbs.emplace_back(
        getValueOrCreateConstantIndexOp(rewriter, loc, range.offset));
    ubs.emplace_back(getValueOrCreateConstantIndexOp(rewriter, loc, range.size));
    steps.emplace_back(
        getValueOrCreateConstantIndexOp(rewriter, loc, range.stride));
  }
}

static void inlineRegionAndEmitStore(PatternRewriter &b, Location loc, LinalgOp linalgOp,
                                     ArrayRef<Value> indexedValues,
                                     ArrayRef<SmallVector<Value>> indexing,
                                     ArrayRef<Value> outputBuffers) {
  auto &block = linalgOp->getRegion(0).front();
  IRMapping map;
  map.map(block.getArguments(), indexedValues);
  for (auto &op : block.without_terminator()) {
    auto *newOp = b.clone(op, map);
    map.map(op.getResults(), newOp->getResults());
  }

  Operation *terminator = block.getTerminator();
  for (OpOperand &operand : terminator->getOpOperands()) {
    Value toStore = map.lookupOrDefault(operand.get());
    b.create<memref::StoreOp>(loc, toStore, outputBuffers[operand.getOperandNumber()],
                        indexing[operand.getOperandNumber()]);
  }
}

static SmallVector<Value> makeCanonicalAffineApplies(PatternRewriter &b, Location loc,
                                                     AffineMap map,
                                                     ArrayRef<Value> vals) {
  if (map.isEmpty())
    return {};

  assert(map.getNumInputs() == vals.size());
  SmallVector<Value> res;
  res.reserve(map.getNumResults());
  auto dims = map.getNumDims();
  for (auto e : map.getResults()) {
    auto exprMap = AffineMap::get(dims, map.getNumSymbols(), e);
    SmallVector<Value> operands(vals);
    affine::canonicalizeMapAndOperands(&exprMap, &operands);
    res.push_back(b.create<affine::AffineApplyOp>(loc, exprMap, operands));
  }
  return res;
}

// Emit loop body which contributes to a reduction.
// This assumes:
// a) linalgOp produces a single result tensor
// b) the actual reduction is perfomed in a single arith op (whose result is yielded by the body terminator)
static void emitReductionScalarImplementation(
    PatternRewriter &rewriter, Location loc,
    ValueRange allIvsRange,
    LinalgOp linalgOp) {
  assert(linalgOp.hasPureBufferSemantics() &&
         "expected linalg op with buffer semantics");
  // Generate loads
  SmallVector<Value> inputValues;
  SmallVector<Value> allIvs(allIvsRange);

  // 1.a. Emit load from input operand or for scalars access the operand itself.
  for (OpOperand *inputOperand : linalgOp.getDpsInputOperands()) {
    if (linalgOp.isScalar(inputOperand)) {
      inputValues.push_back(inputOperand->get());
      continue;
    }
    auto indexing = makeCanonicalAffineApplies(
        rewriter, loc, linalgOp.getMatchingIndexingMap(inputOperand), allIvs);
    inputValues.push_back(
        rewriter.create<memref::LoadOp>(loc, inputOperand->get(), indexing));
  }

  auto &body = linalgOp->getRegion(0).front();
  // Check preconditions of the rewrite logic
  assert(linalgOp.getDpsInitsMutable().size() == 1);
  assert(isa<linalg::YieldOp>(body.getTerminator()));
  Value yielded = body.getTerminator()->getOperand(0);
  // We have only one result, so the final body argument must be the incoming partial reduction
  Value partialReduction = body.getArguments().back();
  // Get the arithmetic operation that performs the reduction join
  Operation* join = yielded.getDefiningOp();
  assert(join->getNumOperands() == 2);
  Value reductionUpdate;
  // The order of join's operands is not specified (one is the partial reduction, so the other must be the update)
  if(join->getOperand(0) == partialReduction)
    reductionUpdate = join->getOperand(1);
  else
    reductionUpdate = join->getOperand(0);

  IRMapping map;
  map.map(body.getArguments(), inputValues);
  // Clone all ops (except join and yield) into the new body
  for (Operation& op : body.without_terminator()) {
    if(&op != join)
      rewriter.clone(op, map);
  }
  // Then create an scf.reduce block, which takes the reduction update as the operand.
  // The scf.reduce serves as the terminator of the new loop
  auto redOp = rewriter.create<scf::ReduceOp>(loc, map.lookupOrDefault(reductionUpdate));
  // Attach to the reduction op.
  Block *redBlock = &redOp.getReductions().front().front();
  auto insertPt = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToEnd(redBlock);
  Operation* newJoin = rewriter.clone(*join);
  // Replaces arguments of the reduction expression by using the block
  // arguments from scf.reduce.
  rewriter.modifyOpInPlace(
      newJoin, [&]() { newJoin->setOperands(redBlock->getArguments()); });
  rewriter.create<scf::ReduceReturnOp>(loc, newJoin->getResult(0));
  rewriter.restoreInsertionPoint(insertPt);
}

// Emit non-reduction loop body
static void emitScalarImplementation(PatternRewriter &b, Location loc,
                                     ValueRange allIvs,
                                     LinalgOp linalgOp) {
  assert(linalgOp.hasPureBufferSemantics() &&
         "expected linalg op with buffer semantics");
  SmallVector<Value> indexedValues;
  indexedValues.reserve(linalgOp->getNumOperands());

  auto allIvsPlusDims = SmallVector<Value>(allIvs);

  // TODO: Avoid the loads if the corresponding argument of the
  // region has no uses.
  // 1.a. Emit load from input operand or for scalars access the operand itself.
  for (OpOperand *inputOperand : linalgOp.getDpsInputOperands()) {
    if (linalgOp.isScalar(inputOperand)) {
      indexedValues.push_back(inputOperand->get());
      continue;
    }
    auto indexing = makeCanonicalAffineApplies(
        b, loc, linalgOp.getMatchingIndexingMap(inputOperand), allIvsPlusDims);
    indexedValues.push_back(
        b.create<memref::LoadOp>(loc, inputOperand->get(), indexing));
  }
  // 1.b. Emit load from output views.
  for (OpOperand &outputOperand : linalgOp.getDpsInitsMutable()) {
    SmallVector<Value> indexing = makeCanonicalAffineApplies(
        b, loc, linalgOp.getMatchingIndexingMap(&outputOperand),
        allIvsPlusDims);
    indexedValues.push_back(
        b.create<memref::LoadOp>(loc, outputOperand.get(), indexing));
  }

  // 2. Inline region, currently only works for a single basic block.
  // 3. Emit store.
  SmallVector<SmallVector<Value>, 8> indexing;
  SmallVector<Value> outputBuffers;
  for (OpOperand &outputOperand : linalgOp.getDpsInitsMutable()) {
    if (!isa<MemRefType>(outputOperand.get().getType()))
      continue;
    indexing.push_back(makeCanonicalAffineApplies(
        b, loc, linalgOp.getMatchingIndexingMap(&outputOperand),
        allIvsPlusDims));
    outputBuffers.push_back(outputOperand.get());
  }
  inlineRegionAndEmitStore(b, loc, linalgOp, indexedValues, indexing, outputBuffers);
}

static void generateParallelLoopNest(
    PatternRewriter &rewriter, Location loc, ValueRange lbs, ValueRange ubs,
    ValueRange steps, ArrayRef<utils::IteratorType> iteratorTypes, LinalgOp linalgOp) {
  size_t n = lbs.size();
  assert(n == ubs.size());
  assert(n == steps.size());
  assert(n == iteratorTypes.size());

  // Partition the iterators into parallels and reductions
  SmallVector<int> parallelIters;
  SmallVector<int> reductionIters;
  for (auto [iterIndex, iterType] : llvm::enumerate(iteratorTypes)) {
    if(isParallelIterator(iterType))
      parallelIters.push_back(iterIndex);
    else
      reductionIters.push_back(iterIndex);
  }
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value> allIVs(n, zero);
  // Parallel iters define outer loop, and reduction iters define inner
  SmallVector<Value> outerLB, innerLB, outerUB, innerUB, outerStep, innerStep;
  for(auto i : parallelIters) {
    outerLB.push_back(lbs[i]);
    outerUB.push_back(ubs[i]);
    outerStep.push_back(steps[i]);
  }
  for(auto i : reductionIters) {
    innerLB.push_back(lbs[i]);
    innerUB.push_back(ubs[i]);
    innerStep.push_back(steps[i]);
  }
  // 3 cases for building loops:
  // - mixed parallel and reduction dimensions
  // - all parallel
  // - all reductions
  if(parallelIters.size() && reductionIters.size()) {
    // Create two scf.parallel ops: outer over parallels, and inner over reductions.
    // Write-back of results will go into the outer loop only.
    rewriter.create<scf::ParallelOp>(
        loc, outerLB, outerUB, outerStep,
        [&](OpBuilder& nestedRewriter1, Location, ValueRange outerIVs) {
          // insert outer IVs into overall list of IVs
          for(auto [i, iv] : llvm::enumerate(outerIVs)) {
            allIVs[parallelIters[i]] = iv;
          }
          // Load the initial reduction values based on the parallel IVs we have
          SmallVector<Value> outputInitValues;
          for (OpOperand &outputOperand : linalgOp.getDpsInitsMutable()) {
            SmallVector<Value> indexing = makeCanonicalAffineApplies(
                rewriter, loc, linalgOp.getMatchingIndexingMap(&outputOperand),
                allIVs);
            outputInitValues.push_back(
                rewriter.create<memref::LoadOp>(loc, outputOperand.get(), indexing));
          }
          auto allIVsDummyInner = allIVs;
          // The body of the outer loop is another parallel which performs the reduction(s).
          auto innerLoop = nestedRewriter1.create<scf::ParallelOp>(
              loc, innerLB, innerUB, innerStep, outputInitValues,
              [&](OpBuilder& nestedRewriter2, Location innerLoc, ValueRange innerIVs, ValueRange /* unused */) {
                for(auto [i, iv] : llvm::enumerate(innerIVs)) {
                  allIVs[reductionIters[i]] = iv;
                }
                // Build the inner body using all the induction variables created so far
                emitReductionScalarImplementation((PatternRewriter&) nestedRewriter2, innerLoc, allIVs, linalgOp);
              });
          // Write reduction results back
          int counter = 0;
          for (auto& outputOperand : linalgOp.getDpsInitsMutable()) {
            SmallVector<Value> indexing = makeCanonicalAffineApplies(
                rewriter, loc, linalgOp.getMatchingIndexingMap(&outputOperand),
                allIVsDummyInner);
            rewriter.create<memref::StoreOp>(loc, innerLoop->getResult(counter++), outputOperand.get(), indexing);
          }
        });
  }
  else if(reductionIters.size()) {
    // Emit load from the output tensor to provide the initial value for reduction
    SmallVector<Value> outputInitValues;
    // If there are only reduction dims, then the output indexing cannot depend on any induction vars.
    // We are outside any loop and have no induction vars anyway.
    // This is why allIVs is populated with dummy zeros initially.
    for (OpOperand &outputOperand : linalgOp.getDpsInitsMutable()) {
      SmallVector<Value> indexing = makeCanonicalAffineApplies(
          rewriter, loc, linalgOp.getMatchingIndexingMap(&outputOperand),
          allIVs);
      outputInitValues.push_back(
          rewriter.create<memref::LoadOp>(loc, outputOperand.get(), indexing));
    }
    // The body of the outer loop is another parallel which performs the reduction(s).
    auto innerLoop = rewriter.create<scf::ParallelOp>(
        loc, innerLB, innerUB, innerStep, outputInitValues,
        [&](OpBuilder& nestedRewriter, Location innerLoc, ValueRange innerIVs, ValueRange /* unused */) {
          for(auto [i, iv] : llvm::enumerate(innerIVs)) {
            allIVs[reductionIters[i]] = iv;
          }
          // Build the inner body using all the induction variables created so far
          emitReductionScalarImplementation((PatternRewriter&) nestedRewriter, innerLoc, allIVs, linalgOp);
        });
    // Write reduction results back
    int counter = 0;
    for (auto& outputOperand : linalgOp.getDpsInitsMutable()) {
      SmallVector<Value> indexing = makeCanonicalAffineApplies(
          rewriter, loc, linalgOp.getMatchingIndexingMap(&outputOperand),
          allIVs);
      rewriter.create<memref::StoreOp>(loc, innerLoop->getResult(counter++), outputOperand.get(), indexing);
    }
  }
  else {
    // There are no reductions, so generate just one (outer) parallel
    rewriter.create<scf::ParallelOp>(
        loc, outerLB, outerUB, outerStep,
        [&](OpBuilder& nestedRewriter, Location loc, ValueRange outerIVs) {
          // insert outer IVs into overall list of IVs
          for(auto [i, iv] : llvm::enumerate(outerIVs)) {
            allIVs[parallelIters[i]] = iv;
          }
          // Inline the original body into new loop
          emitScalarImplementation((PatternRewriter&) nestedRewriter, loc, allIVs, linalgOp);
        });
  }
}

static LogicalResult linalgOpToParallel(PatternRewriter &rewriter, LinalgOp linalgOp) {
  // The flattened loopToOperandRangesMaps is expected to be an invertible
  // permutation map (which is asserted in the inverse calculation).
  assert(linalgOp.hasPureBufferSemantics() &&
         "expected linalg op with buffer semantics");

  Operation* op = linalgOp;

  auto loopRanges = linalgOp.createLoopRanges(rewriter, linalgOp.getLoc());
  auto iteratorTypes = linalgOp.getIteratorTypesArray();

  SmallVector<Value> allIvs;
  auto loc = linalgOp.getLoc();
  SmallVector<Value> iterArgInitValues;
  if (!linalgOp.hasPureBufferSemantics())
    llvm::append_range(iterArgInitValues, linalgOp.getDpsInits());
  assert(iterArgInitValues.empty() && "unexpected ParallelOp init values");
  // This function may be passed more iterator types than ranges.
  assert(iteratorTypes.size() >= loopRanges.size() &&
         "expected iterator type for all ranges");
  //iteratorTypes = iteratorTypes.take_front(loopRanges.size());
  SmallVector<Value> lbs, ubs, steps;

  // Get the loop lb, ub, and step.
  unpackRanges(rewriter, loc, loopRanges, lbs, ubs, steps);

  generateParallelLoopNest(rewriter, loc, lbs, ubs, steps, iteratorTypes, linalgOp);

  rewriter.eraseOp(op);
  return success();
}

// Pattern to rewrite a bufferized linalg.generic op
// as an scf.parallel loop nest.
struct LinalgToParallelPattern : public RewritePattern {

  LinalgToParallelPattern(MLIRContext *context) : RewritePattern(MatchAnyOpTypeTag(), 1, context) {}

  LogicalResult matchAndRewrite(Operation* op, PatternRewriter &rewriter) const override {
    auto linalgOp = dyn_cast<LinalgOp>(op);
    if (!isa<LinalgOp>(op) || !linalgOp.hasPureBufferSemantics()) {
      return rewriter.notifyMatchFailure(op, "expected linalg op with buffer semantics");
    }
    if (!canBeParallelized(linalgOp)) {
      // If we fail to match on this pass, op will instead get lowered by --convert-linalg-to-parallel-loops
      // (which will not attempt to parallelize reductions)
      return rewriter.notifyMatchFailure(op, "pass can only lower linalg ops with at most 1 output and reduction supported by Kokkos");
    }
    return linalgOpToParallel(rewriter, linalgOp);
  }
};

struct DenseLinalgToParallelLoopsPass
    : public impl::DenseLinalgToParallelLoopsBase<DenseLinalgToParallelLoopsPass> {

  DenseLinalgToParallelLoopsPass() = default;
  DenseLinalgToParallelLoopsPass(const DenseLinalgToParallelLoopsPass& pass) = default;

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<LinalgToParallelPattern>(patterns.getContext());
    (void) applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

std::unique_ptr<Pass> mlir::createDenseLinalgToParallelLoopsPass()
{
  return std::make_unique<DenseLinalgToParallelLoopsPass>();
}


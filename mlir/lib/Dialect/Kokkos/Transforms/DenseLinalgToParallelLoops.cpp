//===- DenseLinalgToParallelLoops.cpp
//     Lowering linalg.generic to scf.parallel, with reduction support -===//
//     This must run after sparsification as it works on dense
//     linalg.generic ops only.

#include "lapis/Dialect/Kokkos/IR/KokkosDialect.h"
#include "lapis/Dialect/Kokkos/Transforms/Passes.h"
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

static bool isParallelIterator(utils::IteratorType iteratorType) {
  return iteratorType == utils::IteratorType::parallel;
}

/*
static bool isReductionIterator(utils::IteratorType iteratorType) {
  return iteratorType == utils::IteratorType::reduction;
}
*/

static void unpackRanges(OpBuilder &builder, Location loc,
                         ArrayRef<Range> ranges, SmallVectorImpl<Value> &lbs,
                         SmallVectorImpl<Value> &ubs,
                         SmallVectorImpl<Value> &steps) {
  for (Range range : ranges) {
    lbs.emplace_back(
        getValueOrCreateConstantIndexOp(builder, loc, range.offset));
    ubs.emplace_back(getValueOrCreateConstantIndexOp(builder, loc, range.size));
    steps.emplace_back(
        getValueOrCreateConstantIndexOp(builder, loc, range.stride));
  }
}

static void inlineRegionAndEmitStore(OpBuilder &b, Location loc, linalg::GenericOp op,
                                     ArrayRef<Value> indexedValues,
                                     ArrayRef<SmallVector<Value>> indexing,
                                     ArrayRef<Value> outputBuffers) {
  auto &block = op->getRegion(0).front();
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

static SmallVector<Value> makeCanonicalAffineApplies(OpBuilder &b, Location loc,
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

static void emitScalarImplementation(OpBuilder &b, Location loc,
                                     ArrayRef<Value> allIvs,
                                     linalg::GenericOp linalgOp) {
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

  // TODO: When a region inliner exists, use it.
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
    OpBuilder &b, Location loc, ValueRange lbs, ValueRange ubs,
    ValueRange steps, ArrayRef<utils::IteratorType> iteratorTypes,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuilderFn,
    SmallVectorImpl<Value> &ivStorage) {
  assert(lbs.size() == ubs.size());
  assert(lbs.size() == steps.size());
  assert(lbs.size() == iteratorTypes.size());

  // If there are no (more) loops to be generated, generate the body and be
  // done with it.
  if (iteratorTypes.empty()) {
    bodyBuilderFn(b, loc, ivStorage);
    return;
  }

  // If there are no outer parallel loops, generate one sequential loop and
  // recurse.
  if (!isParallelIterator(iteratorTypes.front())) {
    scf::LoopNest singleLoop = scf::buildLoopNest(
        b, loc, lbs.take_front(), ubs.take_front(), steps.take_front(),
        [&](OpBuilder &b, Location loc, ValueRange ivs) {
          ivStorage.append(ivs.begin(), ivs.end());
          generateParallelLoopNest(
              b, loc, lbs.drop_front(), ubs.drop_front(), steps.drop_front(),
              iteratorTypes.drop_front(),
              bodyBuilderFn, ivStorage);
        });
    return;
  }

  unsigned nLoops = iteratorTypes.size();
  unsigned numProcessed = 0;
  numProcessed = nLoops - iteratorTypes.drop_while(isParallelIterator).size();

  // Generate a single parallel loop-nest operation for all outermost
  // parallel loops and recurse.
  b.create<scf::ParallelOp>(
      loc, lbs.take_front(numProcessed), ubs.take_front(numProcessed),
      steps.take_front(numProcessed),
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange localIvs) {
        ivStorage.append(localIvs.begin(), localIvs.end());
        generateParallelLoopNest(
            nestedBuilder, nestedLoc, lbs.drop_front(numProcessed),
            ubs.drop_front(numProcessed), steps.drop_front(numProcessed),
            iteratorTypes.drop_front(numProcessed),
            bodyBuilderFn, ivStorage);
      });
}

static void generateLoopNest(
    OpBuilder &b, Location loc, ArrayRef<Range> loopRanges, linalg::GenericOp linalgOp,
    ArrayRef<utils::IteratorType> iteratorTypes,
    function_ref<scf::ValueVector(OpBuilder &, Location, ValueRange,
                                  ValueRange)>
        bodyBuilderFn) {
  SmallVector<Value> iterArgInitValues;
  if (!linalgOp.hasPureBufferSemantics())
    llvm::append_range(iterArgInitValues, linalgOp.getDpsInits());
  assert(iterArgInitValues.empty() && "unexpected ParallelOp init values");
  // This function may be passed more iterator types than ranges.
  assert(iteratorTypes.size() >= loopRanges.size() &&
         "expected iterator type for all ranges");
  iteratorTypes = iteratorTypes.take_front(loopRanges.size());
  SmallVector<Value, 8> lbsStorage, ubsStorage, stepsStorage, ivs;
  unsigned numLoops = iteratorTypes.size();
  ivs.reserve(numLoops);
  lbsStorage.reserve(numLoops);
  ubsStorage.reserve(numLoops);
  stepsStorage.reserve(numLoops);

  // Get the loop lb, ub, and step.
  unpackRanges(b, loc, loopRanges, lbsStorage, ubsStorage, stepsStorage);

  ValueRange lbs(lbsStorage), ubs(ubsStorage), steps(stepsStorage);
  generateParallelLoopNest(
      b, loc, lbs, ubs, steps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange ivs) {
        bodyBuilderFn(b, loc, ivs, linalgOp->getOperands());
      },
      ivs);

  assert(ivs.size() == iteratorTypes.size() && "did not generate enough loops");
}

static void replaceIndexOpsByInductionVariables(RewriterBase &rewriter,
                                                linalg::GenericOp linalgOp,
                                                ArrayRef<Operation *> loopOps) {
  // Extract the induction variables of the loop nest from outer to inner.
  SmallVector<Value> allIvs;
  for (Operation *loopOp : loopOps) {
    if(auto parallelOp = dyn_cast<scf::ParallelOp>(loopOp))
      allIvs.append(parallelOp.getInductionVars());
    else if(auto forOp = dyn_cast<scf::ForOp>(loopOp))
      allIvs.push_back(forOp.getInductionVar());
    else
      assert(false && "unexpected op");
  }
  assert(linalgOp.getNumLoops() == allIvs.size() &&
         "expected the number of loops and induction variables to match");
  // Replace the index operations in the body of the innermost loop op.
  if (!loopOps.empty()) {
    auto loopOp = cast<LoopLikeOpInterface>(loopOps.back());
    for (Region *r : loopOp.getLoopRegions())
      for (linalg::IndexOp indexOp : llvm::make_early_inc_range(r->getOps<linalg::IndexOp>()))
        rewriter.replaceOp(indexOp, allIvs[indexOp.getDim()]);
  }
}

static LogicalResult genericToParallel(RewriterBase &rewriter, linalg::GenericOp linalgOp) {
  // The flattened loopToOperandRangesMaps is expected to be an invertible
  // permutation map (which is asserted in the inverse calculation).
  assert(linalgOp.hasPureBufferSemantics() &&
         "expected linalg op with buffer semantics");

  Operation* op = linalgOp;
  auto loopRanges = cast<linalg::LinalgOp>(op).createLoopRanges(rewriter, linalgOp.getLoc());
  auto iteratorTypes = linalgOp.getIteratorTypesArray();

  SmallVector<Value> allIvs;
  generateLoopNest(rewriter, linalgOp.getLoc(), loopRanges, linalgOp, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange ivs,
          ValueRange operandValuesToUse) -> scf::ValueVector {
        assert(operandValuesToUse == linalgOp->getOperands() &&
               "expect operands are captured and not passed by loop argument");
        allIvs.append(ivs.begin(), ivs.end());
        emitScalarImplementation(b, loc, allIvs, linalgOp);
        return scf::ValueVector{};
      });
  // Number of loop ops might be different from the number of ivs since some
  // loops like affine.parallel and scf.parallel have multiple ivs.
  SetVector<Operation *> loopSet;
  for (Value iv : allIvs) {
    if (!iv)
      return failure();
    // The induction variable is a block argument of the entry block of the
    // loop operation.
    BlockArgument ivVal = dyn_cast<BlockArgument>(iv);
    if (!ivVal)
      return failure();
    loopSet.insert(ivVal.getOwner()->getParentOp());
  }
  LinalgLoops loops(loopSet.begin(), loopSet.end());
  // Replace all index operations in the loop body.
  replaceIndexOpsByInductionVariables(rewriter, linalgOp, loops);
  rewriter.eraseOp(op);
  return success();
}

// Pattern to rewrite a bufferized linalg.generic op
// as an scf.parallel loop nest.
struct LinalgToParallelPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LinalgToParallelPattern(MLIRContext *context) : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(linalg::GenericOp op, PatternRewriter &rewriter) const override {
    if (!op.hasPureBufferSemantics()) {
      return rewriter.notifyMatchFailure(op, "expected linalg op with buffer semantics");
    }
    return genericToParallel(rewriter, op);
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


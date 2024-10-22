// ===- PartTensorDialect.cpp - part_tensor dialect implementation -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.  //
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lapis/Dialect/Kokkos/IR/KokkosDialect.h"
#include <utility>

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#include "lapis/Dialect/Kokkos/IR/KokkosEnums.cpp.inc"

// #define GET_ATTRDEF_CLASSES
// #include "lapis/Dialect/Kokkos/IR/KokkosAttrDefs.cpp.inc"

using namespace mlir;
using namespace mlir::kokkos;

void KokkosDialect::initialize() {
  //  addAttributes<
  // #define GET_ATTRDEF_LIST
  // #include "lapis/Dialect/Kokkos/IR/KokkosAttrDefs.cpp.inc"
  //      >();
  addOperations<
#define GET_OP_LIST
#include "lapis/Dialect/Kokkos/IR/Kokkos.cpp.inc"
      >();
}

template <typename TerminatorTy>
static TerminatorTy verifyAndGetTerminator(Operation *op, Region &region,
                                           StringRef errorMessage) {
  Operation *terminatorOperation = nullptr;
  if (!region.empty() && !region.front().empty()) {
    terminatorOperation = &region.front().back();
    if (auto yield = dyn_cast_or_null<TerminatorTy>(terminatorOperation))
      return yield;
  }
  auto diag = op->emitOpError(errorMessage);
  if (terminatorOperation)
    diag.attachNote(terminatorOperation->getLoc()) << "terminator here";
  return nullptr;
}

// ****************** //
//   RangeParallelOp   //
// ****************** //

void RangeParallelOp::build(OpBuilder &builder, OperationState &result,
                            ::mlir::kokkos::ExecutionSpace executionSpace,
                            ::mlir::kokkos::ParallelLevel parallelLevel,
                            ValueRange upperBounds, TypeRange resultTypes) {
  result.addOperands(upperBounds);
  result.addAttribute(
      "executionSpace",
      ExecutionSpaceAttr::get(builder.getContext(), executionSpace));
  result.addAttribute(
      "parallelLevel",
      ParallelLevelAttr::get(builder.getContext(), parallelLevel));
  result.addTypes(resultTypes);

  OpBuilder::InsertionGuard guard(builder);
  unsigned numIVs = upperBounds.size();
  SmallVector<Type, 8> argTypes(numIVs, builder.getIndexType());
  SmallVector<Location, 8> argLocs(numIVs, result.location);
  Region *bodyRegion = result.addRegion();
  builder.createBlock(bodyRegion, {}, argTypes, argLocs);
  RangeParallelOp::ensureTerminator(*bodyRegion, builder, result.location);
}

SmallVector<Region *> RangeParallelOp::getLoopRegions() {
  return SmallVector<Region *>(1, &getRegion());
}

ParseResult RangeParallelOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  auto &builder = parser.getBuilder();
  // Parse an opening `(` followed by induction variables followed by `)`
  SmallVector<OpAsmParser::Argument, 4> ivs;
  if (parser.parseArgumentList(ivs, OpAsmParser::Delimiter::Paren))
    return failure();

  // Parse loop bounds.
  SmallVector<OpAsmParser::UnresolvedOperand, 4> upper;
  if (parser.parseArrow() ||
      parser.parseOperandList(upper, ivs.size(),
                              OpAsmParser::Delimiter::Paren) ||
      parser.resolveOperands(upper, builder.getIndexType(), result.operands))
    return failure();

  // Parse optional results in case there is a reduce.
  if (parser.parseOptionalArrowTypeList(result.types))
    return failure();

  // Now parse the body.
  Region *body = result.addRegion();
  for (auto &iv : ivs)
    iv.type = builder.getIndexType();
  if (parser.parseRegion(*body, ivs))
    return failure();

  // Parse attributes.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Add a terminator if none was parsed.
  mlir::kokkos::RangeParallelOp::ensureTerminator(*body, builder,
                                                  result.location);
  return success();
}

void RangeParallelOp::print(OpAsmPrinter &p) {
  p << " (" << getBody()->getArguments() << ") -> (" << getUpperBound() << ")";
  p.printOptionalArrowTypeList(getResultTypes());
  p << ' ';
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict((*this)->getAttrs());
}

void RangeParallelOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  // Both the operation itself and the region may be branching into the body or
  // back into the operation itself. It is possible for loop not to enter the
  // body.
  regions.push_back(RegionSuccessor(&getRegion()));
  regions.push_back(RegionSuccessor());
}

LogicalResult RangeParallelOp::verify() {
  // Check that there is at least one value in upperBound.
  if (getUpperBound().empty())
    return emitOpError("needs at least one tuple element for upperBound");
  auto loopDim = getUpperBound().size();

  // Check that the body defines the same number of block arguments as there
  // are upper bounds.
  Block *body = getBody();
  if (body->getNumArguments() != loopDim)
    return emitOpError() << "expects the same number of induction variables: "
                         << body->getNumArguments()
                         << " as bounds: " << loopDim;
  for (auto arg : body->getArguments())
    if (!arg.getType().isIndex())
      return emitOpError(
          "expects arguments for the induction variable to be of index type");

  // Check that the yield has no results
  auto yield = verifyAndGetTerminator<kokkos::YieldOp>(
      *this, getRegion(), "expects body to terminate with 'kokkos.yield'");
  if (!yield)
    return failure();
  if (yield->getNumOperands() != 0)
    return yield.emitOpError() << "not allowed to have operands inside '"
                               << RangeParallelOp::getOperationName() << "'";

  // Check that the number of results is the same as the number of
  // UpdateReductionOps. Reductions can appear in 2 places: either directly as a
  // child of body, or in a single. If in a single, the single must be a direct
  // child of body.
  SmallVector<kokkos::UpdateReductionOp, 4> reductions;
  for (auto reduce : body->getOps<kokkos::UpdateReductionOp>()) {
    reductions.push_back(reduce);
  }
  for (auto single : body->getOps<kokkos::SingleOp>()) {
    for (auto reduce :
         single.getRegion().front().getOps<kokkos::UpdateReductionOp>()) {
      reductions.push_back(reduce);
    }
  }
  auto resultsSize = getResults().size();
  if (resultsSize != reductions.size())
    return emitOpError() << "expects number of results: " << resultsSize
                         << " to be the same as number of reductions: "
                         << reductions.size();
  // Check that the types of the results and reductions are the same.
  for (auto resultAndReduce : llvm::zip(getResults(), reductions)) {
    auto resultType = std::get<0>(resultAndReduce).getType();
    auto reduceOp = std::get<1>(resultAndReduce);
    auto reduceType = reduceOp.getUpdate().getType();
    if (resultType != reduceType)
      return reduceOp.emitOpError()
             << "expects type of reduce: " << reduceType
             << " to be the same as result type: " << resultType;
  }
  return success();
}

kokkos::UpdateReductionOp RangeParallelOp::getReduction() {
  Region &body = this->getLoopBody();
  for (kokkos::UpdateReductionOp op : body.getOps<kokkos::UpdateReductionOp>())
    return op;
  for (kokkos::SingleOp single : body.getOps<kokkos::SingleOp>()) {
    for (kokkos::UpdateReductionOp op :
         single.getRegion().getOps<kokkos::UpdateReductionOp>())
      return op;
  }
  return nullptr;
}

// ****************** //
//   TeamParallelOp   //
// ****************** //

void TeamParallelOp::build(OpBuilder &builder, OperationState &result,
                           Value leagueSize, Value teamSizeHint,
                           Value vectorLengthHint, TypeRange resultTypes) {
  result.addOperands(leagueSize);
  result.addOperands(teamSizeHint);
  result.addOperands(vectorLengthHint);
  result.addTypes(resultTypes);

  OpBuilder::InsertionGuard guard(builder);
  SmallVector<Type, 8> argTypes(4, builder.getIndexType());
  SmallVector<Location, 8> argLocs(4, result.location);
  Region *bodyRegion = result.addRegion();
  builder.createBlock(bodyRegion, {}, argTypes, argLocs);
  TeamParallelOp::ensureTerminator(*bodyRegion, builder, result.location);
}

SmallVector<Region *> TeamParallelOp::getLoopRegions() {
  return SmallVector<Region *>(1, &getRegion());
}

/*
ParseResult TeamParallelOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  // Parse an opening `(` followed by induction variables followed by `)`
  SmallVector<OpAsmParser::Argument, 4> ivs;
  if (parser.parseArgumentList(ivs, OpAsmParser::Delimiter::Paren))
    return failure();

  // Parse loop bounds.
  SmallVector<OpAsmParser::UnresolvedOperand, 4> upper;
  if (parser.parseArrow() ||
      parser.parseOperandList(upper, ivs.size(),
                              OpAsmParser::Delimiter::Paren) ||
      parser.resolveOperands(upper, builder.getIndexType(), result.operands))
    return failure();

  // Parse init values.
  SmallVector<OpAsmParser::UnresolvedOperand, 4> initVals;
  if (succeeded(parser.parseOptionalKeyword("init"))) {
    if (parser.parseOperandList(initVals, OpAsmParser::Delimiter::Paren))
      return failure();
  }

  // Parse optional results in case there is a reduce.
  if (parser.parseOptionalArrowTypeList(result.types))
    return failure();

  // Now parse the body.
  Region *body = result.addRegion();
  for (auto &iv : ivs)
    iv.type = builder.getIndexType();
  if (parser.parseRegion(*body, ivs))
    return failure();

  // Set `operandSegmentSizes` attribute.
  result.addAttribute(
      RangeParallelOp::getOperandSegmentSizeAttr(),
      builder.getDenseI32ArrayAttr({static_cast<int32_t>(upper.size()),
                                    static_cast<int32_t>(initVals.size())}));

  // Parse attributes.
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.resolveOperands(initVals, result.types, parser.getNameLoc(),
                             result.operands))
    return failure();

  // Add a terminator if none was parsed.
  mlir::kokkos::RangeParallelOp::ensureTerminator(*body, builder,
result.location); return success();
}
*/

/*
void TeamParallelOp::print(OpAsmPrinter &p) {
  p << " (" << getBody()->getArguments() << ") -> (" << getUpperBound() << ")";
  if (!getInitVals().empty())
    p << " init (" << getInitVals() << ")";
  p.printOptionalArrowTypeList(getResultTypes());
  p << ' ';
  p.printRegion(getRegion(), false);
  p.printOptionalAttrDict(
      (*this)->getAttrs(),
      RangeParallelOp::getOperandSegmentSizeAttr());
}
*/

void TeamParallelOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  // Both the operation itself and the region may be branching into the body or
  // back into the operation itself. It is possible for loop not to enter the
  // body.
  regions.push_back(RegionSuccessor(&getRegion()));
  regions.push_back(RegionSuccessor());
}

/*
LogicalResult TeamParallelOp::verify() {
  // Check that there is at least one value in upperBound.
  if (getUpperBound().empty())
    return emitOpError(
        "needs at least one tuple element for upperBound");
  auto loopDim = getUpperBound().size();

  // Check that the body defines the same number of block arguments as there
  // are upper bounds.
  Block *body = getBody();
  if (body->getNumArguments() != loopDim)
    return emitOpError() << "expects the same number of induction variables: "
                         << body->getNumArguments()
                         << " as bounds: " << loopDim;
  for (auto arg : body->getArguments())
    if (!arg.getType().isIndex())
      return emitOpError(
          "expects arguments for the induction variable to be of index type");

  // Check that the yield has no results
  auto yield = verifyAndGetTerminator<kokkos::YieldOp>(
      *this, getRegion(), "expects body to terminate with 'kokkos.yield'");
  if (!yield)
    return failure();
  if (yield->getNumOperands() != 0)
    return yield.emitOpError() << "not allowed to have operands inside '"
                               << RangeParallelOp::getOperationName() << "'";

  // Check that the number of results is the same as the number of
UpdateReductionOps.
  // Reductions can appear in 2 places: either directly as a child of body,
  // or in a single. If in a single, the single must be a direct child of body.
  SmallVector<kokkos::UpdateReductionOp, 4> reductions;
  for(auto reduce : body->getOps<kokkos::UpdateReductionOp>()) {
      reductions.push_back(reduce);
  }
  for(auto single : body->getOps<kokkos::SingleOp>()) {
    for(auto reduce : single->getOps<kokkos::UpdateReductionOp>()) {
        reductions.push_back(reduce);
    }
  }
  auto resultsSize = getResults().size();
  auto reductionsSize = reductions.size();
  auto initValsSize = getInitVals().size();
  if (resultsSize != reductionsSize)
    return emitOpError() << "expects number of results: " << resultsSize
                         << " to be the same as number of reductions: "
                         << reductionsSize;
  if (resultsSize != initValsSize)
    return emitOpError() << "expects number of results: " << resultsSize
                         << " to be the same as number of initial values: "
                         << initValsSize;

  // Check that the types of the results and reductions are the same.
  for (auto resultAndReduce : llvm::zip(getResults(), reductions)) {
    auto resultType = std::get<0>(resultAndReduce).getType();
    auto reduceOp = std::get<1>(resultAndReduce);
    auto reduceType = reduceOp.getOperand().getType();
    if (resultType != reduceType)
      return reduceOp.emitOpError()
             << "expects type of reduce: " << reduceType
             << " to be the same as result type: " << resultType;
  }
  return success();
}
*/

kokkos::UpdateReductionOp TeamParallelOp::getReduction() {
  Region &body = this->getLoopBody();
  for (kokkos::UpdateReductionOp op : body.getOps<kokkos::UpdateReductionOp>())
    return op;
  for (kokkos::SingleOp single : body.getOps<kokkos::SingleOp>()) {
    for (kokkos::UpdateReductionOp op :
         single.getRegion().getOps<kokkos::UpdateReductionOp>())
      return op;
  }
  return nullptr;
}

// ******************** //
//   ThreadParallelOp   //
// ******************** //

void ThreadParallelOp::build(OpBuilder &builder, OperationState &result,
                             Value numIters, Value vectorLengthHint,
                             TypeRange resultTypes) {
  result.addOperands(numIters);
  result.addOperands(vectorLengthHint);
  result.addTypes(resultTypes);

  OpBuilder::InsertionGuard guard(builder);
  Type argType = builder.getIndexType();
  Location argLoc = result.location;
  Region *bodyRegion = result.addRegion();
  builder.createBlock(bodyRegion, {}, ArrayRef<Type>(argType),
                      ArrayRef<Location>(argLoc));
  ThreadParallelOp::ensureTerminator(*bodyRegion, builder, result.location);
}

SmallVector<Region *> ThreadParallelOp::getLoopRegions() {
  return SmallVector<Region *>(1, &getRegion());
}

void ThreadParallelOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  // Both the operation itself and the region may be branching into the body or
  // back into the operation itself. It is possible for loop not to enter the
  // body.
  regions.push_back(RegionSuccessor(&getRegion()));
  regions.push_back(RegionSuccessor());
}

kokkos::UpdateReductionOp ThreadParallelOp::getReduction() {
  Region &body = this->getLoopBody();
  for (kokkos::UpdateReductionOp op : body.getOps<kokkos::UpdateReductionOp>())
    return op;
  for (kokkos::SingleOp single : body.getOps<kokkos::SingleOp>()) {
    for (kokkos::UpdateReductionOp op :
         single.getRegion().getOps<kokkos::UpdateReductionOp>())
      return op;
  }
  return nullptr;
}

// ********* //
//  SingleOp //
// ********* //

LogicalResult mlir::kokkos::SingleOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    SingleOp::Adaptor adaptor, SmallVectorImpl<Type> &inferredReturnTypes) {
  // Return types are identical to operand types:
  // arguments on executing thread are broadcast to the rest of the team or
  // thread
  for (auto arg : adaptor.getOperands())
    inferredReturnTypes.push_back(arg.getType());
  return success();
}

// ******************** //
//  UpdateReductionOp   //
// ******************** //

void UpdateReductionOp::build(
    OpBuilder &builder, OperationState &result, Value update, Value identity,
    function_ref<void(OpBuilder &, Location, Value, Value)> bodyBuilderFn) {
  auto type = update.getType();
  result.addOperands(update);
  result.addOperands(identity);

  OpBuilder::InsertionGuard guard(builder);
  Region *bodyRegion = result.addRegion();
  Block *body = builder.createBlock(bodyRegion, {}, ArrayRef<Type>{type, type},
                                    {result.location, result.location});
  if (bodyBuilderFn)
    bodyBuilderFn(builder, result.location, body->getArgument(0),
                  body->getArgument(1));
}

#define GET_OP_CLASSES
#include "lapis/Dialect/Kokkos/IR/Kokkos.cpp.inc"
#include "lapis/Dialect/Kokkos/IR/KokkosDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// convenience methods.
//===----------------------------------------------------------------------===//

namespace mlir::kokkos {
Value getParentMemref(Value v) {
  Operation *op = v.getDefiningOp();
  if (!op)
    return v;
  return llvm::TypeSwitch<Operation *, Value>(op)
      .Case<memref::SubViewOp, memref::CollapseShapeOp, memref::CastOp,
            memref::ReinterpretCastOp>(
          [&](auto op) { return getParentMemref(op->getOperand(0)); })
      .Default([&](Operation *) { return v; });
}

func::FuncOp getCalledFunction(func::CallOp callOp) {
  SymbolRefAttr sym =
      llvm::dyn_cast_if_present<SymbolRefAttr>(callOp.getCallableForCallee());
  if (!sym)
    return nullptr;
  return dyn_cast_or_null<func::FuncOp>(
      SymbolTable::lookupNearestSymbolFrom(callOp, sym));
}

MemorySpace getMemSpace(Value v) {
  // First, any value that is passed to or returned from an extern function
  // is assumed to be represented on host (so either host or dualview)
  bool hostRepresented = false;
  bool deviceRepresented = false;
  // device represented if and only if v is used in an op enclosed in a
  // kokkos.team_parallel, kokkos.thread_parallel or a kokkos.range_parallel
  // with execution space == Device.
  for (auto &use : v.getUses()) {
    Operation *usingOp = use.getOwner();
    if (usingOp->getParentOfType<kokkos::ThreadParallelOp>() ||
        usingOp->getParentOfType<kokkos::TeamParallelOp>()) {
      deviceRepresented = true;
    } else if (auto rangePar =
                   usingOp->getParentOfType<kokkos::RangeParallelOp>()) {
      if (rangePar.getExecutionSpace() == ExecutionSpace::Host) {
        hostRepresented = true;
      } else {
        // rangePar's execution space is either TeamHandle
        // (which always indicates execution on device)
        // or Device (for the top-level RangePolicy).
        deviceRepresented = true;
      }
    } else if (auto call = dyn_cast<func::CallOp>(usingOp)) {
      // v is used here as a call argument
      func::FuncOp callee = ::mlir::kokkos::getCalledFunction(call);
      // If we can't resolve call to a FuncOp, we can't do any analysis
      if (!callee)
        continue;
      // If callee is extern (a declaration only), assume the function
      // definition only access v on host. This applies to sparse tensor/part
      // tensor runtime library calls.
      if (callee.isExternal()) {
        hostRepresented = true;
      } else {
        // Assume we need both host and device representation.
        // At runtime, this will translate to a lazy DualView.
        hostRepresented = true;
        deviceRepresented = true;
      }
    }
  }
  // Finally, if v is a result of a call, make sure it's represented correctly.
  // If it's the result of a call to an extern function, assume it's present on
  // host.
  if (auto call = v.getDefiningOp<func::CallOp>()) {
    func::FuncOp callee = getCalledFunction(call);
    // If we can't resolve call to a FuncOp, we can't do any analysis
    if (callee && callee.isExternal()) {
      hostRepresented = true;
    }
  }
  // TODO: analyze the full call graph.
  // Check all return statements in a FuncOp,
  // and join the spaces of all possible returned values.
  // Note: if v appears to be used on neither host nor device, put it on host.
  if (!deviceRepresented) {
    // either host only, or neither
    return MemorySpace::Host;
  } else {
    // Device represented
    if (hostRepresented)
      return MemorySpace::DualView;
    else
      return MemorySpace::Device;
  }
}

// Get the parallel nesting depth of the given Op
// - If Op itself is a kokkos.parallel or scf.parallel, then that counts as 1
// - Otherwise, Op counts for 0
// - Each enclosing parallel counts for 1 more
int getOpParallelDepth(Operation *op) {
  int depth = 0;
  if (isa<scf::ParallelOp, kokkos::RangeParallelOp, kokkos::TeamParallelOp,
          kokkos::ThreadParallelOp>(op))
    depth++;
  Operation *parent = op->getParentOp();
  if (parent)
    return depth + getOpParallelDepth(parent);
  // op has no parent
  return depth;
}

// Determine which execution space (Host or Device) executes the given op.
// Note that op may contain parallel kernels that execute on device,
// but in that case op itself still counts as Host.
// TODO: this will require a different approach if function calls are allowed in
// device kernels.
kokkos::ExecutionSpace getOpExecutionSpace(Operation *op) {
  if (op->getParentOfType<kokkos::ThreadParallelOp>() ||
      op->getParentOfType<kokkos::TeamParallelOp>())
    return kokkos::ExecutionSpace::Device;
  if (auto rangeParallel = op->getParentOfType<kokkos::RangeParallelOp>())
    return rangeParallel.getExecutionSpace();
  return kokkos::ExecutionSpace::Host;
}

// Get a list of the memrefs read by op.
DenseSet<Value> getMemrefsRead(Operation *op, kokkos::ExecutionSpace space) {
  DenseSet<Value> memrefs;
  op->walk([&](Operation *subOp) {
    if (getOpExecutionSpace(subOp) != space)
      return;
    if (auto load = dyn_cast<memref::LoadOp>(subOp))
      memrefs.insert(load.getMemref());
    else if (auto atomicUpdate = dyn_cast<memref::AtomicRMWOp>(subOp))
      memrefs.insert(atomicUpdate.getMemref());
    else if (auto call = dyn_cast<func::CallOp>(subOp)) {
      // Assume that all memref-typed arguments can be read by the callee.
      for (Value arg : call.getArgOperands()) {
        if (isa<MemRefType, UnrankedMemRefType>(arg.getType())) {
          memrefs.insert(arg);
        }
      }
    }
  });
  return memrefs;
}

// Get a list of the memrefs (possibly) written to by op.
DenseSet<Value> getMemrefsWritten(Operation *op, kokkos::ExecutionSpace space) {
  DenseSet<Value> memrefs;
  op->walk([&](Operation *subOp) {
    if (getOpExecutionSpace(subOp) != space)
      return;
    if (auto store = dyn_cast<memref::StoreOp>(subOp))
      memrefs.insert(store.getMemref());
    else if (auto atomicUpdate = dyn_cast<memref::AtomicRMWOp>(subOp))
      memrefs.insert(atomicUpdate.getMemref());
    else if (auto call = dyn_cast<func::CallOp>(subOp)) {
      // Assume that all memref-typed arguments can be read by the callee,
      // since memrefs of const data cannot be represented in MLIR.
      // TODO: actually check non-extern callees for which memrefs get
      // read/written.
      for (Value arg : call.getArgOperands()) {
        if (isa<MemRefType, UnrankedMemRefType>(arg.getType())) {
          memrefs.insert(arg);
        }
      }
    }
  });
  return memrefs;
}

// Is v a compile-time constant integer with value 0?
bool valueIsIntegerConstantZero(Value v) {
  // If we don't know what op generated v, can't assume anything about its value
  if (!v.getDefiningOp())
    return false;
  if (auto constantOp = dyn_cast<arith::ConstantOp>(v.getDefiningOp())) {
    auto valAttr = constantOp.getValue();
    if (auto iAttr = dyn_cast<IntegerAttr>(valAttr)) {
      return iAttr.getValue().isZero();
    }
    return false;
  }
  return false;
}

// Is v a compile-time constant integer with value 1?
bool valueIsIntegerConstantOne(Value v) {
  // If we don't know what op generated v, can't assume anything about its value
  if (!v.getDefiningOp())
    return false;
  if (auto constantOp = dyn_cast<arith::ConstantOp>(v.getDefiningOp())) {
    auto valAttr = constantOp.getValue();
    if (auto iAttr = dyn_cast<IntegerAttr>(valAttr)) {
      return iAttr.getValue().isOne();
    }
    return false;
  }
  return false;
}

} // namespace mlir::kokkos

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
#include "mlir/Interfaces/DataLayoutInterfaces.h"

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

// *************** //
//  AllocScratchOp //
// *************** //

LogicalResult AllocScratchOp::verify() {
  if(!isa<MemRefType>(getMemref().getType()))
    return emitOpError("result type is not a MemRefType");
  size_t elemAlignment = getBuiltinTypeSize(getType().getElementType(), *this);
  if(elemAlignment == 0)
    return emitOpError("cannot statically determine size & alignment of element type");
  // Make sure that the result's type has a known size in bytes
  size_t unused;
  if(!memrefSizeInBytesKnown(getType(), unused, *this))
    return emitOpError("cannot statically determine allocation size in bytes");
  // Make sure that the starting offset is sufficiently aligned for the element type
  if(getScratchBegin() % elemAlignment)
    return emitOpError("scratch begin pointer is not sufficiently aligned for element type");
  return success();
}

uint64_t AllocScratchOp::getScratchBegin() {
  return getOffset().getLimitedValue();
}

uint64_t AllocScratchOp::getScratchEnd() {
  size_t size;
  // We should have already verified that this allocation is statically sized
  bool succeeded = memrefSizeInBytesKnown(getType(), size, *this);
  if(!succeeded) {
    emitOpError("Could not determine scratch end pointer");
  }
  return getScratchBegin() + size;
}

#define GET_OP_CLASSES
#include "lapis/Dialect/Kokkos/IR/Kokkos.cpp.inc"
#include "lapis/Dialect/Kokkos/IR/KokkosDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// convenience methods.
//===----------------------------------------------------------------------===//

namespace mlir::kokkos {

bool isViewAliasingOp(Operation* op) {
  // This is not a complete list, but should be kept up to date with what the emitter supports
  return isa<
    memref::SubViewOp, memref::CollapseShapeOp,
    memref::CastOp, memref::ReinterpretCastOp,
    memref::ExtractStridedMetadataOp, memref::ReshapeOp
      >(op);
}

Value getParentMemref(Value v) {
  Operation *op = v.getDefiningOp();
  if (!op)
    return v;
  if(isViewAliasingOp(op))
    return getParentMemref(op->getOperand(0));
  return v;
}

func::FuncOp getCalledFunction(func::CallOp callOp) {
  SymbolRefAttr sym =
      llvm::dyn_cast_if_present<SymbolRefAttr>(callOp.getCallableForCallee());
  if (!sym)
    return nullptr;
  return dyn_cast_or_null<func::FuncOp>(
      SymbolTable::lookupNearestSymbolFrom(callOp, sym));
}

bool funcHasBody(func::FuncOp op) {
  return op.getCallableRegion() != nullptr;
}

// Tally the memory spaces where v is accessed, without analyzing v's parent memref(s) if any.
// We already assume that v has no parent (it's not the result of a cast/reshape like op)
static MemorySpace getMemSpaceImpl(Value v) {
  // If v is a function parameter, it's DualView automatically
  if(getFuncWithParameter(v))
    return MemorySpace::DualView;
  // If v is the result of a call to a non-extern function, it's also a DualView automatically
  func::CallOp producingCall = v.getDefiningOp<func::CallOp>();
  if(producingCall && funcHasBody(getCalledFunction(producingCall)))
    return MemorySpace::DualView;
  bool hostRepresented = false;
  bool deviceRepresented = false;
  for (auto &use : v.getUses()) {
    Operation *usingOp = use.getOwner();
    // device represented if v is used in an op enclosed in a
    // kokkos.team_parallel, kokkos.thread_parallel or a kokkos.range_parallel
    // with execution space == Device.
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
    else if(isa<memref::LoadOp, memref::StoreOp, memref::AtomicRMWOp>(usingOp)) {
      // Direct element access outside of any parallel loop must be on host.
      hostRepresented = true;
    }
    else if(isViewAliasingOp(usingOp) && v == usingOp->getOperand(0)) {
      // usingOp does not directly access v's memory, but its result can be accessed.
      // Recursively find the usages of this result too.
      MemorySpace childSpace = getMemSpaceImpl(usingOp->getResult(0));
      if(childSpace == MemorySpace::Host || childSpace == MemorySpace::DualView) {
        hostRepresented = true;
      }
      if(childSpace == MemorySpace::Device || childSpace == MemorySpace::DualView) {
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
    if (hostRepresented)
      return MemorySpace::Host;
    else {
      // If v nor any children are accessed directly, it must be a function parameter.
      // Put it on device.
      return MemorySpace::Device;
    }
  } else {
    // Device represented
    if (hostRepresented)
      return MemorySpace::DualView;
    else
      return MemorySpace::Device;
  }
}

MemorySpace getMemSpace(Value v) {
  auto op = v.getDefiningOp();
  if(op && isViewAliasingOp(op)) {
    // op's first operand is the "parent" memref of v.
    // Recursively get the memory space of the parent. If it's a DualView then v needs to be one also.
    Value parent = op->getOperand(0);
    return getMemSpace(parent);
  }
  // v has no parent, so we can analyze its space top-down now.
  return getMemSpaceImpl(v);
}

func::FuncOp getFuncWithParameter(Value v) {
  // Early-out: if an operation produced v, it's definitely not a function parameter
  if(v.getDefiningOp())
    return nullptr;
  Region* r = v.getParentRegion();
  func::FuncOp f = r->getParentOfType<func::FuncOp>();
  Region& body = f.getBody();
  // v is a func parameter iff it's one of body's arguments.
  for(auto arg : body.getArguments()) {
    if(arg == v) return f;
  }
  return nullptr;
}

// Overload of getMemSpace for global memref declarations
// It just iterates over all the memref.get_global ops that reference 
MemorySpace getMemSpace(memref::GlobalOp global) {
  // Get the module that encloses global
  ModuleOp module = global->getParentOfType<ModuleOp>();
  // Then walk all the GetGlobal ops within module
  bool usedOnHost = false;
  bool usedOnDevice = false;
  module->walk([&](memref::GetGlobalOp gg) {
    // Check usage of gg, if it references global by name.
    // note: StringRef::compare is like strcmp; 0 means equal.
    if(!global.getSymName().compare(gg.getName())) {
      MemorySpace usageSpace = getMemSpace(gg.getResult());
      if(usageSpace == MemorySpace::DualView || usageSpace == MemorySpace::Host) {
        usedOnHost = true;
      }
      if(usageSpace == MemorySpace::DualView || usageSpace == MemorySpace::Device) {
        usedOnDevice = true;
      }
    }
  });
  if(usedOnDevice)
    return usedOnHost ? MemorySpace::DualView : MemorySpace::Device;
  // If not used at all, default to host
  return MemorySpace::Host;
}

bool isGlobalUsed(memref::GlobalOp global) {
  ModuleOp module = global->getParentOfType<ModuleOp>();
  bool used = false;
  module->walk([&](memref::GetGlobalOp gg) {
    if(!global.getSymName().compare(gg.getName()))
      used = true;
  });
  return used;
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
      // Assume that all memref-typed arguments can be written to by the callee,
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

// If struct type contains only one type (as either single members,
// nested structs, or LLVM arrays) then return that type.
// Otherwise, return null.
// Also returns null if st has no members.
Type getStructElementType(LLVM::LLVMStructType st)
{
  SmallVector<Type> types;
  for(Type t : st.getBody()) {
    if(auto nestedArray = dyn_cast<LLVM::LLVMArrayType>(t)) {
      types.push_back(nestedArray.getElementType());
    }
    else if(auto nestedStruct = dyn_cast<LLVM::LLVMStructType>(t)) {
      Type nestedElemType = getStructElementType(nestedStruct);
      if(!nestedElemType)
        return nullptr;
      types.push_back(nestedElemType);
    }
    else {
      types.push_back(t);
    }
  }
  if(types.size() == size_t(0))
    return nullptr;
  else if(types.size() == size_t(1))
    return types[0];
  Type t = types[0];
  for(size_t i = 1; i < types.size(); i++) {
    if(types[i] != t)
      return nullptr;
  }
  return t;
}

// If the above function returns true, count the total number of elements in the struct:
// sizeof(structType) / sizeof(elemType)
int getStructElementCount(LLVM::LLVMStructType st)
{
  int size = 0;
  for(Type t : st.getBody()) {
    if(auto nestedArray = dyn_cast<LLVM::LLVMArrayType>(t)) {
      size += nestedArray.getNumElements();
    }
    else if(auto nestedStruct = dyn_cast<LLVM::LLVMStructType>(t)) {
      size += getStructElementCount(nestedStruct);
    }
    else {
      size++;
    }
  }
  return size;
}

size_t getBuiltinTypeSize(Type t, Operation* op)
{
  auto dl = DataLayout::closest(op);
  llvm::TypeSize elementSize = dl.getTypeSize(t);
  if(!elementSize.isFixed())
    return 0;
  return elementSize.getFixedValue();
}

bool memrefSizeInBytesKnown(MemRefType mrt, size_t& size, Operation* op)
{
  // First, check if mrt has all static dimensions.
  if(!mrt.hasStaticShape())
    return false;
  // Then get the number of elements.
  size_t numElements = mrt.getNumElements();
  // Check if the element type has a known size
  size_t elementSize = getBuiltinTypeSize(mrt.getElementType(), op);
  if(elementSize == 0)
    return false;
  size = numElements * elementSize;
  return true;
}

} // namespace mlir::kokkos

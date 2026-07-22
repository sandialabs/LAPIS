//===- OutlineAlgebraicKernels.cpp ----------------------------------------===//

#include "lapis/Transform/AlgebraicKernelFusion.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <string>

//===----------------------------------------------------------------------===//
// Algebraic kernel outlining
//===----------------------------------------------------------------------===//

namespace mlir::lapis {
namespace {

struct OutlineGroup {
  func::FuncOp parent;
  std::int64_t group;
  llvm::SmallVector<linalg::GenericOp, 4> operations;
};

struct PrivateBufferAnalysis {
  llvm::DenseSet<Value> buffers;
  llvm::DenseSet<Operation *> ownedOperations;
};

bool isPrivateBufferAuxiliaryUse(Value buffer, Operation *owner,
                                 PrivateBufferAnalysis &analysis) {
  if (auto copy = dyn_cast<memref::CopyOp>(owner)) {
    if (copy.getTarget() != buffer)
      return false;
    analysis.ownedOperations.insert(owner);
    return true;
  }
  if (auto fill = dyn_cast<linalg::FillOp>(owner)) {
    if (llvm::none_of(fill.getDpsInits(),
                      [&](Value output) { return output == buffer; }))
      return false;
    analysis.ownedOperations.insert(owner);
    return true;
  }
  if (isa<memref::DeallocOp>(owner) && owner->getOperand(0) == buffer) {
    analysis.ownedOperations.insert(owner);
    return true;
  }
  return false;
}

/// Finds allocation-backed buffers that communicate only between operations
/// in one fusion group. Their allocation, initialization, and optional
/// deallocation can move with the group instead of becoming kernel arguments.
PrivateBufferAnalysis findPrivateBuffers(OutlineGroup &group) {
  llvm::DenseSet<Operation *> members;
  for (linalg::GenericOp operation : group.operations)
    members.insert(operation);

  llvm::DenseSet<Value> candidates;
  for (auto [producerNumber, producer] : llvm::enumerate(group.operations)) {
    for (Value output : producer.getDpsInits()) {
      bool consumedLater = llvm::any_of(
          llvm::ArrayRef(group.operations).drop_front(producerNumber + 1),
          [&](linalg::GenericOp consumer) {
            return llvm::is_contained(consumer.getDpsInputs(), output);
          });
      if (consumedLater)
        candidates.insert(output);
    }
  }

  PrivateBufferAnalysis analysis;
  for (Value candidate : candidates) {
    Operation *allocation = candidate.getDefiningOp();
    if (!isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(allocation) ||
        allocation->getBlock() != group.operations.front()->getBlock())
      continue;

    PrivateBufferAnalysis candidateAnalysis;
    bool isPrivate = llvm::all_of(candidate.getUses(), [&](OpOperand &use) {
      Operation *owner = use.getOwner();
      return members.contains(owner) ||
             isPrivateBufferAuxiliaryUse(candidate, owner, candidateAnalysis);
    });
    if (!isPrivate)
      continue;

    analysis.buffers.insert(candidate);
    analysis.ownedOperations.insert(allocation);
    analysis.ownedOperations.insert(candidateAnalysis.ownedOperations.begin(),
                                    candidateAnalysis.ownedOperations.end());
  }
  return analysis;
}

bool canMoveGroupToLastOperation(OutlineGroup &group) {
  llvm::DenseSet<Operation *> members;
  for (linalg::GenericOp operation : group.operations)
    members.insert(operation.getOperation());

  llvm::DenseSet<Value> previouslyAccessedBuffers;
  Operation *first = group.operations.front().getOperation();
  Operation *last = group.operations.back().getOperation();
  for (Operation *operation = first; operation != last;
       operation = operation->getNextNode()) {
    if (members.contains(operation)) {
      previouslyAccessedBuffers.insert(operation->operand_begin(),
                                       operation->operand_end());
      continue;
    }
    if (isMemoryEffectFree(operation) ||
        isa<memref::AllocOp, memref::AllocaOp>(operation))
      continue;
    if (auto copy = dyn_cast<memref::CopyOp>(operation)) {
      // Moving the contractions past a copy is only safe when the copy neither
      // observes a value produced by the group nor overwrites one of its
      // inputs or outputs.
      if (!previouslyAccessedBuffers.contains(copy.getSource()) &&
          !previouslyAccessedBuffers.contains(copy.getTarget()))
        continue;
    }
    if (auto fill = dyn_cast<linalg::FillOp>(operation)) {
      if (llvm::none_of(fill.getDpsInits(), [&](Value output) {
            return previouslyAccessedBuffers.contains(output);
          }))
        continue;
    }
    return false;
  }
  return true;
}

std::string getUniqueKernelName(ModuleOp module, StringRef parentName,
                                std::int64_t group) {
  std::string base =
      (parentName + "__lapis_algebraic_kernel_" + llvm::Twine(group)).str();
  std::string name = base;
  for (unsigned suffix = 0; module.lookupSymbol(name); ++suffix)
    name = (base + "_" + llvm::Twine(suffix)).str();
  return name;
}

LogicalResult outlineGroup(ModuleOp module, OutlineGroup &group) {
  if (group.operations.size() < 2)
    return group.operations.front().emitError(
        "an algebraic fusion group must contain at least two contractions");
  Block *block = group.operations.front()->getBlock();
  if (!llvm::all_of(group.operations, [&](linalg::GenericOp operation) {
        return operation->getBlock() == block &&
               operation.hasPureBufferSemantics() &&
               operation->getNumResults() == 0;
      })) {
    return group.operations.front().emitError(
        "algebraic kernels must be outlined after bufferization from one "
        "block");
  }
  if (!canMoveGroupToLastOperation(group)) {
    return group.operations.front().emitError(
        "cannot outline an algebraic group across an intervening "
        "side-effecting operation");
  }

  PrivateBufferAnalysis privateBuffers = findPrivateBuffers(group);
  llvm::DenseSet<Operation *> clonedOperations(
      privateBuffers.ownedOperations.begin(),
      privateBuffers.ownedOperations.end());
  llvm::DenseSet<Operation *> operationsToErase(
      privateBuffers.ownedOperations.begin(),
      privateBuffers.ownedOperations.end());
  for (linalg::GenericOp operation : group.operations) {
    clonedOperations.insert(operation);
    operationsToErase.insert(operation);
  }

  // Keep constant initialization self-contained. Passing a memref.get_global
  // result through the outlined signature both obscures ownership and causes
  // LAPIS's DualView management to treat the global as a mutable argument.
  // Clone these side-effect-free support operations, but leave their original
  // definitions in place for any other caller-side initializers.
  for (Operation *operation : privateBuffers.ownedOperations) {
    for (Value operand : operation->getOperands()) {
      Operation *definition = operand.getDefiningOp();
      if (isa_and_nonnull<memref::GetGlobalOp, arith::ConstantOp>(definition) &&
          definition->getBlock() == block)
        clonedOperations.insert(definition);
    }
  }

  llvm::SmallVector<Operation *, 8> operationsToClone;
  for (Operation &operation : *block) {
    if (clonedOperations.contains(&operation))
      operationsToClone.push_back(&operation);
  }

  llvm::SetVector<Value> arguments;
  auto addArgument = [&](Value value) {
    Operation *definition = value.getDefiningOp();
    if (definition && clonedOperations.contains(definition))
      return;
    arguments.insert(value);
  };
  for (Operation *operation : operationsToClone) {
    for (Value operand : operation->getOperands())
      addArgument(operand);
    if (operation->getNumRegions() == 0)
      continue;
    llvm::SetVector<Value> captures;
    getUsedValuesDefinedAbove(operation->getRegions(), captures);
    for (Value capture : captures)
      addArgument(capture);
  }

  // Private buffers are defined by cloned allocation operations, so they must
  // never remain in the outlined function signature.
  for (Value buffer : privateBuffers.buffers) {
    if (arguments.contains(buffer)) {
      return group.operations.front().emitError(
          "private intermediate unexpectedly remained a kernel argument");
    }
  }

  OpBuilder builder(group.parent);
  builder.setInsertionPoint(group.parent);
  FunctionType type =
      builder.getFunctionType(TypeRange(arguments.getArrayRef()), {});
  std::string name =
      getUniqueKernelName(module, group.parent.getSymName(), group.group);
  auto kernel = builder.create<func::FuncOp>(group.operations.front().getLoc(),
                                             name, type);
  kernel.setPrivate();
  kernel->setAttr(kAlgebraicKernelAttr, builder.getUnitAttr());
  kernel->setAttr(kAlgebraicFusionGroupAttr,
                  builder.getI64IntegerAttr(group.group));
  Block *entry = kernel.addEntryBlock();

  IRMapping mapping;
  for (auto [argument, blockArgument] :
       llvm::zip(arguments, entry->getArguments()))
    mapping.map(argument, blockArgument);
  builder.setInsertionPointToStart(entry);
  for (Operation *operation : operationsToClone)
    builder.clone(*operation, mapping);
  builder.create<func::ReturnOp>(kernel.getLoc());

  builder.setInsertionPoint(group.operations.back());
  auto call = builder.create<func::CallOp>(group.operations.back().getLoc(),
                                           kernel, arguments.getArrayRef());
  call->setAttr(kAlgebraicKernelAttr, builder.getUnitAttr());
  call->setAttr(kAlgebraicFusionGroupAttr,
                builder.getI64IntegerAttr(group.group));

  for (Operation *operation : llvm::reverse(operationsToClone)) {
    if (operationsToErase.contains(operation))
      operation->erase();
  }
  return success();
}

} // namespace

LogicalResult outlineAlgebraicKernels(ModuleOp module) {
  llvm::SmallVector<OutlineGroup, 4> groups;
  llvm::DenseMap<std::int64_t, unsigned> groupNumbers;

  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    if (function->hasAttr(kAlgebraicKernelAttr))
      continue;
    WalkResult result = function.walk([&](linalg::GenericOp operation) {
      auto groupAttr =
          operation->getAttrOfType<IntegerAttr>(kAlgebraicFusionGroupAttr);
      if (!groupAttr)
        return WalkResult::advance();
      std::int64_t group = groupAttr.getInt();
      auto [iterator, inserted] =
          groupNumbers.try_emplace(group, groups.size());
      if (inserted)
        groups.push_back(OutlineGroup{function, group, {}});
      OutlineGroup &outlineGroup = groups[iterator->second];
      if (outlineGroup.parent != function) {
        operation.emitError(
            "an algebraic fusion group crosses function boundaries");
        return WalkResult::interrupt();
      }
      outlineGroup.operations.push_back(operation);
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return failure();
  }

  // Outlining later groups first keeps every cached operation position valid.
  for (OutlineGroup &group : llvm::reverse(groups)) {
    if (failed(outlineGroup(module, group)))
      return failure();
  }
  return success();
}

} // namespace mlir::lapis

namespace mlir {
#define GEN_PASS_DEF_OUTLINEALGEBRAICKERNELSPASS
#include "lapis/Transform/Passes.h.inc"
} // namespace mlir

namespace {
struct OutlineAlgebraicKernelsPass
    : public mlir::impl::OutlineAlgebraicKernelsPassBase<
          OutlineAlgebraicKernelsPass> {
  using Base::Base;

  void runOnOperation() override {
    if (failed(mlir::lapis::outlineAlgebraicKernels(getOperation())))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> mlir::createOutlineAlgebraicKernelsPass() {
  return std::make_unique<OutlineAlgebraicKernelsPass>();
}

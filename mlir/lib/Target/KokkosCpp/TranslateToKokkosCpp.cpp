//===- TranslateToKokkosCpp.cpp - Translate MLIR to Kokkos -----------------===//
//
// **** This file has been modified from its original in llvm-project ****
// This file was based on mlir/lib/Target/Cpp/TranslateToCpp.cpp
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lapis/Target/KokkosCpp/KokkosCppEmitter.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "lapis/Dialect/Kokkos/IR/KokkosDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include <iostream>
#include <utility>
// TODO: use an LLVM map instead
#include <unordered_map>

#ifdef __unix__
#include <unistd.h>
#endif

using namespace mlir;
using llvm::formatv;

/// Convenience functions to produce interleaved output with functions returning
/// a LogicalResult. This is different than those in STLExtras as functions used
/// on each element doesn't return a string.
template <typename ForwardIterator, typename UnaryFunctor,
          typename NullaryFunctor>
inline LogicalResult
interleaveWithError(ForwardIterator begin, ForwardIterator end,
                    UnaryFunctor eachFn, NullaryFunctor betweenFn) {
  if (begin == end)
    return success();
  if (failed(eachFn(*begin)))
    return failure();
  ++begin;
  for (; begin != end; ++begin) {
    betweenFn();
    if (failed(eachFn(*begin)))
      return failure();
  }
  return success();
}

template <typename Container, typename UnaryFunctor, typename NullaryFunctor>
inline LogicalResult interleaveWithError(const Container &c,
                                         UnaryFunctor eachFn,
                                         NullaryFunctor betweenFn) {
  return interleaveWithError(c.begin(), c.end(), eachFn, betweenFn);
}

template <typename Container, typename UnaryFunctor>
inline LogicalResult interleaveCommaWithError(const Container &c,
                                              raw_ostream &os,
                                              UnaryFunctor eachFn) {
  return interleaveWithError(c.begin(), c.end(), eachFn, [&]() { os << ", "; });
}

namespace {
struct KokkosCppEmitter {
  explicit KokkosCppEmitter(raw_ostream &decl_os, raw_ostream &os, bool enableSparseSupport);
  explicit KokkosCppEmitter(raw_ostream &decl_os, raw_ostream &os, raw_ostream& py_os, bool enableSparseSupport);

  /// Emits attribute or returns failure.
  LogicalResult emitAttribute(Location loc, Attribute attr);

  /// Emits operation 'op' with/without training semicolon or returns failure.
  LogicalResult emitOperation(Operation &op, bool trailingSemicolon);

  /// Emits the functions lapis_initialize() and lapis_finalize()
  /// These are responsible for init/finalize of Kokkos, and allocation/initialization/deallocation
  /// of global Kokkos::Views.
  LogicalResult emitInitAndFinalize();

  LogicalResult emitPythonBoilerplate(bool isLastKernel);

  /// Emits type 'type' or returns failure.
  /// If forSparseRuntime, the emitted type is compatible with PyTACO runtime and the sparse support library.
  /// For example, memrefs use StridedMemRefType instead of Kokkos::View.
  LogicalResult emitType(Location loc, Type type, bool forSparseRuntime = false);

  // Emit a memref type as a Kokkos::View, with the given memory space.
  LogicalResult emitMemrefType(Location loc, MemRefType type, kokkos::MemorySpace space);
  LogicalResult emitMemrefType(Location loc, UnrankedMemRefType type, kokkos::MemorySpace space);

  /// Emits array of types as a std::tuple of the emitted types.
  /// - emits void for an empty array;
  /// - emits the type of the only element for arrays of size one;
  /// - emits a std::tuple otherwise;
  /// See emitType(...) for behavior of supportFunction.
  LogicalResult emitTypes(Location loc, ArrayRef<Type> types, bool forSparseRuntime = false);

  /// Emits array of types as a std::tuple of the emitted types independently of
  /// the array size.
  LogicalResult emitTupleType(Location loc, ArrayRef<Type> types);

  /// Emits a variable declaration for a result of an operation.
  LogicalResult emitVariableDeclaration(Value result,
                                        bool trailingSemicolon);

  /// Emits the variable declaration and assignment prefix for 'op'.
  /// - emits separate variable followed by std::tie for multi-valued operation;
  /// - emits single type followed by variable for single result;
  /// - emits nothing if no value produced by op;
  /// Emits final '=' operator where a type is produced.
  /// Returns failure if any result type could not be converted.
  LogicalResult emitAssignPrefix(Operation &op);

  /// Emits a label for the block.
  LogicalResult emitLabel(Block &block);

  /// Emits the operands and atttributes of the operation. All operands are
  /// emitted first and then all attributes in alphabetical order.
  LogicalResult emitOperandsAndAttributes(Operation &op,
                                          ArrayRef<StringRef> exclude = {});

  /// Emits the operands of the operation. All operands are emitted in order.
  LogicalResult emitOperands(Operation &op);

  /// Return the name of a previously declared Value, or a literal constant.
  LogicalResult emitValue(Value val);

  /// Get a new unique identifier that won't conflict with any other variable names.
  std::string getUniqueIdentifier();

  /// Return the existing or a new name for a Value.
  StringRef getOrCreateName(Value val);

  /// Directly assign a name for a Value.
  void assignName(StringRef name, Value val);

  /// Return the existing or a new label of a Block.
  StringRef getOrCreateName(Block &block);

  /// Whether to map an mlir integer to a unsigned integer in C++.
  bool shouldMapToUnsigned(IntegerType::SignednessSemantics val);

  /// RAII helper function to manage entering/exiting C++ scopes.
  struct Scope {
    Scope(KokkosCppEmitter &emitter)
        : valueMapperScope(emitter.valueMapper),
          blockMapperScope(emitter.blockMapper), emitter(emitter) {
      emitter.valueInScopeCount.push(emitter.valueInScopeCount.top());
      emitter.labelInScopeCount.push(emitter.labelInScopeCount.top());
    }
    ~Scope() {
      emitter.valueInScopeCount.pop();
      emitter.labelInScopeCount.pop();
    }

  private:
    llvm::ScopedHashTableScope<Value, std::string> valueMapperScope;
    llvm::ScopedHashTableScope<Block *, std::string> blockMapperScope;

    KokkosCppEmitter &emitter;
  };

  /// Returns wether the Value is assigned to a C++ variable in the scope.
  bool hasValueInScope(Value val);

  // Returns whether a label is assigned to the block.
  bool hasBlockLabel(Block &block);

  /// Returns the C++ output stream for declarations that will precede the emitted module.
  raw_indented_ostream &decl_ostream() { return decl_os; };

  /// Returns the C++ output stream.
  raw_indented_ostream &ostream() { return os; };

  bool emittingPython() {return py_os.get() != nullptr;}

  /// Returns the Python output stream.
  raw_indented_ostream &py_ostream() { return *py_os.get(); };

  /// Returns true iff. this emitter is producing code with PyTACO sparse tensor support.
  bool supportingSparse() { return enableSparseSupport; };

  //This lets the emitter act like a stream (writes to the C++ file)
  template<typename T>
  raw_indented_ostream& operator<<(const T& t)
  {
    os << t;
    return os;
  }

  void registerGlobalView(memref::GlobalOp op)
  {
    globalViews.push_back(op);
  }

  //Is v a scalar constant?
  bool isScalarConstant(Value v) const
  {
    return scalarConstants.find(v) != scalarConstants.end();
  }

  //val is how other ops reference the value.
  //attr actually describes the type and data of the literal.
  //The original arith::ConstantOp is not needed.
  void registerScalarConstant(Value val, arith::ConstantOp op)
  {
    scalarConstants[val] = op;
  }

  arith::ConstantOp getScalarConstantOp(Value v) const
  {
    return scalarConstants[v];
  }

  //Get the total number of elements (aka span, since it's contiguous) of a
  //statically sized MemRefType.
  static int64_t getMemrefSpan(MemRefType memrefType)
  {
    int64_t span = 1;
    for(auto extent : memrefType.getShape())
    {
      span *= extent;
    }
    return span;
  }

private:
  using ValueMapper = llvm::ScopedHashTable<Value, std::string>;
  using BlockMapper = llvm::ScopedHashTable<Block *, std::string>;

  /// Output stream for declarations that must go before ops, in global namespace.
  raw_indented_ostream decl_os;

  /// C++ output stream to emit to.
  raw_indented_ostream os;

  /// Python output stream to emit to.
  std::shared_ptr<raw_indented_ostream> py_os;

  /// Whether we are adding the boilerplate to support the MLIR sparse
  /// tensor runtime functions (e.g. newSparseTensor)
  bool enableSparseSupport;

  /// Map from value to name of C++ variable that contain the name.
  ValueMapper valueMapper;

  /// Map from block to name of C++ label.
  BlockMapper blockMapper;

  /// The number of values in the current scope. This is used to declare the
  /// names of values in a scope.
  std::stack<int64_t> valueInScopeCount;
  std::stack<int64_t> labelInScopeCount;

  //Bookeeping for scalar constants (individual integer and floating-point values)
  mutable llvm::DenseMap<Value, arith::ConstantOp> scalarConstants;

  //Bookeeping for Kokkos::Views in global scope.
  //Each element has a name, element type, total size and whether it is intialized.
  //If initialized, ${name}_initial is assumed to be a global 1D host array with the data.
  std::vector<memref::GlobalOp> globalViews;

  // This is a string-string map
  // Keys are the names of sparse runtime support functions as they appear in the IR
  //   (e.g. "newSparseTensor")
  // Values are pairs.
  //  First:  whether the result(s) is obtained via pointer arg instead of return value
  //  Second: the actual names of the functions in $SUPPORTLIB
  //   (e.g. "_mlir_ciface_newSparseTensor")
  std::unordered_map<std::string, std::pair<bool, std::string>> sparseSupportFunctions;
  //This helper function (to be called during constructor) populates sparseSupportFunctions
  void registerRuntimeSupportFunctions();
public:
  bool isSparseSupportFunction(StringRef s) {return sparseSupportFunctions.find(s.str()) != sparseSupportFunctions.end();}
  bool sparseSupportFunctionPointerResults(StringRef mlirName) {return sparseSupportFunctions[mlirName.str()].first;}
  // Get the real C function name for the given MLIR function name
  std::string getSparseSupportFunctionName(StringRef mlirName) {return sparseSupportFunctions[mlirName.str()].second;}

  // List of DualViews which originated from sparse support functions
  // (meaning that a SparseTensorStorageBase object owns the host-side memory).
  // Right before a function return, all these should be synced to host in case
  // the owning tensor is an output.
  SmallVector<Value> dualViewsFromTensors;
};
} // namespace

static LogicalResult printConstantOp(KokkosCppEmitter &emitter, Operation *operation,
                                     Attribute value) {
  OpResult result = operation->getResult(0);

  // Emit a variable declaration for an emitc.constant op without value.
  if (auto oAttr = dyn_cast<emitc::OpaqueAttr>(value)) {
    if (oAttr.getValue().empty())
      // The semicolon gets printed by the emitOperation function.
      return emitter.emitVariableDeclaration(result,
                                             /*trailingSemicolon=*/false);
  }

  // Emit a variable declaration.
  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  return emitter.emitAttribute(operation->getLoc(), value);
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    memref::GlobalOp op) {
  auto maybeValue = op.getInitialValue();
  //Emit the View declaration first. 
  //NOTE: using the GlobalOp's symbol name instead of a name generated for the current scope,
  //because GlobalOp does not produce a Result.
  if(failed(emitter.emitType(op.getLoc(), op.getType())))
    return failure();
  emitter << ' ' << op.getSymName() << ";\n";
  //Note: module-wide initialization will be responsible for allocating and copying the initializing data (if any).
  //Then module-wide finalization will deallocate (to avoid Kokkos warning about dealloc after finalize).
  if(maybeValue)
  {
    auto memrefType = op.getType();
    //For constants (initialized views), keep the actual data in a 1D array (with a related name).
    if(failed(emitter.emitType(op.getLoc(), memrefType.getElementType())))
      return failure();
    int64_t span = KokkosCppEmitter::getMemrefSpan(memrefType);
    emitter << ' ' << op.getSymName() << "_initial" << "[" << span << "] = ";
    //Emit the 1D array literal
    if (failed(emitter.emitAttribute(op.getLoc(), maybeValue.value())))
      return failure();
    emitter << ";\n";
  }
  //Register this in list of global views
  emitter.registerGlobalView(op);
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    memref::GetGlobalOp op) {
  // Do not declare a new C++ variable for the result.
  // Instead, just reference the global by name wherever it's used.
  emitter.assignName(op.getName(), op.getResult());
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    memref::AllocOp op) {
  OpResult result = op->getResult(0);
  kokkos::MemorySpace space = kokkos::getMemSpace(result);
  MemRefType type = op.getType();
  if (failed(emitter.emitMemrefType(op.getLoc(), type, space)))
    return failure();
  StringRef name = emitter.getOrCreateName(result);
  emitter << " " << name;
  if(space == kokkos::MemorySpace::DualView)
    emitter << "(\"" << emitter.getOrCreateName(result) << "\"";
  else
    emitter << "(Kokkos::view_alloc(Kokkos::WithoutInitializing, \"" << emitter.getOrCreateName(result) << "\"))";
  emitter << "(Kokkos::view_alloc(Kokkos::WithoutInitializing, \"" << name << "\")";
  for(auto dynSize : op.getDynamicSizes())
  {
    emitter << ", ";
    if(failed(emitter.emitValue(dynSize)))
      return failure();
  }
  emitter << ")";
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    memref::AllocaOp op) {
  OpResult result = op->getResult(0);
  kokkos::MemorySpace space = kokkos::getMemSpace(result);
  MemRefType type = op.getType();
  if (failed(emitter.emitMemrefType(op.getLoc(), type, space)))
    return failure();
  emitter << " " << emitter.getOrCreateName(result);
  if(space == kokkos::MemorySpace::DualView)
    emitter << "(\"" << emitter.getOrCreateName(result) << "\")";
  else
    emitter << "(Kokkos::view_alloc(Kokkos::WithoutInitializing, \"" << emitter.getOrCreateName(result) << "\"))";
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    memref::DeallocOp op) {
  OpResult result = op->getResult(0);
  kokkos::MemorySpace space = kokkos::getMemSpace(result);
  if(failed(emitter.emitValue(op.getMemref())))
    return failure();
  if(space == kokkos::MemorySpace::DualView) {
    // For DualView, call method to free both host and device views
    emitter << ".deallocate()";
  }
  else {
    // For Kokkos::View, free by assigning a default-constructed instance
    emitter << " = ";
    if(failed(emitter.emitMemrefType(op.getLoc(), dyn_cast<MemRefType>(op.getMemref().getType()), space)))
      return failure();
    emitter << "()";
  }
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    memref::DimOp op) {
  if (failed(emitter.emitAssignPrefix(*op)))
    return failure();
  emitter << emitter.getOrCreateName(op.getSource());
  emitter << ".extent(";
  if(op.getConstantIndex())
    emitter << *op.getConstantIndex();
  else
    emitter << emitter.getOrCreateName(op.getIndex());
  emitter << ")";
  return success();
}

/*
static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    memref::ReinterpretCastOp op) {
  Value result = op.getResult();
  auto sourceType = dyn_cast<MemRefType>(op.getSource().getType());
  emitter.memrefSpaces[result] = emitter.memrefSpaces[op.getSource()];
  int sourceRank = sourceType.getRank();
  if (sourceRank != 1 || (emitter.isStridedSubview(op.getSource())) ||
      llvm::any_of(op.getStaticStrides(),
                   [](auto stride) { return stride != 1; })) {
    return op.emitError(
        "ReinterpretCastOp has these restrictions (a) 1D memrefs (b) no "
        "strided input or output (c) staically known size (d) unit stride");
  }
  const bool useDynamicSizes = !op.getSizes().empty();
  const bool useDynamicOffsets = !op.getOffsets().empty();
  auto emitSize = [&](int dim) -> LogicalResult {
    if (useDynamicSizes) {
      if (failed(emitter.emitValue(op.getSizes()[dim])))
        return failure();
    } else
      emitter << op.getStaticSizes()[dim];
    return success();
  };
  auto emitOffset = [&](int dim) -> LogicalResult {
    if (useDynamicOffsets) {
      if (failed(emitter.emitValue(op.getOffsets()[dim])))
        return failure();
    } else
      emitter << op.getStaticOffsets()[dim];
    return success();
  };
  emitter << "auto " << emitter.getOrCreateName(result)
          << " = Kokkos::subview(";
  if (failed(emitter.emitValue(op.getSource())))
    return failure();
  for (int dim = 0; dim < sourceRank; dim++) {
    emitter << ", Kokkos::make_pair<int64_t, int64_t>(";
    // interval for each dim is [offset, offset+size)
    // TODO: this is only correct for unit strides, see above
    if (failed(emitOffset(dim)))
      return failure();
    emitter << ", ";
    if (failed(emitOffset(dim)))
      return failure();
    emitter << " + ";
    if (failed(emitSize(dim)))
      return failure();
    emitter << ")";
  }
  emitter << ")";
  return success();
}
*/

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    emitc::CallOp op) {
  if (failed(emitter.emitAssignPrefix(*op)))
    return failure();
  emitter << op.getCallee();
  emitter << "(";
  if (failed(emitter.emitOperands(*op)))
    return failure();
  emitter << ")";
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    emitc::CallOpaqueOp callOpaqueOp) {
  Operation &op = *callOpaqueOp.getOperation();
  auto& os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(op)))
    return failure();
  emitter << callOpaqueOp.getCallee();

  auto emitArgs = [&](Attribute attr) -> LogicalResult {
    if (auto t = dyn_cast<IntegerAttr>(attr)) {
      // Index attributes are treated specially as operand index.
      if (t.getType().isIndex()) {
        int64_t idx = t.getInt();
        Value operand = op.getOperand(idx);
        if (!emitter.hasValueInScope(operand))
          return op.emitOpError("operand ")
                 << idx << "'s value not defined in scope";
        emitter << emitter.getOrCreateName(operand);
        return success();
      }
    }
    if (failed(emitter.emitAttribute(op.getLoc(), attr)))
      return failure();

    return success();
  };

  if (callOpaqueOp.getTemplateArgs()) {
    emitter << "<";
    if (failed(interleaveCommaWithError(*callOpaqueOp.getTemplateArgs(), os,
                                        emitArgs)))
      return failure();
    emitter << ">";
  }

  emitter << "(";

  LogicalResult emittedArgs =
      callOpaqueOp.getArgs()
          ? interleaveCommaWithError(*callOpaqueOp.getArgs(), os, emitArgs)
          : emitter.emitOperands(op);
  if (failed(emittedArgs))
    return failure();
  emitter << ")";
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    memref::StoreOp op) {
  emitter << emitter.getOrCreateName(op.getMemref());
  if(kokkos::getMemSpace(op.getMemref()) == kokkos::MemorySpace::DualView) {
    // Which view to access depends if we are in host or device context
    if(kokkos::getOpExecutionSpace(op) == kokkos::ExecutionSpace::Device)
      emitter << ".device_view";
    else
      emitter << ".host_view";
  }
  emitter << "(";
  for(auto iter = op.getIndices().begin(); iter != op.getIndices().end(); iter++)
  {
    if(iter != op.getIndices().begin())
      emitter << ", ";
    if(failed(emitter.emitValue(*iter)))
      return failure();
  }
  emitter << ") = ";
  if(failed(emitter.emitValue(op.getValue())))
    return failure();
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    memref::LoadOp op) {
  //TODO: if in host code, use a mirror view?
  if(failed(emitter.emitType(op.getLoc(), op.getResult().getType())))
    return op.emitError("Failed to emit LoadOp result type");
  emitter << ' ' << emitter.getOrCreateName(op.getResult()) << " = ";
  if(failed(emitter.emitValue(op.getMemRef())))
    return op.emitError("Failed to emit the LoadOp's memref value");
  if(kokkos::getMemSpace(op.getMemref()) == kokkos::MemorySpace::DualView) {
    // Which view to access depends if we are in host or device context
    if(kokkos::getOpExecutionSpace(op) == kokkos::ExecutionSpace::Device)
      emitter << ".device_view";
    else
      emitter << ".host_view";
  }
  emitter << "(";
  for(auto iter = op.getIndices().begin(); iter != op.getIndices().end(); iter++)
  {
    if(iter != op.getIndices().begin())
      emitter << ", ";
    if(failed(emitter.emitValue(*iter)))
      return op.emitError("Failed to emit a LoadOp index");
  }
  emitter << ")";
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    memref::CopyOp op) {
  // TODO: deal with case where source and target have different spaces and different layouts.
  emitter << "Kokkos::deep_copy(";
  if(failed(emitter.emitValue(op.getTarget())))
    return failure();
  emitter << ", ";
  if(failed(emitter.emitValue(op.getSource())))
    return failure();
  emitter << ")";
  return success();
}

//static LogicalResult printOperation(KokkosCppEmitter &emitter,
//                                    memref::SubViewOp op) {
//  Value result = op.getResult();
//  auto sourceType = dyn_cast<MemRefType>(op.getSource().getType());
//  // The subview will have the same space as the parent
//  emitter.memrefSpaces[result] = emitter.memrefSpaces[op.getSource()];
//  int sourceRank = sourceType.getRank();
//  if(emitter.isStridedSubview(op.getSource()))
//  {
//    return op.emitError("NOT SUPPORTED YET: strided subview of strided subview. Would need to figure out how to get extents correct");
//  }
//  //SubViewOp behavior
//  //  - Sizes/Offsets are required
//  //    - dynamic must take precedence over static, because sometimes both dynamic and static offsets are given but the static values are all -1.
//  //  - Strides are optional - if neither static nor dynamic strides are given, assume they are all 1
//  //  - Check the op.getDroppedDims() for a bit vector of which dimensions are kept (1) or discarded (0).
//  //    - This works just like the Kokkos::subview arguments - kept dims are given as a [begin, end) interval, and discarded as a single index.
//  emitter << "/* SubViewOp info\n";
//  emitter << "Source (rank-" << sourceRank << "): ";
//  if(failed(emitter.emitValue(op.getSource())))
//    return failure();
//  emitter << "\n";
//  emitter << "Sizes (dynamic): ";
//  for(auto s : op.getSizes())
//  {
//    (void) emitter.emitValue(s);
//    emitter << "  ";
//  }
//  emitter << "\n";
//  emitter << "Offsets (dynamic): ";
//  for(auto o : op.getOffsets())
//  {
//    (void) emitter.emitValue(o);
//    emitter << "  ";
//  }
//  emitter << "\n";
//  emitter << "Strides (dynamic): ";
//  for(auto s : op.getStrides())
//  {
//    (void) emitter.emitValue(s);
//    emitter << "  ";
//  }
//  emitter << "\n";
//  emitter << "Sizes (static): ";
//  for(auto s : op.getStaticSizes())
//  {
//    emitter << s << "  ";
//  }
//  emitter << "\n";
//  emitter << "Offsets (static): ";
//  for(auto o : op.getStaticOffsets())
//  {
//    emitter << o << "  ";
//  }
//  emitter << "\n";
//  emitter << "Strides (static): ";
//  for(auto s : op.getStaticStrides())
//  {
//    emitter << s << "  ";
//  }
//  emitter << "\n";
//  emitter << "Dropped Dims: ";
//  for(size_t i = 0; i < op.getDroppedDims().size(); i++)
//    emitter << op.getDroppedDims()[i] << " ";
//  emitter << "\n*/\n";
//  // TODO: what happens with sizes/staticSizes when a dimension is dropped?
//  // Is the size for dropped dimension just 1, or is the length of sizes one element shorter per dropped dim?
//  for(size_t i = 0; i < op.getDroppedDims().size(); i++)
//  {
//    if(op.getDroppedDims()[i])
//      return op.emitError("Do not yet support dropped dimensions in SubViewOp, see TODO.");
//  }
//  bool useDynamicSizes = op.getSizes().size();
//  bool useDynamicOffsets = op.getOffsets().size();
//  // TODO: need to carefully implement subviews with non-unit strides,
//  // since Kokkos::subview does not support this. Probably have to compute the strides per-dimension and then construct a LayoutStride object.
//  bool haveStrides = false;
//  if(op.getStrides().size())
//  {
//    //dynamic strides given, so they could be non-unit
//    haveStrides = true;
//  }
//  for(auto staticStride : op.getStaticStrides())
//  {
//    if(staticStride != 1)
//      haveStrides = true;
//  }
//  if(haveStrides)
//    return op.emitError("Do not yet support non-unit strides in SubViewOp, see TODO.");
//  auto emitSize = [&](int dim) -> LogicalResult
//  {
//    if(useDynamicSizes)
//    {
//      if(failed(emitter.emitValue(op.getSizes()[dim])))
//        return failure();
//    }
//    else
//      emitter << op.getStaticSizes()[dim];
//    return success();
//  };
//  auto emitOffset = [&](int dim) -> LogicalResult
//  {
//    if(useDynamicOffsets)
//    {
//      if(failed(emitter.emitValue(op.getOffsets()[dim])))
//        return failure();
//    }
//    else
//      emitter << op.getStaticOffsets()[dim];
//    return success();
//  };
//  emitter << "auto " << emitter.getOrCreateName(result) << " = Kokkos::subview(";
//  if(failed(emitter.emitValue(op.getSource())))
//    return failure();
//  for(int dim = 0; dim < sourceRank; dim++)
//  {
//    emitter << ", Kokkos::make_pair<int64_t, int64_t>(";
//    //interval for each dim is [offset, offset+size)
//    //TODO: this is only correct for unit strides, see above
//    if(failed(emitOffset(dim)))
//      return failure();
//    emitter << ", ";
//    if(failed(emitOffset(dim)))
//      return failure();
//    emitter << " + ";
//    if(failed(emitSize(dim)))
//      return failure();
//    emitter << ")";
//  }
//  emitter << ")";
//  return success();
//}

/*
static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    memref::CollapseShapeOp op) {
  //CollapseShape flattens a subset of dimensions. The op semantics require that this can alias the existing data without copying/shuffling.
  //TODO: expressing this with unmanaged view, so need to manage the lifetime of the owning view so it doesn't end up with a dangling pointer.
  //MemRefType srcType = op.src().getType().cast<MemRefType>();
  MemRefType dstType = op.getResult().getType().cast<MemRefType>();
  emitter.memrefSpaces[op.getResult()] = emitter.memrefSpaces[op.getSrc()];
  if(failed(emitter.emitType(op.getLoc(), dstType)))
    return failure();
  emitter << ' ' << emitter.getOrCreateName(op.getResult());
  //TODO: handle dynamic dimensions here
  emitter << '(';
  if(failed(emitter.emitValue(op.getSrc())))
    return failure();
  emitter << ".data())";
  return success();
}
*/

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    memref::CastOp op) {
  // CastOp can only convert between static and dynamic dimensions.
  // For Kokkos views and LAPIS::DualView, this has no real effects.
  emitter.assignName(emitter.getOrCreateName(op.getOperand()), op.getResult());
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    arith::ConstantOp constantOp) {
  //Register the constant with the emitter so that it can replace usage of this variable with 
  //an equivalent literal. Don't need to declare the actual SSA variable.
  emitter.registerScalarConstant(constantOp.getResult(), constantOp);
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    arith::IndexCastOp indexCastOp) {
  if(failed(emitter.emitType(indexCastOp.getLoc(), indexCastOp.getOut().getType())))
    return failure();
  emitter << ' ' << emitter.getOrCreateName(indexCastOp.getOut()) << " = ";
  if(failed(emitter.emitValue(indexCastOp.getIn())))
    return failure();
  emitter << ";\n";
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    arith::SIToFPOp sIToFPOp) {
  if(failed(emitter.emitType(sIToFPOp.getLoc(), sIToFPOp.getOut().getType())))
    return failure();
  emitter << ' ' << emitter.getOrCreateName(sIToFPOp.getOut()) << " = ";
  if(failed(emitter.emitValue(sIToFPOp.getIn())))
    return failure();
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    arith::MinNumFOp op) {
  if(failed(emitter.emitAssignPrefix(*op)))
    return failure();
  emitter << "Kokkos::isnan(";
  if(failed(emitter.emitValue(op.getLhs())))
    return failure();
  emitter << ") ? ";
  if(failed(emitter.emitValue(op.getRhs())))
    return failure();
  emitter << " : Kokkos::min(";
  if(failed(emitter.emitValue(op.getLhs())))
    return failure();
  emitter << ", ";
  if(failed(emitter.emitValue(op.getRhs())))
    return failure();
  emitter << ")";
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    arith::MaxNumFOp op) {
  if(failed(emitter.emitAssignPrefix(*op)))
    return failure();
  emitter << "Kokkos::isnan(";
  if(failed(emitter.emitValue(op.getLhs())))
    return failure();
  emitter << ") ? ";
  if(failed(emitter.emitValue(op.getRhs())))
    return failure();
  emitter << " : Kokkos::max(";
  if(failed(emitter.emitValue(op.getLhs())))
    return failure();
  emitter << ", ";
  if(failed(emitter.emitValue(op.getRhs())))
    return failure();
  emitter << ")";
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    arith::FPToUIOp op) {
  //In C, float->unsigned conversion when input is negative is implementation defined, but MLIR says it should convert to the nearest value (0)
  if(failed(emitter.emitType(op.getLoc(), op.getOut().getType())))
    return failure();
  emitter << ' ' << emitter.getOrCreateName(op.getOut()) << " = (";
  if(failed(emitter.emitValue(op.getIn())))
    return failure();
  emitter << " <= 0.f) ?";
  emitter << "0U : (";
  if(failed(emitter.emitType(op.getLoc(), op.getOut().getType())))
    return failure();
  emitter << ") ";
  if(failed(emitter.emitValue(op.getIn())))
    return failure();
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    func::ConstantOp constantOp) {
  Operation *operation = constantOp.getOperation();
  Attribute value = constantOp.getValueAttr();
  return printConstantOp(emitter, operation, value);
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    cf::AssertOp op) {
  emitter << "if(!" << emitter.getOrCreateName(op.getArg()) << ") Kokkos::abort(";
  if(failed(emitter.emitAttribute(op.getLoc(), op.getMsgAttr())))
    return failure();
  emitter << ")";
  return success();
}

static LogicalResult printSupportCall(KokkosCppEmitter &emitter, func::CallOp callOp)
{
  // NOTE: do not currently support multiple return values (a tuple) from support functions,
  // but this is OK as none of them return more than 1 value.
  if(callOp.getResults().size() > 1)
    return callOp.emitError("Can't handle support function with multiple results");
  bool pointerResults = emitter.sparseSupportFunctionPointerResults(callOp.getCallee());
  raw_indented_ostream &os = emitter.ostream();
  // Declare the result (if any) in current scope
  bool hasResult = callOp.getResults().size() == 1;
  bool resultIsMemref = hasResult && isa<MemRefType>(callOp.getResult(0).getType());
  kokkos::MemorySpace resultSpace = kokkos::MemorySpace::Host;
  if (hasResult) {
    // Register the space of the result as being HostSpace now
    if (resultIsMemref) {
      resultSpace = kokkos::getMemSpace(callOp.getResult(0));
      if(failed(emitter.emitMemrefType(callOp.getLoc(), dyn_cast<MemRefType>(callOp.getResult(0).getType()), resultSpace)))
        return failure();
    } else {
      if(failed(emitter.emitType(callOp.getLoc(), callOp.getResult(0).getType(), false)))
        return failure();
    }
    os << ' ' << emitter.getOrCreateName(callOp.getResult(0)) << ";\n";
  }
  os << "{\n";
  os.indent();
  // If the result is a memref, it is returned via pointer.
  // Declare the StridedMemRefType version here in a local scope.
  if(resultIsMemref)
  {
    if(failed(emitter.emitType(callOp.getLoc(), callOp.getResult(0).getType(), true)))
      return failure();
    os << " " << emitter.getOrCreateName(callOp.getResult(0)) << "_smr;\n";
  }
  // Now go through all the input arguments and convert memrefs to StridedMemRefTypes as well.
  // Because the same Value can be used for multiple arguments, do not create more than one
  // StridedMemRef version per Value.
  llvm::DenseSet<Value> convertedStridedMemrefs;
  for(Value arg : callOp.getOperands())
  {
    if(isa<MemRefType>(arg.getType()))
    {
      if(convertedStridedMemrefs.contains(arg))
        continue;
      if(failed(emitter.emitType(callOp.getLoc(), arg.getType(), true)))
        return failure();
      os << " " << emitter.getOrCreateName(arg) << "_smr = LAPIS::viewToStridedMemref(";
      os << emitter.getOrCreateName(arg);
      os << ");\n";
      convertedStridedMemrefs.insert(arg);
    }
  }
  // Finally, emit the call.
  if(hasResult && !pointerResults)
  {
    os << emitter.getOrCreateName(callOp.getResult(0)) << " = ";
  }
  os << emitter.getSparseSupportFunctionName(callOp.getCallee()) << "(";
  // Pointer to the result is first argument, if pointerResult and there is a result
  if(hasResult && pointerResults) {
    if(resultIsMemref)
      os << "&" << emitter.getOrCreateName(callOp.getResult(0)) << "_smr";
    else
      os << "&" << emitter.getOrCreateName(callOp.getResult(0));
    if(callOp.getOperands().size())
      os << ", ";
  }
  // Emit the input arguments, passing in _smr suffixed values for memrefs
  bool firstArg = true;
  for(Value arg : callOp.getOperands()) {
    if(!firstArg)
      os << ", ";
    if(isa<MemRefType>(arg.getType()))
      os << "&" << emitter.getOrCreateName(arg) << "_smr";
    else {
      if(failed(emitter.emitValue(arg)))
        return failure();
    }
    firstArg = false;
  }
  os << ");\n";
  // Lastly, if result is a memref, convert it back to View
  if(hasResult && resultIsMemref) {
    // stridedMemrefToView produces an unmanaged host view.
    // If result is a DualView, wrap it in a DualView here.
    os << emitter.getOrCreateName(callOp.getResult(0)) << " = ";
    if(resultSpace == kokkos::MemorySpace::DualView) {
      // The result is a DualView and came from a support function - add it to the list
      emitter.dualViewsFromTensors.push_back(callOp.getResult(0));
      //if(failed(emitter.emitMemrefType(callOp.getLoc(), dyn_cast<MemRefType>(callOp.getResult(0).getType()), resultSpace)))
      //  return failure();
      //os << "(";
    }
    os << "LAPIS::stridedMemrefToView<";
    if(failed(emitter.emitMemrefType(callOp.getLoc(), dyn_cast<MemRefType>(callOp.getResult(0).getType()), kokkos::MemorySpace::Host)))
      return failure();
    os << ">(" << emitter.getOrCreateName(callOp.getResult(0)) << "_smr)";
    //if(resultSpace == kokkos::MemorySpace::DualView)
    //  os << ")";
    os << ";\n";
  }
  os.unindent();
  os << "}\n";
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter, func::CallOp callOp) {
  if(emitter.isSparseSupportFunction(callOp.getCallee()))
    return printSupportCall(emitter, callOp);

  if (failed(emitter.emitAssignPrefix(*callOp.getOperation())))
    return failure();

  raw_ostream &os = emitter.ostream();
  os << callOp.getCallee();
  os << "(";
  if (failed(emitter.emitOperands(*callOp.getOperation())))
    return failure();
  os << ")";
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter, scf::ForOp forOp) {

  raw_indented_ostream &os = emitter.ostream();

  OperandRange operands = forOp.getInitArgs();
  Block::BlockArgListType iterArgs = forOp.getRegionIterArgs();
  Operation::result_range results = forOp.getResults();

  for (OpResult result : results) {
    if (failed(emitter.emitVariableDeclaration(result,
                                               /*trailingSemicolon=*/true)))
      return failure();
  }

  for (auto pair : llvm::zip(iterArgs, operands)) {
    if (failed(emitter.emitType(forOp.getLoc(), std::get<0>(pair).getType())))
      return failure();
    os << " " << emitter.getOrCreateName(std::get<0>(pair)) << " = ";
    if(failed(emitter.emitValue(std::get<1>(pair))))
      return failure();
    os << ";";
    os << "\n";
  }

  os << "for (";
  if (failed(
          emitter.emitType(forOp.getLoc(), forOp.getInductionVar().getType())))
    return failure();
  os << " ";
  os << emitter.getOrCreateName(forOp.getInductionVar());
  os << " = ";
  if(failed(emitter.emitValue(forOp.getLowerBound())))
    return failure();
  os << "; ";
  os << emitter.getOrCreateName(forOp.getInductionVar());
  os << " < ";
  if(failed(emitter.emitValue(forOp.getUpperBound())))
    return failure();
  os << "; ";
  os << emitter.getOrCreateName(forOp.getInductionVar());
  os << " += ";
  if(failed(emitter.emitValue(forOp.getStep())))
    return failure();
  os << ") {\n";
  os.indent();

  Region &forRegion = forOp.getRegion();
  auto regionOps = forRegion.getOps();

  // We skip the trailing yield op because this updates the result variables
  // of the for op in the generated code. Instead we update the iterArgs at
  // the end of a loop iteration and set the result variables after the for
  // loop.
  for (auto it = regionOps.begin(); std::next(it) != regionOps.end(); ++it) {
    if (failed(emitter.emitOperation(*it, /*trailingSemicolon=*/true)))
      return failure();
  }

  Operation *yieldOp = forRegion.getBlocks().front().getTerminator();
  // Copy yield operands into iterArgs at the end of a loop iteration.
  for (auto pair : llvm::zip(iterArgs, yieldOp->getOperands())) {
    BlockArgument iterArg = std::get<0>(pair);
    Value operand = std::get<1>(pair);
    os << emitter.getOrCreateName(iterArg) << " = ";
    if(failed(emitter.emitValue(operand)))
      return failure();
    emitter << ";\n";
  }

  os.unindent() << "}";

  // Copy iterArgs into results after the for loop.
  for (auto pair : llvm::zip(results, iterArgs)) {
    OpResult result = std::get<0>(pair);
    BlockArgument iterArg = std::get<1>(pair);
    os << "\n"
       << emitter.getOrCreateName(result) << " = ";
    if(failed(emitter.emitValue(iterArg)))
      return failure();
    emitter << ";";
  }

  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter, scf::WhileOp whileOp) {
  //Declare the before args, after args, and results.
  for (auto pair : llvm::zip(whileOp.getBeforeArguments(), whileOp.getInits())) {
  //for (OpResult beforeArg : whileOp.getBeforeArguments()) {
    // Before args are initialized to the whileOp's "inits"
    if(failed(emitter.emitType(whileOp.getLoc(), std::get<0>(pair).getType())))
      return failure();
    emitter << ' ' << emitter.getOrCreateName(std::get<0>(pair)) << " = ";
    if(failed(emitter.emitValue(std::get<1>(pair))))
      return failure();
    emitter << ";\n";
  }
  for (auto afterArg : whileOp.getAfterArguments()) {
    if (failed(emitter.emitVariableDeclaration(afterArg, /*trailingSemicolon=*/true)))
      return failure();
  }
  for (OpResult result : whileOp.getResults()) {
    if (failed(emitter.emitVariableDeclaration(result, /*trailingSemicolon=*/true)))
      return failure();
  }

  emitter << "/*\n";
  emitter << "Hello from while op.\n";
  emitter << "Inits:\n";
  for(auto a : whileOp.getInits())
    emitter << "  " << emitter.getOrCreateName(a) << "\n";
  emitter << "Before block args:\n";
  for(auto a : whileOp.getBeforeArguments())
    emitter << "  " << emitter.getOrCreateName(a) << "\n";
  emitter << "After block args:\n";
  for(auto a : whileOp.getAfterArguments())
    emitter << "  " << emitter.getOrCreateName(a) << "\n";
  emitter << "Results:\n";
  for(auto a : whileOp.getResults())
    emitter << "  " << emitter.getOrCreateName(a) << "\n";
  emitter << "*/\n";

  emitter << "while(true) {\n";
  emitter.ostream().indent();

  //Emit the "before" block(s)
  for (auto& beforeOp : whileOp.getBefore().getOps()) {
    if (failed(emitter.emitOperation(beforeOp, /*trailingSemicolon=*/true)))
      return failure();
  }

  for (auto pair : llvm::zip(whileOp.getAfterArguments(), whileOp.getConditionOp().getArgs())) {
    // After args are initialized to the args passed by ConditionOp 
    if(failed(emitter.emitType(whileOp.getLoc(), std::get<0>(pair).getType())))
      return failure();
    emitter << ' ' << emitter.getOrCreateName(std::get<0>(pair)) << " = ";
    if(failed(emitter.emitValue(std::get<1>(pair))))
      return failure();
    emitter << ";\n";
  }

  //Emit the "after" block(s)
  for (auto& afterOp : whileOp.getAfter().getOps()) {
    if (failed(emitter.emitOperation(afterOp, /*trailingSemicolon=*/true)))
      return failure();
  }

  // Copy yield operands into before block args at the end of a loop iteration.
  for (auto pair : llvm::zip(whileOp.getBeforeArguments(), whileOp.getYieldOp()->getOperands())) {
    BlockArgument iterArg = std::get<0>(pair);
    Value operand = std::get<1>(pair);
    emitter << emitter.getOrCreateName(iterArg) << " = ";
    if(failed(emitter.emitValue(operand)))
      return failure();
    emitter << ";\n";
  }

  emitter.ostream().unindent();
  emitter << "}\n";

  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter, scf::ConditionOp condOp) {
  //The condition value should already be in scope. Just break out of loop if it's falsey.
  emitter << "if(";
  if(failed(emitter.emitValue(condOp.getCondition())))
    return failure();
  emitter << ") {\n";
  emitter << "}\n";
  emitter << "else {\n";
  //Condition false: breaking out of loop
  emitter << "break;\n";
  emitter << "}\n";
  return success();
}

// If the join represented by op is a built-in reducer in Kokkos, return true and set reduction to its
// name in C++ (e.g. "Kokkos:Min"). Otherwise return false.
static bool isBuiltinReduction(std::string& reduction, kokkos::UpdateReductionOp op) {
  // Built-in joins should have only two ops in the body: a binary arithmetic op of the two arguments, and a yield of that result.
  // Note: all Kokkos built in reductions have commutative joins, so here we test for both permutations of the arguments as operands.
  //
  // TODO! Check that op.getIdentity() matches the corresponding Kokkos reduction identity before returning true.
  // However, this should already match in all cases.
  Region& body = op.getReductionOperator();
  auto bodyArgs = body.getArguments();
  if(bodyArgs.size() != 2)
    return false;
  Value arg1 = bodyArgs[0];
  Value arg2 = bodyArgs[1];
  SmallVector<Operation*> bodyOps;
  for(Operation& op : body.getOps()) {
    bodyOps.push_back(&op);
    if(bodyOps.size() > 2)
      return false;
  }
  Operation* op1 = bodyOps[0];
  Operation* op2 = bodyOps[1];
  if(op1->getNumOperands() != 2 || op1->getNumResults() != 1)
    return false;
  // Is the second op a yield of the first op's result?
  if(!isa<kokkos::YieldOp>(op2) || op1->getResults()[0] != op2->getOperands()[0])
    return false;
  // Does op1 take bodyArgs as its two operands?
  if(!((op1->getOperands()[0] == arg1 && op1->getOperands()[1] == arg2)
      || (op1->getOperands()[0] == arg2 && op1->getOperands()[1] == arg1))) {
    return false;
  }
  Type type = op2->getOperands()[0].getType();
  // Finally, if op1 has one of the supported types, return true.
  if(isa<arith::AddFOp, arith::AddIOp>(op1)) {
    reduction = "Kokkos::Sum";
    return true;
  }
  else if(isa<arith::MulFOp, arith::MulIOp>(op1)) {
    reduction = "Kokkos::Prod";
    return true;
  }
  else if(isa<arith::AndIOp>(op1)) {
    // Note: MLIR makes no distinction between logical and bitwise AND.
    // AndIOp always behaves like a bitwise AND.
    reduction = "Kokkos::BAnd";
    return true;
  }
  else if(isa<arith::OrIOp>(op1)) {
    // Note: MLIR makes no distinction between logical and bitwise OR.
    // OrIOp always behaves like a bitwise OR.
    reduction = "Kokkos::BOr";
    return true;
  }
  // TODO for when LLVM gets updated. They have split arith.maxf into arith.maximumf and arith.maxnumf (same with minf)
  // These have different behavior with respect to NaNs: maximumf(nan, a) = nan, but maxnumf(nan, a) = a.
  // The latter is not what Kokkos::Max will do, so it would have to use a custom reducer.
  //else if(isa<arith::MaximumFOp>(op1) {
  else if(isa<arith::MaximumFOp>(op1)) {
    reduction = "Kokkos::Max";
    return true;
  }
  else if(isa<arith::MinimumFOp>(op1)) {
  //else if(isa<arith::MinimumFOp>(op1)) {
    reduction = "Kokkos::Min";
    return true;
  }
  // We can only use the Kokkos built-in for integer min/max if the signedness of the op matches the type.
  // Otherwise, we have to emit a custom reducer that casts to the correct signedness before applying min/maxc.
  else if((isa<arith::MaxSIOp>(op1) && type.isSignedInteger()) || (isa<arith::MaxUIOp>(op1) && type.isUnsignedInteger())) {
    reduction = "Kokkos::Max";
    return true;
  }
  else if((isa<arith::MinSIOp>(op1) && type.isSignedInteger()) || (isa<arith::MinUIOp>(op1) && type.isUnsignedInteger())) {
    reduction = "Kokkos::Min";
    return true;
  }
  // The reduction is some binary operation, but not one that Kokkos has as a built-in reducer.
  return false;
}

static LogicalResult printOperation(KokkosCppEmitter &emitter, kokkos::RangeParallelOp op) {
  // Declare any results (which can only be produced by reductions).
  // These don't need to be initialized.
  bool isReduction = op.getNumReductions();
  for(Value result : op->getResults())
  {
    if(failed(emitter.emitType(op.getLoc(), result.getType())))
      return failure();
    emitter << ' ' << emitter.getOrCreateName(result) << ";\n";
  }
  if(isReduction)
    emitter << "Kokkos::parallel_reduce";
  else
    emitter << "Kokkos::parallel_for";
  emitter << "(";
  bool isMDPolicy = op.getNumLoops() > 1;
  // Construct the policy, based on the level and iteration rank of op
  switch(op.getParallelLevel()) {
    case kokkos::ParallelLevel::RangePolicy:
      if(isMDPolicy)
        emitter << "Kokkos::MDRangePolicy";
      else
        emitter << "Kokkos::RangePolicy";
      break;
    case kokkos::ParallelLevel::TeamVector:
      if(isMDPolicy)
        emitter << "Kokkos::TeamVectorMDRange";
      else
        emitter << "Kokkos::TeamVectorRange";
      break;
    case kokkos::ParallelLevel::TeamThread:
      if(isMDPolicy)
        emitter << "Kokkos::TeamThreadMDRange";
      else
        emitter << "Kokkos::TeamThreadRange";
      break;
    case kokkos::ParallelLevel::ThreadVector:
      if(isMDPolicy)
        emitter << "Kokkos::ThreadVectorMDRange";
      else
        emitter << "Kokkos::ThreadVectorRange";
      break;
  }
  if(op.getParallelLevel() == kokkos::ParallelLevel::RangePolicy) {
    emitter << "<";
    if(op.getExecutionSpace() == kokkos::ExecutionSpace::Host)
      emitter << "Kokkos::DefaultHostExecutionSpace";
    emitter << ">";
  }
  emitter << "(";
  if(op.getParallelLevel() != kokkos::ParallelLevel::RangePolicy)
    emitter << "team, ";
  if(isMDPolicy) {
    emitter << "{";
    for(size_t i = 0; i < op.getNumLoops(); i++) {
      if(i > 0)
        emitter << ", ";
      emitter << "0";
    }
    emitter << "}, ";
    emitter << "{";
    int count = 0;
    for(Value bound : op.getUpperBound()) {
      if(count++)
        emitter << ", ";
      if(failed(emitter.emitValue(bound)))
        return failure();
    }
    emitter << "}, ";
  }
  else {
    // 1D RangePolicy. Requires both lower and upper bounds.
    emitter << "0, ";
    if(failed(emitter.emitValue(op.getUpperBound().front())))
      return failure();
  }
  emitter << "),\n";
  if(op.getExecutionSpace() == kokkos::ExecutionSpace::Device)
    emitter << "KOKKOS_LAMBDA(";
  else
    emitter << "[=](";
  // Declare induction variables
  int count = 0;
  for(auto iv : op.getInductionVars()) {
    if(count++)
      emitter << ", ";
    if(failed(emitter.emitType(op.getLoc(), iv.getType())))
      return failure();
    emitter << ' ' << emitter.getOrCreateName(iv);
  }
  int depth = kokkos::getOpParallelDepth(op) - 1;
  if(isReduction) {
    for(Value result : op->getResults()) {
      emitter << ", ";
      if(failed(emitter.emitType(op.getLoc(), result.getType())))
        return failure();
      emitter << "& lreduce" << depth;
    }
  }
  emitter << ") {\n";
  // Emit body ops.
  emitter.ostream().indent();
  for(Operation& bodyOp : op.getRegion().getOps()) {
    if(failed(emitter.emitOperation(bodyOp, true)))
      return failure();
  }
  emitter.ostream().unindent();
  emitter << "}";
  if(isReduction) {
    // Determine what kind of reduction is being done, if any
    kokkos::UpdateReductionOp reduction = op.getReduction();
    if(!reduction)
      return op.emitError("Could not find UpdateReductionOp inside parallel op with result(s)");
    std::string kokkosReducer;
    if(!isBuiltinReduction(kokkosReducer, reduction))
      return op.emitError("Do not yet support non-builtin reducers");
    Value result = op.getResults()[0];
    // Pass in reducer arguments to parallel_reduce
    emitter << ", " << kokkosReducer << "<";
    if(failed(emitter.emitType(op.getLoc(), result.getType())))
      return failure();
    emitter << ">(" << emitter.getOrCreateName(result) << ")";
  }
  emitter << ")";
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter, kokkos::TeamParallelOp op) {
  // Declare any results (which can only be produced by reductions).
  // These don't need to be initialized.
  bool isReduction = op.getNumReductions();
  for(Value result : op->getResults())
  {
    if(failed(emitter.emitType(op.getLoc(), result.getType())))
      return failure();
    emitter << ' ' << emitter.getOrCreateName(result) << ";\n";
  }
  std::string lambda = "lambda_" + emitter.getUniqueIdentifier();
  // First, declare the lambda.
  emitter << "auto " << lambda << " = \n";
  emitter << "KOKKOS_LAMBDA(const LAPIS::TeamMember& team";
  if(isReduction) {
    if(op->getNumResults() > 1)
      return op.emitError("Currently, can only handle 1 reducer per parallel");
    for(Value result : op->getResults()) {
      emitter << ", ";
      if(failed(emitter.emitType(op.getLoc(), result.getType())))
        return failure();
      emitter << "& lreduce0";
    }
  }
  emitter << ") {\n";
  // Within the body of this loop, replace uses of the 5 block operands with
  // the correct values from the Kokkos team handle.
  emitter.assignName("team.league_size()", op.getLeagueSize());
  emitter.assignName("team.team_size()", op.getTeamSize());
  emitter.assignName("team.league_rank()", op.getLeagueRank());
  emitter.assignName("team.team_rank()", op.getTeamRank());
  // Emit body ops.
  emitter.ostream().indent();
  for(Operation& bodyOp : op.getRegion().getOps()) {
    if(failed(emitter.emitOperation(bodyOp, true)))
      return failure();
  }
  emitter.ostream().unindent();
  emitter << "};\n";
  // Find the policy's vector length based on two runtime values: the hint operand to ThreadParallel,
  // and the maximum vector length determined by Kokkos.
  // If the hint value is 0, it means no hint was provided so arbitrarily use 8 as the target.
  // TODO: is there a better choice for this?
  std::string vectorLengthTarget = "targetVectorLength_" + emitter.getUniqueIdentifier();
  emitter << "size_t " << vectorLengthTarget << " = ";
  if(failed(emitter.emitValue(op.getVectorLengthHint())))
    return failure();
  emitter << " ? ";
  if(failed(emitter.emitValue(op.getVectorLengthHint())))
    return failure();
  emitter << " : 8;\n";
  std::string vectorLength = "vectorLength_" + emitter.getUniqueIdentifier();
  emitter << "size_t " << vectorLength << " = Kokkos::min<size_t>(";
  emitter << vectorLengthTarget << ", LAPIS::TeamPolicy::vector_length_max());\n";
  // Since we have a lambda and a vector length, we can now query a temporary TeamPolicy for the
  // best team size (if op was not given a team size hint)
  std::string teamSize = "teamSize_" + emitter.getUniqueIdentifier();
  emitter << "size_t " << teamSize << " = ";
  if(failed(emitter.emitValue(op.getTeamSizeHint())))
    return failure();
  emitter << ";\n";
  emitter << "if(" << teamSize << ") {\n";
  emitter.ostream().indent();
  // Team size hint was given, so just cap it at team_size_max
  emitter << teamSize << " = Kokkos::min<size_t>(" << teamSize << ", ";
  emitter << "LAPIS::TeamPolicy(1, 1, " << vectorLength << ").team_size_max(" << lambda << ", ";
  if(isReduction)
    emitter << "Kokkos::ParallelReduceTag{}";
  else
    emitter << "Kokkos::ParallelForTag{}";
  emitter << "));\n";
  emitter.ostream().unindent();
  emitter << "}\n";
  emitter << "else {\n";
  emitter.ostream().indent();
  emitter << teamSize << " = ";
  emitter << "LAPIS::TeamPolicy(1, 1, " << vectorLength << ").team_size_recommended(" << lambda << ", ";
  if(isReduction)
    emitter << "Kokkos::ParallelReduceTag{}";
  else
    emitter << "Kokkos::ParallelForTag{}";
  emitter << ");\n";
  emitter.ostream().unindent();
  emitter << "}\n";
  // Finally, launch the lambda with the correct policy.
  if(isReduction)
    emitter << "Kokkos::parallel_reduce";
  else
    emitter << "Kokkos::parallel_for";
  emitter << "(LAPIS::TeamPolicy(";
  if(failed(emitter.emitValue(op.getLeagueSize())))
    return failure();
  emitter << ", " << teamSize << ", " << vectorLength << "), " << lambda;
  if(isReduction) {
    // Determine what kind of reduction is being done, if any
    kokkos::UpdateReductionOp reduction = op.getReduction();
    if(!reduction)
      return op.emitError("Could not find UpdateReductionOp inside parallel op with result(s)");
    std::string kokkosReducer;
    if(!isBuiltinReduction(kokkosReducer, reduction))
      return op.emitError("Do not yet support non-builtin reducers");
    Value result = op.getResults()[0];
    // Pass in reducer arguments to parallel_reduce
    emitter << ", " << kokkosReducer << "<";
    if(failed(emitter.emitType(op.getLoc(), result.getType())))
      return failure();
    emitter << ">(" << emitter.getOrCreateName(result) << ")";
  }
  emitter << ")";
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter, kokkos::ThreadParallelOp op) {
  // Declare any results (which can only be produced by reductions).
  // These don't need to be initialized.
  bool isReduction = op.getNumReductions();
  for(Value result : op->getResults())
  {
    if(failed(emitter.emitType(op.getLoc(), result.getType())))
      return failure();
    emitter << ' ' << emitter.getOrCreateName(result) << ";\n";
  }
  std::string lambda = "lambda_" + emitter.getUniqueIdentifier();
  // First, declare the lambda.
  emitter << "auto " << lambda << " = \n";
  emitter << "KOKKOS_LAMBDA(const LAPIS::TeamMember& team";
  if(isReduction) {
    if(op->getNumResults() > 1)
      return op.emitError("Currently, can only handle 1 reducer per parallel");
    for(Value result : op->getResults()) {
      emitter << ", ";
      if(failed(emitter.emitType(op.getLoc(), result.getType())))
        return failure();
      emitter << "& lreduce0";
    }
  }
  emitter << ") {\n";
  // Compute the outer induction variable in terms of league rank, team rank and team size.
  emitter << "  size_t induction = team.league_rank() * team.team_size() + team.team_rank();\n";
  // And exit immediately if this thread has nothing to do.
  // This is OK because ThreadParallel can't contain any team-wide synchronization.
  emitter << "  if(induction >= ";
  if(failed(emitter.emitValue(op.getNumIters())))
    return failure();
  emitter << ") return;\n";
  emitter.assignName("induction", op.getInductionVar());
  // Emit body ops.
  emitter.ostream().indent();
  for(Operation& bodyOp : op.getRegion().getOps()) {
    if(failed(emitter.emitOperation(bodyOp, true)))
      return failure();
  }
  emitter.ostream().unindent();
  emitter << "};\n";
  // Find the policy's vector length based on two runtime values: the hint operand to ThreadParallel,
  // and the maximum vector length determined by Kokkos.
  // If the hint value is 0, it means no hint was provided so arbitrarily use 8 as the target.
  // TODO: is there a better choice for this?
  std::string vectorLengthTarget = "targetVectorLength_" + emitter.getUniqueIdentifier();
  emitter << "size_t " << vectorLengthTarget << " = ";
  if(failed(emitter.emitValue(op.getVectorLengthHint())))
    return failure();
  emitter << " ? ";
  if(failed(emitter.emitValue(op.getVectorLengthHint())))
    return failure();
  emitter << " : 8;\n";
  std::string vectorLength = "vectorLength_" + emitter.getUniqueIdentifier();
  emitter << "size_t " << vectorLength << " = Kokkos::min<size_t>(";
  emitter << vectorLengthTarget << ", LAPIS::TeamPolicy::vector_length_max());\n";
  // Since we have a lambda and a vector length, we can now query a temporary TeamPolicy for the best team size
  std::string teamSize = "teamSize_" + emitter.getUniqueIdentifier();
  emitter << "size_t " << teamSize << " = LAPIS::TeamPolicy(1, 1, " << vectorLength << ").team_size_recommended(" << lambda << ", ";
  if(isReduction)
    emitter << "Kokkos::ParallelReduceTag{}";
  else
    emitter << "Kokkos::ParallelForTag{}";
  emitter << ");\n";
  // Get league size from team size and number of outer iters op performs
  std::string leagueSize = "leagueSize_" + emitter.getUniqueIdentifier();
  emitter << "size_t " << leagueSize << " = (";
  if(failed(emitter.emitValue(op.getNumIters())))
    return failure();
  emitter << " + " << teamSize << " - 1) / " << teamSize << ";\n";
  // Finally, launch the lambda with the correct policy.
  if(isReduction)
    emitter << "Kokkos::parallel_reduce";
  else
    emitter << "Kokkos::parallel_for";
  emitter << "(LAPIS::TeamPolicy(" << leagueSize << ", " << teamSize << ", " << vectorLength << "), " << lambda;
  if(isReduction) {
    // Determine what kind of reduction is being done, if any
    kokkos::UpdateReductionOp reduction = op.getReduction();
    if(!reduction)
      return op.emitError("Could not find UpdateReductionOp inside parallel op with result(s)");
    std::string kokkosReducer;
    if(!isBuiltinReduction(kokkosReducer, reduction))
      return op.emitError("Do not yet support non-builtin reducers");
    Value result = op.getResults()[0];
    // Pass in reducer arguments to parallel_reduce
    emitter << ", " << kokkosReducer << "<";
    if(failed(emitter.emitType(op.getLoc(), result.getType())))
      return failure();
    emitter << ">(" << emitter.getOrCreateName(result) << ")";
  }
  emitter << ")";
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter, kokkos::TeamBarrierOp op) {
  // note: the name "team" is hardcoded in the emitter for TeamParallelOp.
  // The team handle is not represented as a Value anywhere in the IR.
  emitter << "team.team_barrier()";
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter, kokkos::SingleOp op) {
  // If op has results (broadcasted values) declare them now.
  // This assumes that any types being broadcast are plain-old-data.
  for(Value result : op->getResults())
  {
    if(failed(emitter.emitType(op.getLoc(), result.getType())))
      return failure();
    emitter << ' ' << emitter.getOrCreateName(result) << ";\n";
  }
  emitter << "Kokkos::single(";
  switch(op.getLevel())
  {
    case kokkos::SingleLevel::PerTeam:
      emitter << "Kokkos::PerTeam"; break;
    case kokkos::SingleLevel::PerThread:
      emitter << "Kokkos::PerThread"; break;
  }
  emitter << "(team), [&](";
  // prefix output arguments with "l" to follow common Kokkos convention
  int count = 0;
  for(Value result : op->getResults()) {
    if(count++) emitter << ", ";
    if(failed(emitter.emitType(op.getLoc(), result.getType())))
      return failure();
    emitter << "& l" << emitter.getOrCreateName(result);
  }
  emitter << ") {\n";
  emitter.ostream().indent();
  // Now emit the statements inside the single body.
  // kokkos.yield has special behavior: assigns to the local broadcast value(s)
  for(Operation& bodyOp : op.getRegion().getOps()) {
    if(auto yield = dyn_cast<kokkos::YieldOp>(bodyOp)) {
      // Assign yield operands to the output arguments one-to-one
      for(auto p : llvm::zip(op->getResults(), yield.getOperands())) {
        Value result = std::get<0>(p);
        Value operand = std::get<1>(p);
        emitter << "l" << emitter.getOrCreateName(result) << " = " << emitter.getOrCreateName(operand) << ";\n";
      }
    }
    else {
      if(failed(emitter.emitOperation(bodyOp, true)))
        return failure();
    }
  }
  emitter.ostream().unindent();
  emitter << "})";
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter, kokkos::UpdateReductionOp op) {
  std::string kokkosReducer;
  bool isBuiltin = isBuiltinReduction(kokkosReducer, op);
  // TODO: for non-builtin reductions, declare a reducer type into declOS (with operator+ overloaded)
  // and wrap result and partial reduction in that.
  if(!isBuiltin)
    return op.emitError("Currently, emitter can only handle reductions that are built-in to Kokkos");
  //auto& declOS = emitter.decl_ostream();
  // Get the depth of the enclosing parallel, which determines the name of the local reduction value
  int depth = kokkos::getOpParallelDepth(op) - 1;
  std::string partialReduction = std::string("lreduce") + std::to_string(depth);
  Value contribute = op.getUpdate();
  Region& body = op.getReductionOperator();
  // Body has 2 arguments
  Value arg1 = body.getArguments()[0];
  Value arg2 = body.getArguments()[1];
  // Within the body of op, replace arg1 with the name partialReduction
  emitter.assignName(partialReduction, arg1);
  // and arg2 with the name of the value to contribute
  emitter.assignName(emitter.getOrCreateName(contribute), arg2);
  // Now emit the ops of the body. When yield is encountered, just assign the operand to partialReduction.
  for(Operation& bodyOp : body.getOps()) {
    if(auto yield = dyn_cast<kokkos::YieldOp>(bodyOp)) {
      emitter << partialReduction << " = " << emitter.getOrCreateName(bodyOp.getOperands()[0]) << ";\n";
    }
    else {
      if(failed(emitter.emitOperation(bodyOp, true)))
        return failure();
    }
  }
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter, kokkos::SyncOp op) {
  emitter << emitter.getOrCreateName(op.getView()) << ".sync";
  if(op.getMemorySpace() == kokkos::MemorySpace::Host)
    emitter << "Host()";
  else
    emitter << "Device()";
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter, kokkos::ModifyOp op) {
  emitter << emitter.getOrCreateName(op.getView()) << ".modify";
  if(op.getMemorySpace() == kokkos::MemorySpace::Host)
    emitter << "Host()";
  else
    emitter << "Device()";
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter, kokkos::YieldOp op) {
  // YieldOp in general doesn't do anything.
  // In contexts where it does (e.g. UpdateReductionOp), it will be handled there.
  return success();
}

/// Matches a block containing a "simple" reduction. The expected shape of the
/// block is as follows.
///
///   ^bb(%arg0, %arg1):
///     %0 = OpTy(%arg0, %arg1)
///     scf.reduce.return %0
template <typename... OpTy>
static bool matchSimpleReduction(Block &block) {
  if (block.empty() || llvm::hasSingleElement(block) ||
      std::next(block.begin(), 2) != block.end())
    return false;

  if (block.getNumArguments() != 2)
    return false;

  SmallVector<Operation *, 4> combinerOps;
  Value reducedVal = matchReduction({block.getArguments()[1]},
                                    /*redPos=*/0, combinerOps);

  if (!reducedVal || !reducedVal.isa<BlockArgument>() ||
      combinerOps.size() != 1)
    return false;

  return isa<OpTy...>(combinerOps[0]) &&
         isa<scf::ReduceReturnOp>(block.back()) &&
         block.front().getOperands() == block.getArguments();
}

/*
static LogicalResult printOperation(KokkosCppEmitter &emitter, scf::ReduceOp reduceOp) {
  raw_indented_ostream &os = emitter.ostream();

  Block &reduction = reduceOp.getRegion().front();

  Operation *terminator = &reduceOp.getRegion().front().back();
  assert(isa<scf::ReduceReturnOp>(terminator) &&
         "expected reduce op to be terminated by redure return");

  if (matchSimpleReduction<arith::AddFOp, LLVM::FAddOp>(reduction)) {
    os << emitter.getOrCreateName(reduceOp.getRegion().getOps().begin()->getResult(0));
    os << " += ";
    os << emitter.getOrCreateName(reduceOp.getOperand());
    return success();
  }

  return failure();
}
*/

static LogicalResult printOperation(KokkosCppEmitter &emitter, scf::IfOp ifOp) {
  raw_indented_ostream &os = emitter.ostream();

  for (OpResult result : ifOp.getResults()) {
    if (failed(emitter.emitVariableDeclaration(result,
                                               /*trailingSemicolon=*/true)))
      return failure();
  }

  os << "if (";
  if (failed(emitter.emitOperands(*ifOp.getOperation())))
    return failure();
  os << ") {\n";
  os.indent();

  Region &thenRegion = ifOp.getThenRegion();
  for (Operation &op : thenRegion.getOps()) {
    // Note: This prints a superfluous semicolon if the terminating yield op has
    // zero results.
    if (failed(emitter.emitOperation(op, /*trailingSemicolon=*/true)))
      return failure();
  }

  os.unindent() << "}";

  Region &elseRegion = ifOp.getElseRegion();
  if (!elseRegion.empty()) {
    os << " else {\n";
    os.indent();

    for (Operation &op : elseRegion.getOps()) {
      // Note: This prints a superfluous semicolon if the terminating yield op
      // has zero results.
      if (failed(emitter.emitOperation(op, /*trailingSemicolon=*/true)))
        return failure();
    }

    os.unindent() << "}";
  }

  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter, scf::YieldOp yieldOp) {
  raw_ostream &os = emitter.ostream();
  Operation &parentOp = *yieldOp.getOperation()->getParentOp();

/*
  if (yieldOp.getNumOperands() != parentOp.getNumResults()) {
    return yieldOp.emitError("number of operands does not to match the number "
                             "of the parent op's results");
  }
*/

  if (failed(interleaveWithError(
          llvm::zip(parentOp.getResults(), yieldOp.getOperands()),
          [&](auto pair) -> LogicalResult {
            auto result = std::get<0>(pair);
            auto operand = std::get<1>(pair);
            os << emitter.getOrCreateName(result) << " = ";

            if (!emitter.hasValueInScope(operand))
              return yieldOp.emitError("operand value not in scope");
            os << emitter.getOrCreateName(operand);
            return success();
          },
          [&]() { os << ";\n"; })))
    return failure();

  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    func::ReturnOp returnOp) {
  // First, sync all tensor-owned DualViews with
  // lifetimes inside this function to host.
  for(auto dv : emitter.dualViewsFromTensors) {
    emitter << emitter.getOrCreateName(dv) << ".syncHost();\n";
  }
  raw_ostream &os = emitter.ostream();
  os << "return";
  switch (returnOp.getNumOperands()) {
  case 0:
    return success();
  case 1:
    os << " " << emitter.getOrCreateName(returnOp.getOperand(0));
    return success(emitter.hasValueInScope(returnOp.getOperand(0)));
  default:
    os << " std::make_tuple(";
    if (failed(emitter.emitOperandsAndAttributes(*returnOp.getOperation())))
      return failure();
    os << ")";
    return success();
  }
}

static LogicalResult printOperation(KokkosCppEmitter &emitter, ModuleOp moduleOp) {
  KokkosCppEmitter::Scope scope(emitter);

  for (Operation &op : moduleOp) {
    if (failed(emitter.emitOperation(op, /*trailingSemicolon=*/false)))
      return failure();
  }
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter, func::FuncOp functionOp) {
  // Clear dualViewsFromTensors of any values from previously emitted functions.
  emitter.dualViewsFromTensors.clear();
  // Need to replace function names in 2 cases:
  //  1. functionOp is a forward declaration for a sparse support function
  //  2. functionOp's name is "main"
  bool isSupportFunc = emitter.isSparseSupportFunction(functionOp.getName());
  // Does the function provide its results via pointer arguments, preceding the input arguments?
  bool pointerResults = isSupportFunc && emitter.sparseSupportFunctionPointerResults(functionOp.getName());
  std::string functionOpName;
  if(isSupportFunc) {
    functionOpName = emitter.getSparseSupportFunctionName(functionOp.getName());
  }
  else if (functionOp.getName().str() == "main") {
    functionOpName = "mymain";
  }
  else {
    functionOpName = functionOp.getName().str();
  }
  // We need to declare variables at top if the function has multiple blocks.
  if (functionOp.getBlocks().size() > 1) {
    return functionOp.emitOpError(
        "with multiple blocks needs variables declared at top");
  }
  raw_indented_ostream &os = emitter.ostream();
  // Handle function declarations (empty body). Don't need to give parameters names either.
  if(functionOp.getBody().empty())
  {
    //Prevent support lib function names from being mangled
    if(isSupportFunc)
    {
      os << "#ifndef PYTACO_CPP_DRIVER\n";
      os << "extern \"C\" ";
    }
    if(pointerResults)
    {
      os << "void ";
    }
    else
    {
      if (failed(emitter.emitTypes(functionOp.getLoc(), functionOp.getFunctionType().getResults(), isSupportFunc)))
        return failure();
      os << ' ';
    }
    os << functionOpName << '(';
    if (pointerResults)
    {
      if (failed(interleaveCommaWithError(functionOp.getFunctionType().getResults(), os,
        [&](Type resultType) -> LogicalResult
        {
          if (failed(emitter.emitType(functionOp.getLoc(), resultType, isSupportFunc)))
            return failure();
          // Memrefs are returned by pointer
          if (isSupportFunc && isa<MemRefType>(resultType))
            os << "*";
          return success();
        })))
      {
        return failure();
      }
      //If there will be any arg types, add an extra comma
      if (!(functionOp.getFunctionType().getResults().empty() ||
            functionOp.getArgumentTypes().empty())) {
        os << ", ";
      }
    }
    if (failed(interleaveCommaWithError(functionOp.getArgumentTypes(), os,
      [&](Type argType) -> LogicalResult
      {
        if (failed(emitter.emitType(functionOp.getLoc(), argType, isSupportFunc)))
          return failure();
        // Memrefs are passed by pointer
        if (isSupportFunc && isa<MemRefType>(argType))
          os << "*";
        return success();
      })))
    {
      return failure();
    }
    os << ");\n";
    if(isSupportFunc)
      os << "#endif\n";
    return success();
  }
  // Otherwise, it's a function definition with body.
  KokkosCppEmitter::Scope scope(emitter);
  if (failed(emitter.emitTypes(functionOp.getLoc(),
                               functionOp.getResultTypes())))
    return failure();
  os << ' ' << functionOpName;
  os << "(";
  //Make a list of the memref parameters that need to be converted to Kokkos::Views inside the body
  std::vector<BlockArgument> stridedMemrefParams;
  if (failed(interleaveCommaWithError(
          functionOp.getArguments(), os,
          [&](BlockArgument arg) -> LogicalResult {
            //PyTACO runtime passes dense memrefs to the entry point (pytaco_main) as
            //StridedMemRefType<...>*, not raw data pointer (like with Torch/NumPy).
            bool isStridedMemRef =
              emitter.supportingSparse() && isa<MemRefType>(arg.getType()) && functionOpName == "pytaco_main";
            if(isStridedMemRef)
            {
              stridedMemrefParams.push_back(arg);
              if (failed(emitter.emitType(functionOp.getLoc(), arg.getType(), true)))
                return failure();
              emitter << "*";
            }
            else
            {
              // Emit normal types (e.g. Kokkos::View<..> or LAPIS::DualView for MemRefType)
              if (MemRefType mrt = dyn_cast<MemRefType>(arg.getType())) {
                // Get the space based on how this argument gets used
                kokkos::MemorySpace space = kokkos::getMemSpace(arg);
                if (failed(emitter.emitMemrefType(functionOp.getLoc(), mrt, space)))
                  return failure();
              }
              else {
                if (failed(emitter.emitType(functionOp.getLoc(), arg.getType())))
                  return failure();
              }
            }
            os << " " << emitter.getOrCreateName(arg);
            // Suffix the names of StridedMemRefType parameters with _smr
            if(isStridedMemRef)
              os << "_smr";
            return success();
          })))
    return failure();
  os << ") {\n";
  os.indent();

  //Convert any StridedMemRefType parameters to Kokkos::View
  for(BlockArgument arg : stridedMemrefParams)
  {
    MemRefType mrt = cast<MemRefType>(arg.getType());
    kokkos::MemorySpace space = kokkos::getMemSpace(arg);
    emitter << "auto " << emitter.getOrCreateName(arg) << " = LAPIS::stridedMemrefToView<";
    if (failed(emitter.emitMemrefType(functionOp.getLoc(), mrt, space)))
      return failure();
    emitter << ">(*" << emitter.getOrCreateName(arg) << "_smr);\n";
  }

  Region::BlockListType &blocks = functionOp.getBlocks();
  // Create label names for basic blocks.
  for (Block &block : blocks) {
    emitter.getOrCreateName(block);
  }

  // Declare variables for basic block arguments.
  for (auto it = std::next(blocks.begin()); it != blocks.end(); ++it) {
    Block &block = *it;
    for (BlockArgument &arg : block.getArguments()) {
      if (emitter.hasValueInScope(arg))
        return functionOp.emitOpError(" block argument #")
               << arg.getArgNumber() << " is out of scope";
      if (failed(
              emitter.emitType(block.getParentOp()->getLoc(), arg.getType()))) {
        return failure();
      }
      os << " " << emitter.getOrCreateName(arg) << ";\n";
    }
  }

  for (Block &block : blocks) {
    // Only print a label if the block has predecessors.
    if (!block.hasNoPredecessors()) {
      if (failed(emitter.emitLabel(block)))
        return failure();
    }
    for (Operation &op : block.getOperations()) {
      // When generating code for an scf.if or cf.cond_br op no semicolon needs
      // to be printed after the closing brace.
      // When generating code for an scf.for op, printing a trailing semicolon
      // is handled within the printOperation function.
      bool trailingSemicolon =
          !isa<scf::IfOp, scf::ForOp, cf::CondBranchOp>(op);

      if (failed(emitter.emitOperation(
              op, /*trailingSemicolon=*/trailingSemicolon)))
        return failure();
    }
  }
  os.unindent() << "}\n\n";
  // Finally, create the corresponding wrapper function that is callable from Python (if one is needed)
  if(!emitter.emittingPython())
    return success();
  // Emit a wrapper function without Kokkos::View for Python to call
  os << "extern \"C\" void " << "py_" << functionOpName << '(';
  // Put the results first: primitives and memrefs are both passed by pointer.
  // Python interface will enforce LayoutRight on all numpy arrays.
  //
  FunctionType ftype = functionOp.getFunctionType();
  size_t numResults = ftype.getNumResults();
  size_t numParams = ftype.getNumInputs();
  for(size_t i = 0; i < numResults; i++)
  {
    if(i != 0)
      os << ", ";
    auto retType = ftype.getResult(i);
    if(auto memrefType = dyn_cast<MemRefType>(retType))
    {
      //This is represented using a pointer to the element type
      if(failed(emitter.emitType(functionOp.getLoc(), memrefType.getElementType())))
        return failure();
      os << "* ret" << i;
    }
    else
    {
      //Assuming it is a scalar primitive
      if(failed(emitter.emitType(functionOp.getLoc(), retType)))
        return failure();
      os << "* ret" << i;
    }
  }
  // Now emit the parameters - primitives passed by value
  for(size_t i = 0; i < numParams; i++)
  {
    if(i != 0 || numResults)
      os << ", ";
    auto paramType = ftype.getInput(i);
    if(auto memrefType = dyn_cast<MemRefType>(paramType))
    {
      if(emitter.supportingSparse())
      {
        //For PyTACO: memrefs are passed as pointer to StridedMemRefType
        if(failed(emitter.emitType(functionOp.getLoc(), memrefType, true)))
          return failure();
      }
      else
      {
        //For Torch/NumPy:
        //This is represented using a pointer to the element type
        if(failed(emitter.emitType(functionOp.getLoc(), memrefType.getElementType())))
          return failure();
      }
      os << "* param" << i;
    }
    else
    {
      //Assuming it is a scalar primitive
      if(failed(emitter.emitType(functionOp.getLoc(), paramType)))
        return failure();
      os << " param" << i;
    }
  }
  os << ")\n";
  os << "{\n";
  os.indent();
  //FOR DEBUGGING THE EMITTED CODE:
  //The next 3 lines makes the generated function pause to let you attach a debugger
  //os << "std::cout << \"Starting MLIR function on process \" << getpid() << '\\n';\n";
  //os << "std::cout << \"Optionally attach debugger now, then press <Enter> to continue: \";\n";
  //os << "std::cin.get();\n";
  //Create/allocate device Kokkos::Views for the memref inputs.
  //TODO: if executing on on host, we might as well use the NumPy buffers directly
  if(!emitter.supportingSparse())
  {
    for(size_t i = 0; i < numParams; i++)
    {
      auto paramType = ftype.getInput(i);
      if(auto memrefType = dyn_cast<MemRefType>(paramType))
      {
        int64_t span = KokkosCppEmitter::getMemrefSpan(memrefType);
        os << "Kokkos::View<";
        if(failed(emitter.emitType(functionOp.getLoc(), memrefType.getElementType())))
          return failure();
        os << "*> param" << i << "_buf(\"param buffer " << i << "\", ";
        os << span << ");\n";
        //Copy from the NumPy buffer
        os << "Kokkos::deep_copy(exec_space(), param" << i << "_buf, Kokkos::View<";
        if(failed(emitter.emitType(functionOp.getLoc(), memrefType.getElementType())))
          return failure();
        os << "*, Kokkos::HostSpace>(param" << i << ", " << span << "));\n";
      }
    }
  }
  os << "auto results = " << functionOpName << "(";
  //Construct a Kokkos::View for each memref input, from raw pointer.
  for(size_t i = 0; i < numParams; i++)
  {
    if(i != 0)
      os << ", ";
    auto paramType = ftype.getInput(i);
    auto memrefType = dyn_cast<MemRefType>(paramType);
    if(!emitter.supportingSparse() && memrefType)
    {
      //TODO: handle dynamic sized and unranked memrefs here
      if(failed(emitter.emitType(functionOp.getLoc(), paramType)))
        return failure();
      os << "(param" << i << "_buf.data())";
    }
    else
    {
      os << "param" << i;
    }
  }
  os << ");\n";
  //Now, unpack the results (if any) to the return values.
  //If there are multiple results, 'results' will be a std::tuple.
  //Need to deep_copy memref returns back to the NumPy buffers.
  for(size_t i = 0; i < numResults; i++)
  {
    if(i != 0)
      os << ", ";
    auto retType = ftype.getResult(i);
    auto memrefType = dyn_cast<MemRefType>(retType);
    if(memrefType && !emitter.supportingSparse())
    {
      int64_t span = KokkosCppEmitter::getMemrefSpan(memrefType);
      os << "Kokkos::deep_copy(exec_space(), Kokkos::View<";
      if(failed(emitter.emitType(functionOp.getLoc(), memrefType.getElementType())))
        return failure();
      os << "*, Kokkos::HostSpace>(ret" << i << ", " << span << "), Kokkos::View<";
      if(failed(emitter.emitType(functionOp.getLoc(), memrefType.getElementType())))
        return failure();
      os << "*>(";
      if(numResults == size_t(1))
        os << "results";
      else
        os << "std::get<" << i << ">(results)";
      os << ".data(), " << span << "));\n";
    }
    else
    {
      os << "*ret" << i << " = ";
      if(numResults == size_t(1))
        os << "results;\n";
      else
        os << "std::get<" << i << ">(results);\n";
    }
  }
  os.unindent();
  os << "}\n";
  // Now that the native function (name: "py_" + functionOpName)
  // exists, generate the Python function to call it.
  //
  // Get the NumPy type corresponding to MLIR primitive.
  auto getNumpyType = [](Type t) -> std::string
  {
    if(t.isIndex())
      return "numpy.uint64";
    //Note: treating MLIR "signless" integer types as equivalent to unsigned NumPy integers.
    if(t.isSignlessInteger(8) || t.isUnsignedInteger(8))
      return "numpy.uint8";
    if(t.isSignlessInteger(16) || t.isUnsignedInteger(16))
      return "numpy.uint16";
    if(t.isSignlessInteger(32) || t.isUnsignedInteger(32))
      return "numpy.uint32";
    if(t.isSignlessInteger(64) || t.isUnsignedInteger(64))
      return "numpy.uint64";
    if(t.isSignedInteger(8))
      return "numpy.int8";
    if(t.isSignedInteger(16))
      return "numpy.int16";
    if(t.isSignedInteger(32))
      return "numpy.int32";
    if(t.isSignedInteger(64))
      return "numpy.int64";
    if(t.isF16())
      return "numpy.float16";
    if(t.isF32())
      return "numpy.float32";
    if(t.isF64())
      return "numpy.float64";
    return "";
  };

  // Use this to enforce LayoutRight for input memrefs (zero cost if it already is):
  // arr = numpy.require(arr, dtype=TYPE, requirements=['C'])
  // NOTE: numpy.zeros(shape, dtype=...) already defaults to LayoutRight, so that is ok.
  //
  // TODO: need to pass and return runtime extents here.
  // For return values, need 3 phases:
  //  - compute results in Kokkos and hold them temporarily.
  //  - get the extents back to Python and allocate
  //  - copy out the results to Python and then free the Kokkos temporaries.
  auto& py_os = emitter.py_ostream();
  //NOTE: this function is a member of the module's class, but py_os is already indented to write methods.
  py_os << "def " << functionOpName << "(self, ";
  for(size_t i = 0; i < numParams; i++)
  {
    if(i != 0)
      py_os << ", ";
    py_os << "param" << i;
  }
  py_os << "):\n";
  py_os.indent();
  // Enforce types on all inputs (for Torch/NumPy only)
  if(!emitter.supportingSparse())
  {
  for(size_t i = 0; i < numParams; i++)
  {
    auto paramType = ftype.getInput(i);
      if(!isa<LLVM::LLVMPointerType>(paramType))
      {
        py_os << "param" << i << " = ";
        if(auto memrefType = dyn_cast<MemRefType>(paramType))
        {
          std::string numpyDType = getNumpyType(memrefType.getElementType());
          if(!numpyDType.size())
            return failure();
          py_os << "numpy.require(param" << i << ", dtype=" << numpyDType << ", requirements=['C'])\n";
        }
        else 
        {
          //Wrap scalar primitives in 1D NumPy array.
          //This gives it the correct type, and lets us use the same ndarray CTypes API as memrefs.
          std::string numpyDType = getNumpyType(paramType);
          if(!numpyDType.size())
            return failure();
          py_os << "numpy.array(param" << i << ", dtype=" << numpyDType << ", ndmin=1)\n";
        }
      }
    }
  }
  // Construct outputs
  // Note: by default, numpy.zeros uses LayoutRight
  for(size_t i = 0; i < numResults; i++)
  {
    auto retType = ftype.getResult(i);
    //TODO: support returning a StridedMemRefType for PyTACO
    if(auto memrefType = dyn_cast<MemRefType>(retType))
    {
      std::string numpyDType = getNumpyType(memrefType.getElementType());
      if(!numpyDType.size())
        return failure();
      py_os << "ret" << i << " = numpy.zeros((";
      //TODO: this assumes static dimensions
      for(size_t j = 0; j < memrefType.getShape().size(); j++)
      {
        if(j != 0)
          py_os << ", ";
        py_os << memrefType.getShape()[j];
      }
      py_os << "), dtype=" << numpyDType << ")\n";
    }
    else if(isa<LLVM::LLVMPointerType>(retType))
    {
      //For pointer results, declare a void* and pass its address (void**)
      py_os << "ret" << i << " = ctypes.pointer(ctypes.pointer(ctypes.c_char(0)))\n";
    }
    else
    {
      //For scalars, construct a single-element numpy ndarray so that we can use its CTypes API
      std::string numpyDType = getNumpyType(retType);
      if(!numpyDType.size())
        return failure();
      py_os << "ret" << i << " = numpy.zeros(1, dtype=" << numpyDType << ")\n";
    }
  }
  // Generate the native call. It always returns void.
  py_os << "self.libHandle.py_" << functionOpName << "(";
  // Outputs go first
  for(size_t i = 0; i < numResults; i++)
  {
    auto retType = ftype.getResult(i);
    if(i != 0)
      py_os << ", ";
    if(isa<LLVM::LLVMPointerType>(retType))
    {
      // Pointer
      py_os << "ctypes.pointer(ret" << i << ")";
    }
    else
    {
      // numpy array, or scalar
      py_os << "ret" << i << ".ctypes.data_as(ctypes.c_void_p)";
    }
  }
  for(size_t i = 0; i < numParams; i++)
  {
    auto paramType = ftype.getInput(i);
    if(i != 0 || numResults != size_t(0))
    {
      py_os << ", ";
    }
    if(isa<LLVM::LLVMPointerType>(paramType) || emitter.supportingSparse())
    {
      py_os << "param" << i;
    }
    else
    {
      //Numpy array (or a scalar from a numpy array)
      py_os << "param" << i << ".ctypes.data_as(ctypes.c_void_p)";
    }
  }
  py_os << ")\n";
  // Finally, generate the return. Note that in Python, a 1-elem tuple is equivalent to scalar.
  if(numResults)
  {
    py_os << "return (";
    for(size_t i = 0; i < numResults; i++)
    {
      if(i != 0)
        py_os << ", ";
      auto retType = ftype.getResult(i);
      if(isa<MemRefType, LLVM::LLVMPointerType>(retType))
      {
        //Return the whole memref
        py_os << "ret" << i;
      }
      else
      {
        //Return just the single element
        py_os << "ret" << i << "[0]";
      }
    }
    py_os << ")\n";
  }
  py_os.unindent();
  return success();
}

KokkosCppEmitter::KokkosCppEmitter(raw_ostream& decl_os_, raw_ostream& os_, bool enableSparseSupport_)
    : decl_os(decl_os_), os(os_), py_os(nullptr), enableSparseSupport(enableSparseSupport_) {
  valueInScopeCount.push(0);
  labelInScopeCount.push(0);
  if(enableSparseSupport)
    registerRuntimeSupportFunctions();
}

KokkosCppEmitter::KokkosCppEmitter(raw_ostream& decl_os_, raw_ostream& os_, raw_ostream& py_os_, bool enableSparseSupport_)
    : decl_os(decl_os_), os(os_), enableSparseSupport(enableSparseSupport_) {
  this->py_os = std::make_shared<raw_indented_ostream>(py_os_); 
  valueInScopeCount.push(0);
  labelInScopeCount.push(0);
  if(enableSparseSupport)
    registerRuntimeSupportFunctions();
}

void KokkosCppEmitter::registerRuntimeSupportFunctions()
{
  // pointerResult is true for a function if it returns void,
  // but produces a memref result through an output pointer argument.
  //
  // Most sparse support functions are prefixed with _mlir_ciface_ in the library,
  // but a few do not have the prefix.
  auto registerCIface =
    [&](bool pointerResult, std::string name)
    {
      sparseSupportFunctions.insert({name, {pointerResult, std::string("_mlir_ciface_") + name}});
    };
  auto registerNonPrefixed =
    [&](bool pointerResult, std::string name)
    {
      sparseSupportFunctions.insert({name, {pointerResult, name}});
    };
  // SparseTensor functions
  for (std::string funcName :
       {"sparseCoordinates0",  "sparseCoordinates8",
        "sparseCoordinates16", "sparseCoordinates32", "sparseCoordinates64",
        "sparsePositions0",    "sparsePositions8",    "sparsePositions16",
        "sparsePositions32",   "sparsePositions64",   "sparseValuesBF16",
        "sparseValuesC32",     "sparseValuesC64",     "sparseValuesF16",
        "sparseValuesF32",     "sparseValuesF64",     "sparseValuesI8",
        "sparseValuesI16",     "sparseValuesI32",     "sparseValuesI64"}) {
    registerCIface(true, funcName);
  }
  for (std::string funcName :
      { "newSparseTensor", "lexInsertI8", "lexInsertI16", "lexInsertI32",
      "lexInsertI64", "lexInsertF32", "lexInsertF64",
      "expInsertF32", "expInsertF64", "expInsertI8",
      "expInsertI16", "expInsertI32", "expInsertI64"
      }) {
    registerCIface(false, funcName);
  }
  // SparseTensor functions not prefixed with "_mlir_ciface_"
  for (std::string funcName :
      {"endInsert", "sparseDimSize", "sparseLvlSize"}) {
    registerNonPrefixed(false, funcName);
  }
  // PartTensor functions
  for (std::string funcName :
       {"mpi_getActiveMask", "getSlice", "mpi_getSlice",
        "mpi_getSliceForActiveMask", "krs_getRank", "mpi_getRank"}) {
    registerCIface(false, funcName);
  }
  for (std::string funcName :
       {"getPartitions",       "mpi_getPartitions",   "updateSlice", "mpi_setSlice"}) {
    registerCIface(true, funcName);
  }
}

std::string KokkosCppEmitter::getUniqueIdentifier()
{
  return std::string("v") + std::to_string(++valueInScopeCount.top());
}

/// Return the existing or a new name for a Value.
StringRef KokkosCppEmitter::getOrCreateName(Value val) {
  if (!valueMapper.count(val))
    valueMapper.insert(val, formatv("v{0}", ++valueInScopeCount.top()));
  return *valueMapper.begin(val);
}

void KokkosCppEmitter::assignName(StringRef name, Value val) {
  if (!valueMapper.count(val))
    valueMapper.insert(val, std::string(name));
}

/// Return the existing or a new label for a Block.
StringRef KokkosCppEmitter::getOrCreateName(Block &block) {
  if (!blockMapper.count(&block))
    blockMapper.insert(&block, formatv("label{0}", ++labelInScopeCount.top()));
  return *blockMapper.begin(&block);
}

bool KokkosCppEmitter::shouldMapToUnsigned(IntegerType::SignednessSemantics val) {
  switch (val) {
  case IntegerType::Signless:
    return false;
  case IntegerType::Signed:
    return false;
  case IntegerType::Unsigned:
    return true;
  }
  llvm_unreachable("Unexpected IntegerType::SignednessSemantics");
}

bool KokkosCppEmitter::hasValueInScope(Value val) { return valueMapper.count(val) || isScalarConstant(val); }

bool KokkosCppEmitter::hasBlockLabel(Block &block) {
  return blockMapper.count(&block);
}

LogicalResult KokkosCppEmitter::emitAttribute(Location loc, Attribute attr) {
  auto printInt = [&](const APInt &val, bool isUnsigned) {
    if (val.getBitWidth() == 1) {
      if (val.getBoolValue())
        os << "true";
      else
        os << "false";
    } else {
      SmallString<128> strValue;
      val.toString(strValue, 10, !isUnsigned, false);
      os << strValue;
    }
  };

  auto printFloat = [&](const APFloat &val) {
    if (val.isFinite()) {
      SmallString<128> strValue;
      // Use default values of toString except don't truncate zeros.
      val.toString(strValue, 0, 0, false);
      os << strValue;
      switch (llvm::APFloatBase::SemanticsToEnum(val.getSemantics())) {
      case llvm::APFloatBase::S_IEEEsingle:
        os << "f";
        break;
      case llvm::APFloatBase::S_IEEEdouble:
        //no suffix for double literal
        break;
      default:
        puts("WARNING: literal printing only supports float and double now!");
        break;
      };
    } else if (val.isNaN()) {
      os << "NAN";
    } else if (val.isInfinity()) {
      if (val.isNegative())
        os << "-";
      os << "INFINITY";
    }
  };

  // Print floating point attributes.
  if (auto fAttr = dyn_cast<FloatAttr>(attr)) {
    printFloat(fAttr.getValue());
    return success();
  }
  if (auto dense = dyn_cast<DenseFPElementsAttr>(attr)) {
    os << '{';
    interleaveComma(dense, os, [&](const APFloat &val) { printFloat(val); });
    os << '}';
    return success();
  }

  // Print integer attributes.
  if (auto iAttr = dyn_cast<IntegerAttr>(attr)) {
    if (auto iType = dyn_cast<IntegerType>(iAttr.getType())) {
      printInt(iAttr.getValue(), shouldMapToUnsigned(iType.getSignedness()));
      return success();
    }
    if (auto iType = dyn_cast<IndexType>(iAttr.getType())) {
      printInt(iAttr.getValue(), false);
      return success();
    }
  }
  if (auto dense = dyn_cast<DenseIntElementsAttr>(attr)) {
    if (auto iType = dyn_cast<IntegerType>(cast<TensorType>(dense.getType()).getElementType())) {
      os << '{';
      interleaveComma(dense, os, [&](const APInt &val) {
        printInt(val, shouldMapToUnsigned(iType.getSignedness()));
      });
      os << '}';
      return success();
    }
    if (auto iType = dyn_cast<IndexType>(cast<TensorType>(dense.getType()).getElementType())) {
      os << '{';
      interleaveComma(dense, os,
                      [&](const APInt &val) { printInt(val, false); });
      os << '}';
      return success();
    }
  }

  // Print symbolic reference attributes.
  if (auto sAttr = dyn_cast<SymbolRefAttr>(attr)) {
    if (sAttr.getNestedReferences().size() > 1)
      return emitError(loc, "attribute has more than 1 nested reference");
    os << sAttr.getRootReference().getValue();
    return success();
  }

  // Print string attribute (including quotes). Using hex of each character so that special characters don't need escaping.
  if (auto strAttr = dyn_cast<StringAttr>(attr))
  {
    os << '"';
    auto val = strAttr.strref();
    for(char c : val)
    {
      char buf[4];
      snprintf(buf, 4, "%02x", (unsigned) c);
      os << "\\x" << buf;
    }
    os << '"';
    return success();
  }

  // Print type attributes.
  if (auto type = dyn_cast<TypeAttr>(attr))
    return emitType(loc, type.getValue());

  return emitError(loc, "cannot emit attribute of unsupported type");
}

LogicalResult KokkosCppEmitter::emitOperands(Operation &op) {
  auto emitOperandName = [&](Value result) -> LogicalResult {
    if (!hasValueInScope(result))
      return op.emitOpError() << "operand value not in scope";
    if(failed(emitValue(result)))
      return failure();
    return success();
  };
  return interleaveCommaWithError(op.getOperands(), os, emitOperandName);
}

LogicalResult
KokkosCppEmitter::emitOperandsAndAttributes(Operation &op,
                                      ArrayRef<StringRef> exclude) {
  if (failed(emitOperands(op)))
    return failure();
  // Insert comma in between operands and non-filtered attributes if needed.
  if (op.getNumOperands() > 0) {
    for (NamedAttribute attr : op.getAttrs()) {
      if (!llvm::is_contained(exclude, attr.getName().strref())) {
        os << ", ";
        break;
      }
    }
  }
  // Emit attributes.
  auto emitNamedAttribute = [&](NamedAttribute attr) -> LogicalResult {
    if (llvm::is_contained(exclude, attr.getName().strref()))
      return success();
    os << "/* " << attr.getName().getValue() << " */";
    if (failed(emitAttribute(op.getLoc(), attr.getValue())))
      return failure();
    return success();
  };
  return interleaveCommaWithError(op.getAttrs(), os, emitNamedAttribute);
}

LogicalResult
KokkosCppEmitter::emitValue(Value val)
{
  if(isScalarConstant(val))
  {
    arith::ConstantOp op = getScalarConstantOp(val);
    Attribute value = op.getValue();
    return emitAttribute(op.getLoc(), value);
  }
  else
  {
    //If calling this, the value should have already been declared
    if (!valueMapper.count(val))
      return failure();
    os << *valueMapper.begin(val);
    return success();
  }
}

LogicalResult KokkosCppEmitter::emitVariableDeclaration(Value result,
                                                  bool trailingSemicolon) {
  auto op = result.getDefiningOp();
  if (hasValueInScope(result)) {
    if(op) {
      return op->emitError(
          "result variable for the operation already declared");
    }
    else {
      return failure();
    }
  }
  Location loc = op ? op->getLoc() : Location(LocationAttr());
  if (failed(emitType(loc, result.getType())))
    return failure();
  os << " " << getOrCreateName(result);
  if (trailingSemicolon)
    os << ";\n";
  return success();
}

LogicalResult KokkosCppEmitter::emitAssignPrefix(Operation &op) {
  switch (op.getNumResults()) {
  case 0:
    break;
  case 1: {
    OpResult result = op.getResult(0);
    if (failed(emitVariableDeclaration(result, /*trailingSemicolon=*/false)))
      return failure();
    os << " = ";
    break;
  }
  default:
    for (OpResult result : op.getResults()) {
      if (failed(emitVariableDeclaration(result, /*trailingSemicolon=*/true)))
        return failure();
    }
    os << "std::tie(";
    interleaveComma(op.getResults(), os,
                    [&](Value result) { os << getOrCreateName(result); });
    os << ") = ";
  }
  return success();
}

LogicalResult KokkosCppEmitter::emitLabel(Block &block) {
  if (!hasBlockLabel(block))
    return block.getParentOp()->emitError("label for block not found");
  // FIXME: Add feature in `raw_indented_ostream` to ignore indent for block
  // label instead of using `getOStream`.
  os.getOStream() << getOrCreateName(block) << ":\n";
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    arith::NegFOp op) {
  if (failed(emitter.emitType(op.getLoc(), op.getResult().getType())))
    return failure();
  emitter << ' ' << emitter.getOrCreateName(op.getResult()) << " = -" << emitter.getOrCreateName(op.getOperand());
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    arith::CmpFOp op) {
  //Note: see ArithmeticOpsEnums.h.inc for values of arith::CmpFPredicate
  //2 types of float comparisons in MLIR: ordered and unordered. Ordered is like C.
  //Unordered is true if the ordered version would be true, OR if neither a<=b nor a>=b is true.
  //The second case applies for example when a and/or b is NaN.
  emitter << "bool " << emitter.getOrCreateName(op.getResult()) << " = ";
  //Handle two easy cases first - always true or always false.
  if(op.getPredicate() == arith::CmpFPredicate::AlwaysFalse)
  {
    emitter << "false";
    return success();
  }
  if(op.getPredicate() == arith::CmpFPredicate::AlwaysTrue)
  {
    emitter << "true";
    return success();
  }
  if(op.getPredicate() == arith::CmpFPredicate::ORD)
  {
    emitter << "!(Kokkos::isnan(";
    if(failed(emitter.emitValue(op.getLhs())))
      return failure();
    emitter << ") || Kokkos::isnan(";
    if(failed(emitter.emitValue(op.getRhs())))
      return failure();
    emitter << "))";
    return success();
  }
  if(op.getPredicate() == arith::CmpFPredicate::UNO)
  {
    emitter << "(Kokkos::isnan(";
    if(failed(emitter.emitValue(op.getLhs())))
      return failure();
    emitter << ") || Kokkos::isnan(";
    if(failed(emitter.emitValue(op.getRhs())))
      return failure();
    emitter << "))";
    return success();
  }
  //CmpFOp predicate is an enum, 0..15 inclusive. 1..6 are ordered comparisons (== > >= < <= !=), and 8..13 are corresponding unordered comparisons.
  int rawPred = (int) op.getPredicate();
  bool isUnordered = rawPred >= 8 && rawPred < 15;
  //Now, can convert unordered predicates to equivalent ordered.
  if(isUnordered)
    rawPred -= 7;
  if(isUnordered)
  {
    emitter << "(Kokkos::isnan(";
    if(failed(emitter.emitValue(op.getLhs())))
      return failure();
    emitter << ") || Kokkos::isnan(";
    if(failed(emitter.emitValue(op.getRhs())))
      return failure();
    emitter << ")) || ";
  }
  emitter << "(";
  if(failed(emitter.emitValue(op.getLhs())))
    return failure();
  emitter << ' ';
  switch(rawPred)
  {
    case 1:
      emitter << "=="; break;
    case 2:
      emitter << ">"; break;
    case 3:
      emitter << ">="; break;
    case 4:
      emitter << "<"; break;
    case 5:
      emitter << "<="; break;
    case 6:
      emitter << "!="; break;
    default:
      return op.emitError("CmpFOp: should never get here");
  }
  emitter << ' ';
  if(failed(emitter.emitValue(op.getRhs())))
    return failure();
  emitter << ")";
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    arith::CmpIOp op) {
  //Note: see ArithmeticOpsEnums.h.inc for values of arith::CmpIPredicate
  int rawPred = (int) op.getPredicate();
  bool needsCast = rawPred > 1;
  bool castToUnsigned = rawPred >= 6;
  emitter << "bool " << emitter.getOrCreateName(op.getResult()) << " = ";
  //Emit a value, but cast to signed/unsigned depending on needsCast and castToUnsigned.
  auto emitValueWithSignednessCast = [&](Value v)
  {
    if(needsCast)
    {
      if(castToUnsigned)
      {
        emitter << "static_cast<std::make_unsigned_t<";
        if(failed(emitter.emitType(op.getLoc(), v.getType())))
          return failure();
        emitter << ">>(";
      }
      else
      {
        emitter << "static_cast<std::make_signed_t<";
        if(failed(emitter.emitType(op.getLoc(), v.getType())))
          return failure();
        emitter << ">>(";
      }
    }
    if(failed(emitter.emitValue(v)))
      return failure();
    if(needsCast)
      emitter << ')';
    return success();
  };
  if(failed(emitValueWithSignednessCast(op.getLhs())))
    return failure();
  emitter << ' ';
  switch(op.getPredicate())
  {
    case arith::CmpIPredicate::eq:
      emitter << "=="; break;
    case arith::CmpIPredicate::ne:
      emitter << "!="; break;
    case arith::CmpIPredicate::slt:
    case arith::CmpIPredicate::ult:
      emitter << "<"; break;
    case arith::CmpIPredicate::sle:
    case arith::CmpIPredicate::ule:
      emitter << "<="; break;
    case arith::CmpIPredicate::sgt:
    case arith::CmpIPredicate::ugt:
      emitter << ">"; break;
    case arith::CmpIPredicate::sge:
    case arith::CmpIPredicate::uge:
      emitter << ">="; break;
  }
  emitter << ' ';
  if(failed(emitValueWithSignednessCast(op.getRhs())))
    return failure();
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    arith::SelectOp op) {
  if(failed(emitter.emitType(op.getLoc(), op.getResult().getType())))
    return failure();
  emitter << ' ' << emitter.getOrCreateName(op.getResult()) << " = ";
  if(failed(emitter.emitValue(op.getCondition())))
    return failure();
  emitter << "? ";
  if(failed(emitter.emitValue(op.getTrueValue())))
    return failure();
  emitter << " : ";
  if(failed(emitter.emitValue(op.getFalseValue())))
    return failure();
  return success();
}

template<typename T>
static LogicalResult printFloatMinMax(KokkosCppEmitter &emitter, T op, const char* selectOperator) {
  if(failed(emitter.emitType(op.getLoc(), op.getResult().getType())))
    return failure();
  emitter << ' ' << emitter.getOrCreateName(op.getResult()) << " = ";
  emitter << '(';
  if(failed(emitter.emitValue(op.getLhs())))
    return failure();
  emitter << ' ' << selectOperator << ' ';
  if(failed(emitter.emitValue(op.getRhs())))
    return failure();
  emitter << ')';
  emitter << " ? ";
  if(failed(emitter.emitValue(op.getLhs())))
    return failure();
  emitter << " : ";
  if(failed(emitter.emitValue(op.getRhs())))
    return failure();
  return success();
}

template<typename T>
static LogicalResult printIntMinMax(KokkosCppEmitter &emitter, T op, const char* selectOperator, bool castToUnsigned) {
  auto emitValueWithSignednessCast = [&](Value v)
  {
    if(castToUnsigned)
    {
      emitter << "static_cast<std::make_unsigned_t<";
      if(failed(emitter.emitType(op.getLoc(), v.getType())))
        return failure();
      emitter << ">>(";
    }
    else
    {
      emitter << "static_cast<std::make_signed_t<";
      if(failed(emitter.emitType(op.getLoc(), v.getType())))
        return failure();
      emitter << ">>(";
    }
    if(failed(emitter.emitValue(v)))
      return failure();
    emitter << ')';
    return success();
  };
  if(failed(emitter.emitType(op.getLoc(), op.getResult().getType())))
    return failure();
  emitter << ' ' << emitter.getOrCreateName(op.getResult()) << " = ";
  emitter << '(';
  if(failed(emitValueWithSignednessCast(op.getLhs())))
    return failure();
  emitter << ' ' << selectOperator << ' ';
  if(failed(emitValueWithSignednessCast(op.getRhs())))
    return failure();
  emitter << ')';
  emitter << " ? ";
  if(failed(emitValueWithSignednessCast(op.getLhs())))
    return failure();
  emitter << " : ";
  if(failed(emitValueWithSignednessCast(op.getRhs())))
    return failure();
  return success();
}

template<typename T>
struct ArithBinaryInfixOperator
{
  static std::string get() {return std::string("/* ERROR: binary infix operator for ") + T::getOperationName() + " is not registered */";}
};

template<>
struct ArithBinaryInfixOperator<arith::AddFOp>
{
  static std::string get() {return "+";}
};

template<>
struct ArithBinaryInfixOperator<arith::AddIOp>
{
  static std::string get() {return "+";}
};

template<>
struct ArithBinaryInfixOperator<arith::SubFOp>
{
  static std::string get() {return "-";}
};

template<>
struct ArithBinaryInfixOperator<arith::SubIOp>
{
  static std::string get() {return "-";}
};

template<>
struct ArithBinaryInfixOperator<arith::MulFOp>
{
  static std::string get() {return "*";}
};

template<>
struct ArithBinaryInfixOperator<arith::MulIOp>
{
  static std::string get() {return "*";}
};

template<>
struct ArithBinaryInfixOperator<arith::DivFOp>
{
  static std::string get() {return "/";}
};

template<>
struct ArithBinaryInfixOperator<arith::DivSIOp>
{
  static std::string get() {return "/";}
};

template<>
struct ArithBinaryInfixOperator<arith::DivUIOp>
{
  static std::string get() {return "/";}
};

template<>
struct ArithBinaryInfixOperator<arith::AndIOp>
{
  static std::string get() {return "&";}
};

template<>
struct ArithBinaryInfixOperator<arith::OrIOp>
{
  static std::string get() {return "|";}
};

template<>
struct ArithBinaryInfixOperator<arith::XOrIOp>
{
  static std::string get() {return "^";}
};

template<typename T>
static LogicalResult printBinaryInfixOperation(KokkosCppEmitter &emitter, T op) {
  if(failed(emitter.emitType(op.getLoc(), op.getResult().getType())))
    return failure();
  emitter << ' ' << emitter.getOrCreateName(op.getResult()) << " = ";
  if(failed(emitter.emitValue(op.getLhs())))
    return failure();
  emitter << ' ' << ArithBinaryInfixOperator<T>::get() << ' ';
  if(failed(emitter.emitValue(op.getRhs())))
    return failure();
  return success();
}

template<typename T>
static LogicalResult printImplicitConversionOperation(KokkosCppEmitter &emitter, T op) {
  if(failed(emitter.emitType(op.getLoc(), op.getOut().getType())))
    return failure();
  emitter << ' ' << emitter.getOrCreateName(op.getOut()) << " = ";
  if(failed(emitter.emitValue(op.getIn())))
    return failure();
  return success();
}

template<typename T>
struct MathFunction
{
  static std::string get() {return std::string("/* ERROR: math function for operator ") + T::getOperationName() + " is not registered */";}
};

template<>
struct MathFunction<math::SqrtOp>
{
  static std::string get() {return "Kokkos::sqrt";}
};

template<>
struct MathFunction<math::AbsIOp>
{
  static std::string get() {return "Kokkos::abs";}
};

template<>
struct MathFunction<math::AbsFOp>
{
  static std::string get() {return "Kokkos::abs";}
};

template<>
struct MathFunction<math::ExpOp>
{
  static std::string get() {return "Kokkos::exp";}
};

template<>
struct MathFunction<math::Exp2Op>
{
  static std::string get() {return "Kokkos::exp2";}
};

template<>
struct MathFunction<math::SinOp>
{
  static std::string get() {return "Kokkos::sin";}
};

template<>
struct MathFunction<math::CosOp>
{
  static std::string get() {return "Kokkos::cos";}
};

template<>
struct MathFunction<math::AtanOp>
{
  static std::string get() {return "Kokkos::atan";}
};

template<>
struct MathFunction<math::TanhOp>
{
  static std::string get() {return "Kokkos::tanh";}
};

template<>
struct MathFunction<math::ErfOp>
{
  static std::string get() {return "Kokkos::erf";}
};

template<>
struct MathFunction<math::LogOp>
{
  static std::string get() {return "Kokkos::log";}
};

template<>
struct MathFunction<math::Log2Op>
{
  static std::string get() {return "Kokkos::log2";}
};

template<typename T>
static LogicalResult printMathOperation(KokkosCppEmitter &emitter, T op) {
  if(failed(emitter.emitType(op.getLoc(), op.getResult().getType())))
    return failure();
  emitter << ' ' << emitter.getOrCreateName(op.getResult()) << " = ";
  emitter << MathFunction<T>::get() << "(";
  if(failed(emitter.emitValue(op.getOperand())))
    return failure();
  emitter << ")";
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter, math::RsqrtOp op) {
  if(failed(emitter.emitType(op.getLoc(), op.getResult().getType())))
    return failure();
  emitter << ' ' << emitter.getOrCreateName(op.getResult()) << " = ";
  if(failed(emitter.emitType(op.getLoc(), op.getResult().getType())))
    return failure();
  emitter << "(1.0f) / Kokkos::sqrt(";
  if(failed(emitter.emitValue(op.getOperand())))
    return failure();
  emitter << ")";
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    LLVM::ZeroOp op) {
  if(failed(emitter.emitType(op.getLoc(), op.getResult().getType())))
    return failure();
  emitter << ' ' << emitter.getOrCreateName(op.getResult()) << " = 0";
  return success();
}

LogicalResult KokkosCppEmitter::emitOperation(Operation &op, bool trailingSemicolon) {
  // Some ops need to be processed by the emitter, but don't print anything to the C++ code.
  // Avoid printing the semicolon in this case.
  bool skipPrint = false;
  if(isa<arith::ConstantOp, memref::CastOp, memref::GetGlobalOp, kokkos::YieldOp>(&op)) {
    skipPrint = true;
  }
  //os << "// " << op.getName() << '\n';
  LogicalResult status =
      llvm::TypeSwitch<Operation *, LogicalResult>(&op)
          // Builtin ops.
          .Case<func::FuncOp, ModuleOp>(
              [&](auto op) { return printOperation(*this, op); })
          // Kokkos ops.
          .Case<kokkos::RangeParallelOp, kokkos::TeamParallelOp, kokkos::ThreadParallelOp, kokkos::TeamBarrierOp, kokkos::SingleOp, kokkos::UpdateReductionOp, kokkos::SyncOp, kokkos::ModifyOp, kokkos::YieldOp>(
              [&](auto op) { return printOperation(*this, op); })
          // CF ops.
          .Case<cf::AssertOp>(
              [&](auto op) { return printOperation(*this, op); })
          // Func ops.
          .Case<func::CallOp, func::ConstantOp, func::ReturnOp>(
              [&](auto op) { return printOperation(*this, op); })
          // SCF ops.
          .Case<scf::ForOp, scf::WhileOp, scf::IfOp, scf::YieldOp, scf::ConditionOp>(
              [&](auto op) { return printOperation(*this, op); })
          // Arithmetic ops: general
          .Case<arith::ConstantOp, arith::FPToUIOp, arith::NegFOp, arith::CmpFOp, arith::CmpIOp, arith::SelectOp, arith::IndexCastOp, arith::SIToFPOp, arith::MinNumFOp, arith::MaxNumFOp>(
              [&](auto op) { return printOperation(*this, op); })
          // Arithmetic ops: standard binary infix operators. All have the same syntax "result = lhs <operator> rhs;".
          // ArithBinaryInfixOperator<Op>::get() will provide the <operator>.
          .Case<arith::AddFOp, arith::AddIOp, arith::SubFOp, arith::SubIOp, arith::MulFOp, arith::MulIOp, arith::DivFOp, arith::DivSIOp, arith::DivUIOp, arith::AndIOp, arith::OrIOp, arith::XOrIOp>(
              [&](auto op) { return printBinaryInfixOperation(*this, op); })
          // Arithmetic ops: type casting that C++ compiler can handle automatically with implicit conversion: "result = operand;"
          .Case<arith::UIToFPOp, arith::FPToSIOp, arith::TruncFOp, arith::ExtFOp, arith::ExtSIOp, arith::ExtUIOp>(
              [&](auto op) { return printImplicitConversionOperation(*this, op); })
          // Arithmetic ops: min/max expressed using ternary operator.
          .Case<arith::MinimumFOp>(
              [&](auto op) { return printFloatMinMax(*this, op, "<"); })
          .Case<arith::MaximumFOp>(
              [&](auto op) { return printFloatMinMax(*this, op, ">"); })
          .Case<arith::MinSIOp>(
              [&](auto op) { return printIntMinMax(*this, op, "<", false); })
          .Case<arith::MaxSIOp>(
              [&](auto op) { return printIntMinMax(*this, op, ">", false); })
          .Case<arith::MinUIOp>(
              [&](auto op) { return printIntMinMax(*this, op, "<", true); })
          .Case<arith::MaxUIOp>(
              [&](auto op) { return printIntMinMax(*this, op, ">", true); })
          // Math ops: general
          .Case<math::RsqrtOp>(
              [&](auto op) { return printOperation(*this, op); })
          // Math ops: unary functions supported by Kokkos
          .Case<math::SqrtOp, math::AbsIOp, math::AbsFOp, math::ExpOp, math::Exp2Op, math::SinOp, math::CosOp, math::AtanOp, math::TanhOp, math::ErfOp, math::LogOp, math::Log2Op>(
              [&](auto op) { return printMathOperation(*this, op); })
          // Memref ops.
          .Case<memref::GlobalOp, memref::GetGlobalOp, memref::AllocOp,
                memref::AllocaOp, memref::StoreOp, memref::LoadOp,
                memref::CopyOp, /* memref::SubViewOp, memref::CollapseShapeOp, */
                memref::CastOp, memref::DeallocOp, memref::DimOp /*,
                memref::ReinterpretCastOp*/>(
              [&](auto op) { return printOperation(*this, op); })
          // EmitC ops.
          .Case<emitc::CallOp, emitc::CallOpaqueOp>(
              [&](auto op) { return printOperation(*this, op); })
          // LLVM ops.
          .Case<LLVM::ZeroOp>(
              [&](auto op) { return printOperation(*this, op); })
          // Other operations are unknown/unsupported.
          .Default([&](Operation *) {
            return op.emitOpError("unable to find printer for op");
          });

  if (failed(status))
    return failure();
  if(!skipPrint) {
    os << (trailingSemicolon ? ";\n" : "\n");
  }
  return success();
}

LogicalResult KokkosCppEmitter::emitInitAndFinalize()
{
  os << "extern \"C\" void lapis_initialize()\n";
  os << "{\n";
  os.indent();
  os << "if (!Kokkos::is_initialized()) Kokkos::initialize();\n";
  //For each global view: allocate it, and if there is initializing data, copy it
  for(auto& op : globalViews)
  {
    os << "{\n";
    os.indent();
    os << op.getSymName() << " = ";
    if(failed(emitType(op.getLoc(), op.getType())))
      return failure();
    //TODO: handle dynamic sized view here? This assumes all compile time sizes.
    MemRefType type = cast<MemRefType>(op.getType());
    os << "(Kokkos::view_alloc(Kokkos::WithoutInitializing, \"" << op.getSymName() << "\"));\n";
    auto maybeValue = op.getInitialValue();
    if(maybeValue)
    {
      MemRefType memrefType = op.getType();
      int64_t span = KokkosCppEmitter::getMemrefSpan(memrefType);
      //This view has intializing data (span elements)
      os << "Kokkos::View<";
      if(failed(emitType(op.getLoc(), type.getElementType())))
        return failure();
      os << "*> tempDst(" << op.getSymName() << ".data(), " << span << ");\n";
      os << "Kokkos::View<";
      if(failed(emitType(op.getLoc(), type.getElementType())))
        return failure();
      os << "*, Kokkos::HostSpace> tempSrc(" << op.getSymName() << "_initial, " << span << ");\n";
      os << "Kokkos::deep_copy(exec_space(), tempDst, tempSrc);\n";
    }
    os.unindent();
    os << "}\n";
  }
  os.unindent();
  os << "}\n\n";
  os << "extern \"C\" void lapis_finalize()\n";
  os << "{\n";
  os.indent();
  for(auto& op : globalViews)
  {
    os << op.getSymName() << " = ";
    if(failed(emitType(op.getLoc(), op.getType())))
      return failure();
    os << "();\n";
  }
  os << "Kokkos::finalize();\n";
  os.unindent();
  os << "}\n";
  return success();
}

LogicalResult KokkosCppEmitter::emitPythonBoilerplate(bool isLastKernel)
{
  *py_os << "import ctypes\n";
  *py_os << "import numpy\n";
  *py_os << "class LAPISModule:\n";
  *py_os << "  def __init__(self, libPath):\n";
  *py_os << "    print('Hello from LAPISModule.__init__!')\n";
  *py_os << "    self.libHandle = ctypes.CDLL(libPath)\n";
  // Do all initialization immediately
  *py_os << "    print('Initializing module.')\n";
  *py_os << "    self.libHandle.lapis_initialize()\n";
  *py_os << "    print('Done initializing module.')\n";
  *py_os << "  def __del__(self):\n";
  *py_os << "    print('Finalizing module.')\n";
  if(isLastKernel)
    *py_os << "    self.libHandle.lapis_finalize()\n";
  //From here, just function wrappers are emitted as class members.
  //Indent now for all of them.
  py_os->indent();
  return success();
}

LogicalResult KokkosCppEmitter::emitType(Location loc, Type type, bool forSparseRuntime) {
  if (auto iType = dyn_cast<IntegerType>(type)) {
    switch (iType.getWidth()) {
    case 1:
      return (os << "bool"), success();
    case 8:
    case 16:
    case 32:
    case 64:
      if (shouldMapToUnsigned(iType.getSignedness()))
        return (os << "uint" << iType.getWidth() << "_t"), success();
      else
        return (os << "int" << iType.getWidth() << "_t"), success();
    default:
      return emitError(loc, "cannot emit integer type ") << type;
    }
  }
  if (auto fType = dyn_cast<FloatType>(type)) {
    switch (fType.getWidth()) {
    case 32:
      return (os << "float"), success();
    case 64:
      return (os << "double"), success();
    default:
      return emitError(loc, "cannot emit float type ") << type;
    }
  }
  if (auto iType = dyn_cast<IndexType>(type))
    return (os << "size_t"), success();
  if (auto tType = dyn_cast<TensorType>(type)) {
    return emitError(loc, "cannot directly emit tensor type, should be lowered to memref");
  }
  if (auto tType = dyn_cast<TupleType>(type))
    return emitTupleType(loc, tType.getTypes());
  if (auto mrType = dyn_cast<MemRefType>(type)) {
    if (forSparseRuntime) {
      os << "StridedMemRefType<";
      if (failed(emitType(loc, mrType.getElementType())))
        return failure();
      os << ", " << mrType.getShape().size() << ">";
    }
    else {
      // Default to host space for these declarations.
      return emitMemrefType(loc, mrType, kokkos::MemorySpace::Host);
    }
    return success();
  }
  if (auto mrType = dyn_cast<UnrankedMemRefType>(type)) {
    return emitMemrefType(loc, mrType, kokkos::MemorySpace::Host);
  }
  if (auto pType = dyn_cast<LLVM::LLVMPointerType>(type)) {
    // LLVMPointerType is untyped
    os << "void*";
    return success();
  }
  return emitError(loc, "cannot emit type ") << type << "\n";
}

LogicalResult KokkosCppEmitter::emitTypes(Location loc, ArrayRef<Type> types, bool forSparseRuntime) {
  switch (types.size()) {
  case 0:
    os << "void";
    return success();
  case 1:
    return emitType(loc, types.front(), forSparseRuntime);
  default:
    return emitTupleType(loc, types);
  }
}

LogicalResult KokkosCppEmitter::emitMemrefType(Location loc, MemRefType type, kokkos::MemorySpace space)
{
  if(space == kokkos::MemorySpace::DualView) {
    os << "LAPIS::DualView<";
    if (failed(emitType(loc, type.getElementType())))
      return failure();
    for(auto extent : type.getShape()) {
      if(type.hasStaticShape()) {
          os << '[' << extent << ']';
      }
      else {
          os << '*';
      }
    }
    os << ", Kokkos::LayoutRight>";
  }
  else {
    os << "Kokkos::View<";
    if (failed(emitType(loc, type.getElementType())))
      return failure();
    for(auto extent : type.getShape()) {
      if(type.hasStaticShape()) {
          os << '[' << extent << ']';
      }
      else {
          os << '*';
      }
    }
    os << ", Kokkos::LayoutRight, ";
    if(space == kokkos::MemorySpace::Device)
      os << "Kokkos::DefaultExecutionSpace";
    else
      os << "Kokkos::DefaultHostExecutionSpace";
    os << ">";
  }
  return success();
}

LogicalResult KokkosCppEmitter::emitMemrefType(Location loc, UnrankedMemRefType type, kokkos::MemorySpace space)
{
  if(space == kokkos::MemorySpace::DualView) {
    os << "LAPIS::DualView<";
    if (failed(emitType(loc, type.getElementType())))
      return failure();
    os << "*>";
  }
  else {
    os << "Kokkos::View<";
    if (failed(emitType(loc, type.getElementType())))
      return failure();
    os << "*, ";
    if(space == kokkos::MemorySpace::Device)
      os << "Kokkos::DefaultExecutionSpace";
    else
      os << "Kokkos::DefaultHostExecutionSpace";
    os << ">";
  }
  return success();
}

LogicalResult KokkosCppEmitter::emitTupleType(Location loc, ArrayRef<Type> types) {
  os << "std::tuple<";
  if (failed(interleaveCommaWithError(
          types, os, [&](Type type) { return emitType(loc, type); })))
    return failure();
  os << ">";
  return success();
}

inline void pauseForDebugger()
{
#ifdef __unix__
  std::cout << "Starting Kokkos emitter on process " << getpid() << '\n';
  std::cout << "Optionally attach debugger now, then press <Enter> to continue: ";
  std::cin.get();
#else
  std::cerr << "Can't pause for debugging on non-POSIX system\n";
#endif
}

static void emitCppBoilerplate(KokkosCppEmitter &emitter, bool enablePythonWrapper, bool enableSparseSupport)
{
  // To edit the boilerplate to include in LAPIS generated programs:
  // - edit LAPISSupport.hpp (in this directory)
  // - run updateLAPISSupport.sh (turn the header into a C++ string literal)
  emitter <<
    #include "LAPISSupportFormatted.hpp"
    ;
}

//Version for when we are just emitting C++
LogicalResult kokkos::translateToKokkosCpp(Operation *op, raw_ostream &os, bool enableSparseSupport) {
  //Uncomment to pause so you can attach debugger
  //pauseForDebugger();
  std::string cppDeclBuffer;
  std::string cppBuffer;
  llvm::raw_string_ostream cppDeclStream(cppDeclBuffer);
  llvm::raw_string_ostream cppStream(cppBuffer);
  KokkosCppEmitter emitter(cppDeclStream, cppStream, enableSparseSupport);
  emitCppBoilerplate(emitter, false, enableSparseSupport);
  //Emit the actual module (global variables and functions)
  if(failed(emitter.emitOperation(*op, /*trailingSemicolon=*/false)))
    return failure();
  // Emit the init and finalize function definitions.
  if (failed(emitter.emitInitAndFinalize()))
    return failure();
  // Now emit the contents of the two cpp streams into os.
  os << cppDeclBuffer << cppBuffer;
  return success();
}

//Version for when we are emitting both C++ and Python wrappers
LogicalResult kokkos::translateToKokkosCpp(Operation *op, raw_ostream &os, raw_ostream &py_os, bool enableSparseSupport, bool useHierarchical, bool isLastKernel) {
  //Uncomment to pause so you can attach debugger
  //pauseForDebugger();
  std::string cppDeclBuffer;
  std::string cppBuffer;
  llvm::raw_string_ostream cppDeclStream(cppDeclBuffer);
  llvm::raw_string_ostream cppStream(cppBuffer);
  KokkosCppEmitter emitter(cppDeclStream, cppStream, py_os, enableSparseSupport);
  //Emit the C++ boilerplate to os
  emitCppBoilerplate(emitter, true, enableSparseSupport);
  //Emit the ctypes boilerplate to py_os first - function wrappers need to come after this.
  if(failed(emitter.emitPythonBoilerplate(isLastKernel)))
      return failure();
  //Global preamble.
  //Emit the actual module (global variables and functions)
  if(failed(emitter.emitOperation(*op, /*trailingSemicolon=*/false)))
    return failure();
  //Emit the init and finalize function definitions.
  if(failed(emitter.emitInitAndFinalize()))
    return failure();
  os << cppDeclBuffer << cppBuffer;
  return success();
}


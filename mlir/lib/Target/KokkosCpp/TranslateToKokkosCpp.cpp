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
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "lapis/Dialect/Kokkos/IR/KokkosDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
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
  explicit KokkosCppEmitter(raw_ostream &decl_os, raw_ostream &os);
  explicit KokkosCppEmitter(raw_ostream &decl_os, raw_ostream &os, raw_ostream& py_os);

  /// Emits attribute or returns failure.
  LogicalResult emitAttribute(Location loc, Attribute attr);

  /// Emits operation 'op' with/without training semicolon or returns failure.
  LogicalResult emitOperation(Operation &op, bool trailingSemicolon);

  /// Emits the functions lapis_initialize() and lapis_finalize()
  /// These are responsible for init/finalize of Kokkos, and allocation/initialization/deallocation
  /// of global Kokkos::Views.
  LogicalResult emitInitAndFinalize(bool finalizeKokkos);

  void emitCppBoilerplate();
  void emitPythonBoilerplate();

  /// Emits type 'type' or returns failure.
  /// If forSparseRuntime, the emitted type is compatible with PyTACO runtime and the sparse support library.
  /// In particular, memrefs use StridedMemRefType instead of Kokkos::View.
  //
  /// If !forSparseRuntime, then memrefs are represented as host Kokkos::Views.
  LogicalResult emitType(Location loc, Type type, bool forSparseRuntime = false);

  // Emit a memref type as a Kokkos::View, with the given memory space.
  LogicalResult emitMemrefType(Location loc, MemRefType type, kokkos::MemorySpace space);
  LogicalResult emitMemrefType(Location loc, UnrankedMemRefType type, kokkos::MemorySpace space);

  LogicalResult emitStridedMemrefType(Location loc, MemRefType type, kokkos::MemorySpace space);

  /// Emits array of types as a std::tuple of the emitted types.
  /// - emits void for an empty array;
  /// - emits the type of the only element for arrays of size one;
  /// - emits a std::tuple otherwise;
  /// See emitType(...) for behavior of supportFunction.
  LogicalResult emitTypes(Location loc, ArrayRef<Type> types, bool forSparseRuntime = false);

  // Emit array of types, specifically for normal (non-extern) function results
  LogicalResult emitFuncResultTypes(Location loc, ArrayRef<Type> types);

  /// Emits array of types as a std::tuple of the emitted types independently of
  /// the array size.
  LogicalResult emitTupleType(Location loc, ArrayRef<Type> types);

  /// Emits a variable declaration for a result of an operation.
  LogicalResult emitVariableDeclaration(Location loc, Value result, bool trailingSemicolon);

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

  /// Declare the device and host views for a DualView
  /// This should be added after most operations that produce a DualView result,
  /// since it places the device and host views in the same scope as the DualView.
  void declareDeviceHostViews(Value val) {
    auto name = getOrCreateName(val);
    *this << "auto " << name << "_d = " << name << ".device_view();\n";
    *this << "auto " << name << "_h = " << name << ".host_view();\n";
  }

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

  /// Returns the currently selected C++ output stream.
  raw_indented_ostream &ostream() { return *current_os; };

  void indent() {current_os->indent();}
  void unindent() {current_os->unindent();}

  bool emittingPython() {return py_os.get() != nullptr;}

  /// Returns the Python output stream.
  raw_indented_ostream &py_ostream() { return *py_os.get(); };

  //This lets the emitter act like a stream (writes to the C++ file)
  template<typename T>
  raw_indented_ostream& operator<<(const T& t)
  {
    *current_os << t;
    return *current_os;
  }

  void selectMainCppStream() {
    current_os = &os;
  }

  void selectDeclCppStream() {
    current_os = &decl_os;
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

  raw_indented_ostream* current_os;

  /// Python output stream to emit to.
  std::shared_ptr<raw_indented_ostream> py_os;

  /// Map from value to name of C++ variable that contain the name.
  ValueMapper valueMapper;

  /// Map from block to name of C++ label.
  BlockMapper blockMapper;

  /// The number of values in the current scope. This is used to declare the
  /// names of values in a scope.
  std::stack<int64_t> valueInScopeCount;
  std::stack<int64_t> labelInScopeCount;

  int structCount = 0;

  //Bookeeping for scalar constants (individual integer and floating-point values)
  mutable llvm::DenseMap<Value, arith::ConstantOp> scalarConstants;

  // Mapping from unique LLVMStructTypes appearing in program to C++ struct name.
  mutable llvm::DenseMap<LLVM::LLVMStructType, std::string> structTypes;

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

  void ensureTypeDeclared(Location loc, Type t);
};
} // namespace

static LogicalResult printConstantOp(KokkosCppEmitter &emitter, Operation *operation,
                                     Attribute value) {
  OpResult result = operation->getResult(0);

  // Emit a variable declaration for an emitc.constant op without value.
  if (auto oAttr = dyn_cast<emitc::OpaqueAttr>(value)) {
    if (oAttr.getValue().empty())
      // The semicolon gets printed by the emitOperation function.
      return emitter.emitVariableDeclaration(operation->getLoc(), result,
                                             /*trailingSemicolon=*/false);
  }

  // Emit a variable declaration.
  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  return emitter.emitAttribute(operation->getLoc(), value);
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    memref::GlobalOp op) {
  // Don't emit op at all if never used.
  if(!kokkos::isGlobalUsed(op))
    return success();
  //Emit the View declaration first. 
  //NOTE: using the GlobalOp's symbol name instead of a name generated for the current scope,
  //because GlobalOp does not produce a Result.
  kokkos::MemorySpace space = kokkos::getMemSpace(op);
  if(failed(emitter.emitMemrefType(op.getLoc(), op.getType(), space)))
    return failure();
  emitter << ' ' << op.getSymName() << ";\n";
  //Note: module-wide initialization will be responsible for allocating and copying the initializing data (if any).
  //Then module-wide finalization will deallocate (to avoid Kokkos warning about dealloc after finalize).
  auto maybeValue = op.getInitialValue();
  if(maybeValue)
  {
    auto memrefType = op.getType();
    //For constants (initialized views), keep the actual data in a 1D array (with a related name).
    if(failed(emitter.emitType(op.getLoc(), memrefType.getElementType())))
      return failure();
    if(!memrefType.hasStaticShape())
      return op.emitError("GlobalOp must have static shape");
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
  // Shallow copy in local scope. Can't reference host global in device code
  emitter << "auto " << emitter.getOrCreateName(op.getResult()) << " = " << op.getName() << ";\n";
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
    emitter << "(Kokkos::view_alloc(Kokkos::WithoutInitializing, \"" << name << "\")";
  // If ANY dim is dynamic, we use ALL dynamic dimensions for the Kokkos::View.
  // This is because Kokkos/C++ limit the orders that static and dynamic dimensions can go in the type,
  // but MLIR allows all orderings.
  if(!type.hasStaticShape()) {
    int dynSizeIndex = 0;
    for(int64_t staticSize : type.getShape()) {
      emitter << ", ";
      if(staticSize < 0) {
        // Output the next dynamic size
        if(failed(emitter.emitValue(op.getDynamicSizes()[dynSizeIndex++])))
          return failure();
      }
      else {
        // Output the static size
        emitter << staticSize;
      }
    }
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
  Value toDealloc = op.getOperand();
  kokkos::MemorySpace space = kokkos::getMemSpace(toDealloc);
  if(failed(emitter.emitValue(toDealloc)))
    return failure();
  if(space == kokkos::MemorySpace::DualView) {
    // For DualView, call method to free both host and device views
    emitter << ".deallocate()";
  }
  else {
    // For Kokkos::View, free by assigning a default-constructed instance
    emitter << " = ";
    if(failed(emitter.emitMemrefType(op.getLoc(), dyn_cast<MemRefType>(toDealloc.getType()), space)))
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

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    memref::ReinterpretCastOp op) {
  Value result = op.getResult();
  Value source = op.getSource();
  auto resultName = emitter.getOrCreateName(result);
  MemRefType resultType = dyn_cast<MemRefType>(result.getType());
  int sourceRank = resultType.getRank();
  auto space = kokkos::getMemSpace(result);
  // An OpFoldResult is just a variant<Value, Attribute>
  // (either a runtime value or a static constant).
  // Need this because op's offset, sizes and strides
  // can have mixed runtime and static values.
  auto emitOpFoldResult = [&](OpFoldResult ofr) -> LogicalResult {
    if(ofr.is<Attribute>()) {
      Attribute attr = ofr.get<Attribute>();
      if(failed(emitter.emitAttribute(op.getLoc(), attr)))
        return failure();
    }
    else {
      Value value = ofr.get<Value>();
      if(failed(emitter.emitValue(value)))
        return failure();
    }
    return success();
  };
  auto sizes = op.getConstifiedMixedSizes();
  auto strides = op.getConstifiedMixedStrides();
  OpFoldResult offset = op.getConstifiedMixedOffset();
  auto emitOffset = [&]() -> LogicalResult {
    return emitOpFoldResult(offset);
  };
  auto emitSize = [&](int dim) -> LogicalResult {
    return emitOpFoldResult(sizes[dim]);
  };
  auto emitStride = [&](int dim) -> LogicalResult {
    return emitOpFoldResult(strides[dim]);
  };
  // In general, the result of this op is a strided view.
  // create the correct LayoutStride object and construct unmanaged views from it.
  emitter << "Kokkos::LayoutStride " << resultName << "_layout(";
  // LayoutStride constructor takes alternating extent, stride for each dimension.
  for(int i = 0; i < sourceRank; i++) {
    if(i)
      emitter << ", ";
    if(failed(emitSize(i)))
      return failure();
    emitter << ", ";
    if(failed(emitStride(i)))
      return failure();
  }
  emitter << ");\n";
  if(space == kokkos::MemorySpace::DualView) {
    if(failed(emitter.emitStridedMemrefType(op.getLoc(), resultType, kokkos::MemorySpace::Host))) {
      return failure();
    }
    emitter << " " << resultName << "_host(";
    if (failed(emitter.emitValue(source)))
      return failure();
    emitter << "_h.data() + ";
    if (failed(emitOffset()))
      return failure();
    emitter << ", " << resultName << "_layout);\n";
    if(failed(emitter.emitStridedMemrefType(op.getLoc(), resultType, kokkos::MemorySpace::Device))) {
      return failure();
    }
    emitter << " " << resultName << "_device(";
    if (failed(emitter.emitValue(source)))
      return failure();
    emitter << "_d.data() + ";
    if (failed(emitOffset()))
      return failure();
    emitter << ", " << resultName << "_layout);\n";
    if(failed(emitter.emitStridedMemrefType(op.getLoc(), resultType, kokkos::MemorySpace::DualView))) {
      return failure();
    }
    emitter << " " << resultName << "(" << resultName << "_device, " << resultName << "_host, ";
    if (failed(emitter.emitValue(source)))
      return failure();
    emitter << ");\n";
  }
  else {
    if(failed(emitter.emitStridedMemrefType(op.getLoc(), resultType, space))) {
      return failure();
    }
    emitter << " " << resultName << "(";
    if (failed(emitter.emitValue(source)))
      return failure();
    emitter << ".data() + ";
    if (failed(emitOffset()))
      return failure();
    emitter << ", " << resultName << "_layout);\n";
  }
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    memref::ExtractStridedMetadataOp op) {
  auto base = op.getBaseBuffer();
  auto sizes = op.getSizes();
  auto strides = op.getStrides();
  MemRefType sourceType = cast<MemRefType>(op.getSource().getType());
  MemRefType destType = cast<MemRefType>(base.getType());
  int rank = sourceType.getRank();
  auto sourceSpace = kokkos::getMemSpace(op.getSource());
  auto destSpace = kokkos::getMemSpace(base);
  auto loc = op.getLoc();
  if(sourceSpace != destSpace) {
    return op.emitError("Source and base buffer (result) memrefs should have the same Kokkos memory space");
  }
  // op produces 2 + 2*rank results:
  // - base buffer (rank-0 memref)
  // - offset into base buffer
  // - sizes (one per rank)
  // - strides (one per rank)
  // In the kokkos implementation of memrefs, the offset is always 0.
  if(!op.getOffset().use_empty()) {
    // Only declare the value if it actually has at least one use
    emitter << "size_t " << emitter.getOrCreateName(op.getOffset()) << " = 0;\n";
  }
  for(int i = 0; i < rank; i++) {
    if(!sizes[i].use_empty())
      emitter << "size_t " << emitter.getOrCreateName(sizes[i]) << " = " << emitter.getOrCreateName(op.getSource()) << ".extent(" << i << ");\n";
  }
  for(int i = 0; i < rank; i++) {
    if(!strides[i].use_empty())
      emitter << "size_t " << emitter.getOrCreateName(strides[i]) << " = " << emitter.getOrCreateName(op.getSource()) << ".stride(" << i << ");\n";
  }
  // Base: get a rank 0 view or DualView with the same pointer(s) as source.
  if(destSpace == kokkos::MemorySpace::DualView) {
    // Declare the device and host views first
    if(failed(emitter.emitMemrefType(loc, destType, kokkos::MemorySpace::Device)))
      return failure();
    emitter << " " << emitter.getOrCreateName(base) << "_d_temp(" << emitter.getOrCreateName(op.getSource()) << ".device_view().data());\n";
    if(failed(emitter.emitMemrefType(loc, destType, kokkos::MemorySpace::Host)))
      return failure();
    emitter << " " << emitter.getOrCreateName(base) << "_h_temp(" << emitter.getOrCreateName(op.getSource()) << ".host_view().data());\n";
    if(failed(emitter.emitMemrefType(loc, destType, destSpace)))
      return failure();
    emitter << " " << emitter.getOrCreateName(base);
    emitter << "(" << emitter.getOrCreateName(base) << "_d_temp, ";
    emitter << emitter.getOrCreateName(base) << "_h_temp, " << emitter.getOrCreateName(op.getSource()) << ")";
  }
  else {
    if(failed(emitter.emitMemrefType(loc, destType, destSpace)))
      return failure();
    emitter << " " << emitter.getOrCreateName(base) << "(" << emitter.getOrCreateName(op.getSource()) << ".";
    if(destSpace == kokkos::MemorySpace::Device)
      emitter << "device_view";
    else
      emitter << "host_view";
    emitter << "().data())";
  }
  return success();
}

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
      emitter << "_d";
    else
      emitter << "_h";
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
      emitter << "_d";
    else
      emitter << "_h";
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
  // TODO: deal with case where source and target have different layouts and different spaces.
  auto srcSpace = kokkos::getMemSpace(op.getSource());
  auto dstSpace = kokkos::getMemSpace(op.getTarget());
  if(srcSpace != kokkos::MemorySpace::DualView && dstSpace != kokkos::MemorySpace::DualView) {
    // Neither src nor dst are DualView, so just do a standard deep_copy
    // TODO: could use async deep copies unless it's host->host?
    emitter << "Kokkos::deep_copy(";
    if(failed(emitter.emitValue(op.getTarget())))
      return failure();
    emitter << ", ";
    if(failed(emitter.emitValue(op.getSource())))
      return failure();
    emitter << ")";
  }
  else if(srcSpace == kokkos::MemorySpace::DualView && dstSpace == kokkos::MemorySpace::DualView) {
    // Both src and dst are DualView.
    // At runtime figure out where src is up-to-date (preferring device if both spaces are up-to-date). Then:
    // - sync dst to that space, since dst cannot be modified in both spaces at once
    // - copy the data to that space of dst
    // - mark dst modified in that space
    emitter << "if(";
    if(failed(emitter.emitValue(op.getSource())))
      return failure();
    emitter << ".modifiedHost())\n";
    emitter << "{\n";
    emitter.indent();
    if(failed(emitter.emitValue(op.getTarget())))
      return failure();
    emitter << ".syncHost();\n";
    emitter << "Kokkos::deep_copy(";
    if(failed(emitter.emitValue(op.getTarget())))
      return failure();
    emitter << "_h, ";
    if(failed(emitter.emitValue(op.getSource())))
      return failure();
    emitter << "_h);";
    if(failed(emitter.emitValue(op.getTarget())))
      return failure();
    emitter << ".modifyHost();\n";
    emitter.unindent();
    emitter <<" } else {\n";
    emitter.indent();
    if(failed(emitter.emitValue(op.getTarget())))
      return failure();
    emitter << ".syncDevice();\n";
    emitter << "Kokkos::deep_copy(";
    if(failed(emitter.emitValue(op.getTarget())))
      return failure();
    emitter << "_d, ";
    if(failed(emitter.emitValue(op.getSource())))
      return failure();
    emitter << "_d);";
    if(failed(emitter.emitValue(op.getTarget())))
      return failure();
    emitter << ".modifyDevice();\n";
    emitter.unindent();
    emitter <<" }\n";
  }
  else if(srcSpace == kokkos::MemorySpace::DualView) {
    // src is DualView but dst isn't.
    if(failed(emitter.emitValue(op.getSource())))
      return failure();
    emitter << ".sync";
    if(dstSpace == kokkos::MemorySpace::Device)
      emitter << "Device();\n";
    else
      emitter << "Host();\n";
    emitter << "Kokkos::deep_copy(";
    if(failed(emitter.emitValue(op.getTarget())))
      return failure();
    emitter << ", ";
    if(failed(emitter.emitValue(op.getSource())))
      return failure();
    if(dstSpace == kokkos::MemorySpace::Device)
      emitter << "_d);\n";
    else
      emitter << "_h);\n";
  }
  else {
    // dst is DualView but src isn't.
    if(failed(emitter.emitValue(op.getTarget())))
      return failure();
    bool isDevice = srcSpace == kokkos::MemorySpace::Device;
    emitter << ".sync" << (isDevice ? "Device" : "Host") << "();\n";
    emitter << "Kokkos::deep_copy(";
    if(failed(emitter.emitValue(op.getTarget())))
      return failure();
    emitter << "." << (isDevice ? "device" : "host") << "_view, ";
    if(failed(emitter.emitValue(op.getSource())))
      return failure();
    emitter << ");";
    if(failed(emitter.emitValue(op.getTarget())))
      return failure();
    emitter << ".modify" << (isDevice ? "Device" : "Host") << "();\n";
  }
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    memref::SubViewOp op) {
  Value result = op.getResult();
  Value source = op.getSource();
  auto resultName = emitter.getOrCreateName(result);
  MemRefType resultType = dyn_cast<MemRefType>(result.getType());
  int sourceRank = resultType.getRank();
  auto space = kokkos::getMemSpace(result);
  const bool useDynamicOffset = !op.getOffsets().empty();
  const bool useDynamicSizes = !op.getSizes().empty();
  const bool useDynamicStrides = !op.getStrides().empty();
  auto emitOffset = [&](int dim) -> LogicalResult {
    if (useDynamicOffset) {
      if (failed(emitter.emitValue(op.getOffsets()[dim])))
        return failure();
    } else
      emitter << op.getStaticOffsets()[dim];
    return success();
  };
  auto emitSize = [&](int dim) -> LogicalResult {
    if (useDynamicSizes) {
      if (failed(emitter.emitValue(op.getSizes()[dim])))
        return failure();
    } else
      emitter << op.getStaticSizes()[dim];
    return success();
  };
  auto emitStride = [&](int dim) -> LogicalResult {
    if (useDynamicStrides) {
      if (failed(emitter.emitValue(op.getStrides()[dim])))
        return failure();
    } else
      emitter << op.getStaticStrides()[dim];
    return success();
  };
  // In general, the result of this op is a strided view.
  // create the correct LayoutStride object and construct unmanaged views from it.
  emitter << "Kokkos::LayoutStride " << resultName << "_layout(";
  // LayoutStride constructor takes alternating extent, stride for each dimension.
  for(int i = 0; i < sourceRank; i++) {
    if(i)
      emitter << ", ";
    if(failed(emitSize(i)))
      return failure();
    emitter << ", ";
    if(failed(emitStride(i)))
      return failure();
  }
  emitter << ");\n";
  if(space == kokkos::MemorySpace::DualView) {
    if(failed(emitter.emitStridedMemrefType(op.getLoc(), resultType, kokkos::MemorySpace::Host))) {
      return failure();
    }
    emitter << " " << resultName << "_host(";
    if (failed(emitter.emitValue(source)))
      return failure();
    emitter << "_h.data() + ";
    for(int i = 0; i < sourceRank; i++) {
      if(i)
        emitter << " + ";
      if(failed(emitOffset(i)))
        return failure();
      emitter << " * ";
      if (failed(emitter.emitValue(source)))
        return failure();
      emitter << "_h.stride_" << i << "()";
    }
    emitter << ", " << resultName << "_layout);\n";
    if(failed(emitter.emitStridedMemrefType(op.getLoc(), resultType, kokkos::MemorySpace::Device))) {
      return failure();
    }
    emitter << " " << resultName << "_device(";
    if (failed(emitter.emitValue(source)))
      return failure();
    emitter << "_d.data() + ";
    for(int i = 0; i < sourceRank; i++) {
      if(i)
        emitter << " + ";
      if(failed(emitOffset(i)))
        return failure();
      emitter << " * ";
      if (failed(emitter.emitValue(source)))
        return failure();
      emitter << "_d.stride_" << i << "()";
    }
    emitter << ", " << resultName << "_layout);\n";
    if(failed(emitter.emitStridedMemrefType(op.getLoc(), resultType, kokkos::MemorySpace::DualView))) {
      return failure();
    }
    emitter << " " << resultName << "(" << resultName << "_device, " << resultName << "_host, ";
    if (failed(emitter.emitValue(source)))
      return failure();
    emitter << ");\n";
  }
  else {
    if(failed(emitter.emitStridedMemrefType(op.getLoc(), resultType, space))) {
      return failure();
    }
    emitter << " " << resultName << "(";
    if (failed(emitter.emitValue(source)))
      return failure();
    emitter << ".data() + ";
    for(int i = 0; i < sourceRank; i++) {
      if(i)
        emitter << " + ";
      if(failed(emitOffset(i)))
        return failure();
      emitter << " * ";
      if (failed(emitter.emitValue(source)))
        return failure();
      emitter << ".stride_" << i << "()";
    }
    emitter << ", " << resultName << "_layout);\n";
  }
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    memref::ReshapeOp op) {
  Value result = op.getResult();
  Value source = op.getSource();
  // Shape is a statically sized, 1D memref of integers defining result shape
  Value shape = op.getShape();
  if(kokkos::getMemSpace(shape) != kokkos::MemorySpace::Host) {
    return op.emitError("shape memref is used in device code, this case must be added to Kokkos emitter");
  }
  // Using that simplifying assumption, can easily read shape values on host
  auto resultName = emitter.getOrCreateName(result);
  MemRefType resultType = dyn_cast<MemRefType>(result.getType());
  if(!resultType) {
    return op.emitError("not supporting unranked result yet, this case must be added to Kokkos emitter");
  }
  int resultRank = resultType.getRank();
  auto space = kokkos::getMemSpace(result);
  if(space == kokkos::MemorySpace::DualView) {
    if(failed(emitter.emitMemrefType(op.getLoc(), resultType, kokkos::MemorySpace::Host))) {
      return failure();
    }
    emitter << " " << resultName << "_host(";
    if (failed(emitter.emitValue(source)))
      return failure();
    emitter << "_h.data()";
    for(int i = 0; i < resultRank; i++) {
      emitter << ", ";
      if(failed(emitter.emitValue(shape)))
        return failure();
      emitter << "(" << i << ")";
    }
    emitter << ");\n";
    if(failed(emitter.emitMemrefType(op.getLoc(), resultType, kokkos::MemorySpace::Device))) {
      return failure();
    }
    emitter << " " << resultName << "_device(";
    if (failed(emitter.emitValue(source)))
      return failure();
    emitter << "_d.data()";
    for(int i = 0; i < resultRank; i++) {
      emitter << ", ";
      if(failed(emitter.emitValue(shape)))
        return failure();
      emitter << "(" << i << ")";
    }
    emitter << ");\n";
    if(failed(emitter.emitMemrefType(op.getLoc(), resultType, kokkos::MemorySpace::DualView))) {
      return failure();
    }
    emitter << " " << resultName << "(" << resultName << "_device, " << resultName << "_host, ";
    if (failed(emitter.emitValue(source)))
      return failure();
    emitter << ")";
  }
  else {
    if(failed(emitter.emitMemrefType(op.getLoc(), resultType, space))) {
      return failure();
    }
    emitter << " " << resultName << "(";
    if (failed(emitter.emitValue(source)))
      return failure();
    emitter << ".data()";
    for(int i = 0; i < resultRank; i++) {
      emitter << ", ";
      if(failed(emitter.emitValue(shape)))
        return failure();
      emitter << "(" << i << ")";
    }
    emitter << ")";
  }
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    memref::CastOp op) {
  // TODO: IR from pipeline usually seems to use this only for converting
  // from static to dynamic dimensions.
  // However, it can also go the other direction and this needs additional logic
  // for that case.
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
      // arg's memory space should be either Host or DualView
      auto argMemSpace = kokkos::getMemSpace(arg);
      if(argMemSpace == kokkos::MemorySpace::DualView) {
        os << "_h";
      }
      else if(argMemSpace != kokkos::MemorySpace::Host) {
        return callOp.emitError("Passing memref to support function, whose space is neither host nor DualView");
      }
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
    os << "LAPIS::stridedMemrefToView<";
    if(failed(emitter.emitMemrefType(callOp.getLoc(), dyn_cast<MemRefType>(callOp.getResult(0).getType()), kokkos::MemorySpace::Host)))
      return failure();
    os << ">(" << emitter.getOrCreateName(callOp.getResult(0)) << "_smr)";
    os << ";\n";
    if(resultSpace == kokkos::MemorySpace::DualView) {
      os << emitter.getOrCreateName(callOp.getResult(0)) << ".syncHostOnDestroy();\n";
    }
  }
  os.unindent();
  os << "}";
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
    if (failed(emitter.emitVariableDeclaration(forOp.getLoc(), result,
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
    if (failed(emitter.emitVariableDeclaration(whileOp.getLoc(), afterArg, /*trailingSemicolon=*/true)))
      return failure();
  }
  for (OpResult result : whileOp.getResults()) {
    if (failed(emitter.emitVariableDeclaration(whileOp.getLoc(), result, /*trailingSemicolon=*/true)))
      return failure();
  }

  emitter << "while(true) {\n";
  emitter.indent();

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

  emitter.unindent();
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
    else
      emitter << "Kokkos::DefaultExecutionSpace";
    // For multidimensional policies, use Iterate::Right to maximize locality in common cases (since all views are LayoutRight).
    if(isMDPolicy) {
      emitter << ", Kokkos::Rank<" << op.getNumLoops() << ", Kokkos::Iterate::Right, Kokkos::Iterate::Right>";
    }
    emitter << ">";
  }
  emitter << "(";
  if(op.getParallelLevel() != kokkos::ParallelLevel::RangePolicy)
    emitter << "team, ";
  if(isMDPolicy) {
    if(op.getParallelLevel() == kokkos::ParallelLevel::RangePolicy) {
      // (Device-level) MDRangePolicy takes two N-element tuples (begins, ends)
      emitter << "{";
      for(size_t i = 0; i < op.getNumLoops(); i++) {
        if(i > 0)
          emitter << ", ";
        emitter << "int64_t(0)";
      }
      emitter << "}, ";
      emitter << "{";
      int count = 0;
      for(Value bound : op.getUpperBound()) {
        if(count++)
          emitter << ", ";
        emitter << "int64_t(";
        if(failed(emitter.emitValue(bound)))
          return failure();
        emitter << ")";
      }
      emitter << "}";
    }
    else {
      // But nested MD policies just take the upper bounds as arguments directly
      int count = 0;
      for(Value bound : op.getUpperBound()) {
        if(count++)
          emitter << ", ";
        if(failed(emitter.emitValue(bound)))
          return failure();
      }
    }
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
  std::string vectorLength = "vectorLength_" + emitter.getUniqueIdentifier();
  emitter << "int " << vectorLength << " = ";
  if(failed(emitter.emitValue(op.getVectorLengthHint())))
    return failure();
  emitter << " ? LAPIS::threadParallelVectorLength(";
  if(failed(emitter.emitValue(op.getVectorLengthHint())))
    return failure();
  emitter << ") : 8;\n";
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

static LogicalResult printOperation(KokkosCppEmitter &emitter, scf::IfOp ifOp) {
  raw_indented_ostream &os = emitter.ostream();

  for (OpResult result : ifOp.getResults()) {
    if (failed(emitter.emitVariableDeclaration(ifOp.getLoc(), result,
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
  // Need to replace function names in 2 cases:
  //  1. functionOp is a forward declaration for a sparse support function
  //  2. functionOp's name is "main"
  bool isSupportFunc = emitter.isSparseSupportFunction(functionOp.getName());
  // Does the function provide its results via pointer arguments, preceding the input arguments?
  bool pointerResults = isSupportFunc && emitter.sparseSupportFunctionPointerResults(functionOp.getName());
  auto loc = functionOp.getLoc();
  std::string funcName;
  if(isSupportFunc) {
    funcName = emitter.getSparseSupportFunctionName(functionOp.getName());
  }
  else if (functionOp.getName().str() == "main") {
    // TorchFX exporter names the entry point "main",
    // but this name can't be used in C++
    funcName = "lapis_main";
  }
  else {
    funcName = functionOp.getName().str();
  }
  // We need to declare variables at top if the function has multiple blocks.
  if (functionOp.getBlocks().size() > 1) {
    return functionOp.emitOpError(
        "with multiple blocks needs variables declared at top");
  }
  raw_indented_ostream &os = emitter.ostream();
  // Handle function declarations (empty body). Don't need to give parameters names either.
  if(functionOp.getBody().empty()) {
    //Prevent support lib function names from being mangled
    if(isSupportFunc) {
      os << "#ifndef LAPIS_CPP_DRIVER\n";
      os << "extern \"C\" ";
    }
    if(pointerResults) {
      os << "void ";
    }
    else {
      if(isSupportFunc) {
        if (failed(emitter.emitTypes(loc, functionOp.getFunctionType().getResults(), isSupportFunc)))
          return failure();
      }
      else {
        if (failed(emitter.emitFuncResultTypes(loc, functionOp.getFunctionType().getResults())))
          return failure();
      }
      os << ' ';
    }
    os << funcName << '(';
    if (pointerResults) {
      if (failed(interleaveCommaWithError(functionOp.getFunctionType().getResults(), os,
        [&](Type resultType) -> LogicalResult {
          if (failed(emitter.emitType(loc, resultType, isSupportFunc)))
            return failure();
          // Memrefs are returned by pointer
          if (isSupportFunc && isa<MemRefType>(resultType))
            os << "*";
          return success();
        }))) {
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
        if (failed(emitter.emitType(loc, argType, isSupportFunc)))
          return failure();
        // Memrefs are passed by pointer
        if (isSupportFunc && isa<MemRefType>(argType))
          os << "*";
        return success();
      }))) {
      return failure();
    }
    os << ");\n";
    if(isSupportFunc)
      os << "#endif\n";
    return success();
  }
  // Otherwise, it's a function definition with body.
  KokkosCppEmitter::Scope scope(emitter);
  if (failed(emitter.emitFuncResultTypes(loc, functionOp.getFunctionType().getResults())))
    return failure();
  os << ' ' << funcName;
  os << "(";
  //Make a list of the memref parameters (these will all be DualViews)
  std::vector<BlockArgument> memrefParams;
  if (failed(interleaveCommaWithError(
          functionOp.getArguments(), os,
          [&](BlockArgument arg) -> LogicalResult {
            // Emit normal types (e.g. Kokkos::View<..> or LAPIS::DualView for MemRefType)
            if (MemRefType mrt = dyn_cast<MemRefType>(arg.getType())) {
              // Get the space based on how this argument gets used
              kokkos::MemorySpace space = kokkos::getMemSpace(arg);
              if (failed(emitter.emitMemrefType(loc, mrt, space)))
                return failure();
              memrefParams.push_back(arg);
            }
            else {
              if (failed(emitter.emitType(loc, arg.getType())))
                return failure();
            }
            os << " " << emitter.getOrCreateName(arg);
            return success();
          })))
    return failure();
  os << ") {\n";
  os.indent();

  for(BlockArgument arg : memrefParams) {
    emitter << "auto " << emitter.getOrCreateName(arg) << "_d = " << emitter.getOrCreateName(arg) << ".device_view();\n";
    emitter << "auto " << emitter.getOrCreateName(arg) << "_h = " << emitter.getOrCreateName(arg) << ".host_view();\n";
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
  if(!emitter.emittingPython())
    return success();
  // Finally, create the corresponding wrapper function that is callable from Python (if one is needed)
  // This version of the function:
  //  - has non-mangled name, so easier to call through ctypes
  //  - returns all results through pointers (preceding the input parameters)
  //  - uses StridedMemRefType<T,N> to represent memrefs, instead of Kokkos::View

  os << "extern \"C\" void " << "py_" << funcName << '(';
  // Put the results first: primitives and memrefs are both passed by pointer.
  // Python interface will enforce LayoutRight on all input memrefs.
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
      os << "StridedMemRefType<";
      if (failed(emitter.emitType(loc, memrefType.getElementType())))
        return functionOp.emitError("Failed to emit result type as StridedMemRefType");
      os << ", " << memrefType.getShape().size() << ">** ret" << i;
    }
    else
    {
      //Assuming it is a scalar primitive
      if(failed(emitter.emitType(loc, retType)))
        return functionOp.emitError("Failed to emit non-memref result type");
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
      os << "StridedMemRefType<";
      if (failed(emitter.emitType(loc, memrefType.getElementType())))
        return functionOp.emitError("Failed to emit param type as StridedMemRefType");
      os << ", " << memrefType.getShape().size() << ">* param" << i;
    }
    else
    {
      bool isStruct = isa<LLVM::LLVMStructType>(paramType);
      // Structs are passed by const reference
      if(isStruct) {
        os << "const ";
      }
      if(failed(emitter.emitType(loc, paramType)))
        return functionOp.emitError("Failed to emit non-memref param type");
      if(isStruct) {
        os << "&";
      }
      os << " param" << i;
    }
  }
  os << ")\n";
  os << "{\n";
  os.indent();
  //FOR DEBUGGING THE EMITTED CODE:
  //If uncommented, the following 3 lines make the generated function pause to let user attach a debugger
  //os << "std::cout << \"Starting MLIR function on process \" << getpid() << '\\n';\n";
  //os << "std::cout << \"Optionally attach debugger now, then press <Enter> to continue: \";\n";
  //os << "std::cin.get();\n";
  //Construct an unmanaged, LayoutRight Kokkos::View for each memref input parameter.
  //Note: stridedMemrefToView with LayoutRight will check the strides at runtime,
  //and the python wrapper will use numpy.require to deep-copy the data to the correct
  //layout if it's not already.
  for(size_t i = 0; i < numParams; i++)
  {
    auto paramType = ftype.getInput(i);
    auto memrefType = dyn_cast<MemRefType>(paramType);
    if(memrefType)
    {
      os << "auto param" << i << "_smr = LAPIS::stridedMemrefToView<";
      if(failed(emitter.emitMemrefType(loc, memrefType, kokkos::MemorySpace::Host)))
        return functionOp.emitError("Failed to emit memref type as host view");
      os << ">(*param" << i << ");\n";
    }
  }
  // Emit the call
  if(numResults)
    os << "auto results = ";
  os << funcName << "(";
  for(size_t i = 0; i < numParams; i++)
  {
    if(i != 0)
      os << ", ";
    auto paramType = ftype.getInput(i);
    auto memrefType = dyn_cast<MemRefType>(paramType);
    if(memrefType)
    {
      os << "param" << i << "_smr";
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
    auto retType = ftype.getResult(i);
    auto memrefType = dyn_cast<MemRefType>(retType);
    if(memrefType)
    {
      if(numResults == size_t(1))
        os << "results.syncHost();\n";
      else
        os << "std::get<" << i << ">(results).syncHost();\n";
      os << "**ret" << i << " = LAPIS::viewToStridedMemref(";
      if(numResults == size_t(1))
        os << "results";
      else
        os << "std::get<" << i << ">(results)";
      os << ".host_view());\n";
      // Keep the host view alive until lapis_finalize() is called.
      // Otherwise it would be deallocated as soon as this function returns.
      os << "LAPIS::keepAlive(";
      if(numResults == size_t(1))
        os << "results";
      else
        os << "std::get<" << i << ">(results)";
      os << ".host_view());\n";
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
  // All kernels and deep copies so far have been asynchronous on default instance.
  // Fence to make sure results are ready for Python to access.
  os << "Kokkos::DefaultExecutionSpace().fence();\n";
  os.unindent();
  os << "}\n";
  // Now that the native function (name: "py_" + funcName)
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
  auto getCtypesType = [](Type t) -> std::string
  {
    if(t.isIndex())
      return "ctypes.c_ulong";
    //Note: treating MLIR "signless" integer types as equivalent to unsigned NumPy integers.
    if(t.isSignlessInteger(8) || t.isUnsignedInteger(8))
      return "ctypes.c_ubyte";
    if(t.isSignlessInteger(16) || t.isUnsignedInteger(16))
      return "ctypes.c_ushort";
    if(t.isSignlessInteger(32) || t.isUnsignedInteger(32))
      return "ctypes.c_uint";
    if(t.isSignlessInteger(64) || t.isUnsignedInteger(64))
      return "ctypes.c_ulong";
    if(t.isSignedInteger(8))
      return "ctypes.c_byte";
    if(t.isSignedInteger(16))
      return "ctypes.c_short";
    if(t.isSignedInteger(32))
      return "ctypes.c_int";
    if(t.isSignedInteger(64))
      return "ctypes.c_long";
    if(t.isF16())
      // ctypes doesn't have an fp16/half type
      return "ctypes.c_short";
    if(t.isF32())
      return "ctypes.c_float";
    if(t.isF64())
      return "ctypes.c_double";
    return "";
  };

  // Use this to enforce LayoutRight for input memrefs (zero cost if it already is):
  // arr = numpy.require(arr, dtype=TYPE, requirements=['C'])
  // NOTE: numpy.zeros(shape, dtype=...) already defaults to LayoutRight (and probably most other functions)
  // so in practice this shouldn't usually trigger a deep-copy.
  auto& py_os = emitter.py_ostream();
  //NOTE: this function is a member of the module's class, but py_os is already indented to write methods.
  py_os << "def " << funcName << "(self, ";
  for(size_t i = 0; i < numParams; i++)
  {
    if(i != 0)
      py_os << ", ";
    py_os << "param" << i;
  }
  py_os << "):\n";
  py_os.indent();
  // Enforce types on all dense inputs, including scalars
  //
  // Pointer-typed params are only used for sparse tensors, so there is no equivalent to "require" for them.
  // They need to have the correct data type on input.
  std::function<void(std::string, std::string, int&, Type)> genStructFlatten =
    [&](std::string orig, std::string flat, int& index, Type t)
    {
      if(auto structType = dyn_cast<LLVM::LLVMStructType>(t)) {
        int memIdx = 0;
        for(auto mem : structType.getBody()) {
          std::string memExpr = orig + "[" + std::to_string(memIdx) + "]";
          genStructFlatten(memExpr, flat, index, mem);
          memIdx++;
        }
      }
      else if(auto arrayType = dyn_cast<LLVM::LLVMArrayType>(t)) {
        int n = arrayType.getNumElements();
        py_os << "for i in range(" << n << "):\n";
        py_os.indent();
        py_os << flat << "[" << index << " + i] = " << orig << "[i]\n";
        py_os.unindent();
        index += n;
      }
      else {
        //must be a scalar of the correct type
        py_os << flat << "[" << index++ << "] = " << orig << "\n";
      }
    };
  // Generate an un-flattened nested tuple expression for the given type.
  // This can be constructed directly in the Python return statement.
  std::function<void(std::string, int&, Type)> genStructUnflatten =
    [&](std::string flat, int& index, Type t) -> void
    {
      if(auto structType = dyn_cast<LLVM::LLVMStructType>(t)) {
        py_os << "(";
        int memIdx = 0;
        for(auto mem : structType.getBody()) {
          if(memIdx)
            py_os << ", ";
          genStructUnflatten(flat, index, mem);
          memIdx++;
        }
        py_os << ")";
      }
      else if(auto arrayType = dyn_cast<LLVM::LLVMArrayType>(t)) {
        int n = arrayType.getNumElements();
        py_os << "(";
        for(int i = 0; i < n; i++) {
          if(i)
            py_os << ", ";
          py_os << flat << "[" << index++ << "]";
        }
      }
      else {
        //must be a scalar of the correct type
        py_os << flat << "[" << index++ << "]";
      }
    };
  for(size_t i = 0; i < numParams; i++)
  {
    auto paramType = ftype.getInput(i);
    if(!isa<LLVM::LLVMPointerType>(paramType))
    {
      if(auto memrefType = dyn_cast<MemRefType>(paramType))
      {
        std::string numpyDType = getNumpyType(memrefType.getElementType());
        if(!numpyDType.size())
          return functionOp.emitError("Could not determine corresponding numpy type for memref element type");
        py_os << "param" << i << " = numpy.require(param" << i << ", dtype=" << numpyDType << ", requirements=['C'])\n";
      }
      else if(auto structType = dyn_cast<LLVM::LLVMStructType>(paramType)) {
        // Expect this parameter to be a tuple with the correct structure. Flatten it to a numpy array.
        Type elem = kokkos::getStructElementType(structType);
        if(!elem)
          return functionOp.emitError("Cannot yet pass structs with multiple element types to/from Python");
        int size = kokkos::getStructElementCount(structType);
        std::string numpyDType = getNumpyType(elem);
        if(!numpyDType.size())
          return functionOp.emitError("Could not determine corresponding numpy type for result scalar type");
        py_os << "param_flat" << i << " = numpy.zeros(" << size << ", dtype=" << numpyDType << ")\n";
        int flatIdx = 0;
        genStructFlatten("param" + std::to_string(i), "param_flat" + std::to_string(i), flatIdx, structType);
        // Replace original param with flattened version, as we don't need original anymore
        py_os << "param" << i << " = param_flat" << i << "\n";
      }
      else {
        //Wrap scalar primitives in 1D NumPy array.
        //This gives it the correct type, and lets us use the same ndarray CTypes API as memrefs.
        std::string numpyDType = getNumpyType(paramType);
        if(!numpyDType.size())
          return functionOp.emitError("Could not determine corresponding numpy type for scalar type");
        py_os << "param" << i << " = numpy.array(param" << i << ", dtype=" << numpyDType << ", ndmin=1)\n";
      }
    }
  }
  // Construct outputs
  // Note: by default, numpy.zeros uses LayoutRight
  for(size_t i = 0; i < numResults; i++)
  {
    auto retType = ftype.getResult(i);
    if(auto memrefType = dyn_cast<MemRefType>(retType))
    {
      int rank = memrefType.hasRank() ? memrefType.getShape().size() : 1;
      py_os << "ret" << i << " = ctypes.pointer(ctypes.pointer(rt.make_nd_memref_descriptor(" << rank << ", " << getCtypesType(memrefType.getElementType()) << ")()))\n";
    }
    else if(isa<LLVM::LLVMPointerType>(retType))
    {
      //For pointer results, declare a void* (initially null) and pass its address (void**)
      py_os << "ret" << i << " = ctypes.pointer(ctypes.pointer(ctypes.c_char(0)))\n";
    }
    else if(auto structType = dyn_cast<LLVM::LLVMStructType>(retType)) {
      Type elem = kokkos::getStructElementType(structType);
      if(!elem)
        return functionOp.emitError("Cannot yet pass structs with multiple element types to/from Python");
      int size = kokkos::getStructElementCount(structType);
      std::string numpyDType = getNumpyType(elem);
      if(!numpyDType.size())
        return functionOp.emitError("Could not determine corresponding numpy type for result scalar type");
      py_os << "ret" << i << " = numpy.zeros(" << size << ", dtype=" << numpyDType << ")\n";
    }
    else
    {
      //For scalars, construct a single-element numpy ndarray so that we can use its CTypes API
      std::string numpyDType = getNumpyType(retType);
      if(!numpyDType.size())
        return functionOp.emitError("Could not determine corresponding numpy type for result scalar type");
      py_os << "ret" << i << " = numpy.zeros(1, dtype=" << numpyDType << ")\n";
    }
  }
  // Generate the native call. It always returns void.
  py_os << "self.libHandle.py_" << funcName << "(";
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
    else if(isa<MemRefType>(retType))
    {
      py_os << "ret" << i;
    }
    else
    {
      // numpy array, flattened struct or scalar
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
    if(isa<LLVM::LLVMPointerType>(paramType))
    {
      py_os << "param" << i;
    }
    else if(isa<MemRefType>(paramType))
    {
      //Numpy array (or a scalar from a numpy array)
      py_os << "ctypes.pointer(rt.get_ranked_memref_descriptor(param" << i << "))";
    }
    else
    {
      //Numpy array (or a scalar from a numpy array)
      py_os << "param" << i << ".ctypes.data_as(ctypes.c_void_p)";
    }
  }
  py_os << ")\n";
  // Finally, generate the return statement.
  // Note that in Python, a 1-elem tuple is equivalent to scalar.
  if(numResults)
  {
    py_os << "return (";
    for(size_t i = 0; i < numResults; i++)
    {
      if(i != 0)
        py_os << ", ";
      auto retType = ftype.getResult(i);
      if(isa<LLVM::LLVMPointerType>(retType))
      {
        py_os << "ret" << i;
      }
      else if(isa<MemRefType>(retType))
      {
        py_os << "rt.ranked_memref_to_numpy(ret" << i << "[0])";
      }
      else if(auto structType = dyn_cast<LLVM::LLVMStructType>(retType)) {
        int idx = 0;
        genStructUnflatten("ret" + std::to_string(i), idx, structType);
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

KokkosCppEmitter::KokkosCppEmitter(raw_ostream& decl_os_, raw_ostream& os_)
    : decl_os(decl_os_), os(os_), current_os(&os), py_os(nullptr) {
  valueInScopeCount.push(0);
  labelInScopeCount.push(0);
  registerRuntimeSupportFunctions();
}

KokkosCppEmitter::KokkosCppEmitter(raw_ostream& decl_os_, raw_ostream& os_, raw_ostream& py_os_)
    : decl_os(decl_os_), os(os_), current_os(&os) {
  this->py_os = std::make_shared<raw_indented_ostream>(py_os_); 
  valueInScopeCount.push(0);
  labelInScopeCount.push(0);
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
       {"getPartitions",       "mpi_getPartitions",   "updateSlice",
        "mpi_updateSliceWithActiveMask",
        "mpi_setSlice",        "sparseCoordinates0",  "sparseCoordinates8",
        "sparseCoordinates0",  "sparseCoordinates8",
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
       {"delSparseTensor",     "endInsert",     "endLexInsert",
        "overwrite_csrv_csrv", "sparseDimSize", "sparseLvlSize"}) {
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

LogicalResult KokkosCppEmitter::emitVariableDeclaration(
    Location loc, Value result, bool trailingSemicolon) {
  auto op = result.getDefiningOp();
  auto type = result.getType();
  if (hasValueInScope(result)) {
    if(op) {
      return op->emitError(
          "result variable for the operation already declared");
    }
    else {
      return failure();
    }
  }
  if (auto mrType = dyn_cast<MemRefType>(type)) {
    auto space = kokkos::getMemSpace(result);
    if(failed(emitMemrefType(loc, mrType, space)))
      return failure();
  }
  else if (auto umrType = dyn_cast<UnrankedMemRefType>(type)) {
    auto space = kokkos::getMemSpace(result);
    if (failed(emitMemrefType(loc, umrType, space)))
      return failure();
  }
  else {
    if (failed(emitType(loc, result.getType())))
      return failure();
  }
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
    if (failed(emitVariableDeclaration(op.getLoc(), result, /*trailingSemicolon=*/false)))
      return failure();
    os << " = ";
    break;
  }
  default:
    for (OpResult result : op.getResults()) {
      if (failed(emitVariableDeclaration(op.getLoc(), result, /*trailingSemicolon=*/true)))
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
static LogicalResult printScalarCastOp(KokkosCppEmitter &emitter, T op) {
  if(failed(emitter.emitType(op.getLoc(), op.getOut().getType())))
    return failure();
  emitter << " " << emitter.getOrCreateName(op.getOut()) << " = ";
  emitter << "(";
  if(failed(emitter.emitType(op.getLoc(), op.getOut().getType())))
    return failure();
  emitter << ") ";
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

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    LLVM::ExtractValueOp op) {
  if(failed(emitter.emitType(op.getLoc(), op.getResult().getType())))
    return failure();
  emitter << ' ' << emitter.getOrCreateName(op.getResult()) << " = ";
  if(failed(emitter.emitValue(op.getContainer())))
    return failure();
  Type container = op.getContainer().getType();
  for(size_t i = 0; i < op.getPosition().size(); i++) {
    if (auto structType = dyn_cast<LLVM::LLVMStructType>(container)) {
      emitter << ".m" << op.getPosition()[i];
      container = structType.getBody()[op.getPosition()[i]];
    }
    else if(auto arrayType = dyn_cast<LLVM::LLVMArrayType>(container)) {
      emitter << "[" << op.getPosition()[i] << "]";
      container = arrayType.getElementType();
    }
    else {
      return op.emitError("trying to extract value from container that is neither LLVMStructType nor LLVMArrayType");
    }
  }
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    vector::PrintOp op) {
  // It's possible that op doesn't print anything
  if(!op.getSource()
      && op.getPunctuation() == vector::PrintPunctuation::NoPunctuation
      && !op.getStringLiteral()) {
    return success();
  }
  emitter << "std::cout ";
  // Print the operand (if there is one)
  if(Value val = op.getSource()) {
    emitter << "<< ";
    if(failed(emitter.emitValue(val)))
      return failure();
  }
  // Then the string literal (if there is one)
  if(auto stringLiteral = op.getStringLiteral()) {
    emitter << "<< \"";
    auto& os = emitter.ostream();
    os.write_escaped(stringLiteral.value().getValue(), true);
    emitter << "\"";
  }
  // and finally punctuation
  emitter << "<< \"";
  switch(op.getPunctuation()) {
    case vector::PrintPunctuation::NoPunctuation:
      break;
    case vector::PrintPunctuation::NewLine:
      // It appears that the upstream vector->LLVM lowering
      // skips newlines for PrintOps with a string literal,
      // so match that behavior here for consistent output
      if(!op.getStringLiteral())
        emitter << "\\n";
      break;
    case vector::PrintPunctuation::Comma:
      emitter << ",";
      break;
    case vector::PrintPunctuation::Open:
      emitter << "[";
      break;
    case vector::PrintPunctuation::Close:
      emitter << "]";
      break;
  }
  emitter << '"';
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
          // Arithmetic ops: scalar type casting that can be done easily with C-style cast
          .Case<arith::UIToFPOp, arith::FPToSIOp, arith::TruncIOp, arith::TruncFOp, arith::ExtFOp, arith::ExtSIOp, arith::ExtUIOp>(
              [&](auto op) { return printScalarCastOp(*this, op); })
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
                memref::CopyOp, memref::SubViewOp, memref::ReshapeOp,
                memref::CastOp, memref::DeallocOp, memref::DimOp,
                memref::ReinterpretCastOp, memref::ExtractStridedMetadataOp>(
              [&](auto op) { return printOperation(*this, op); })
          // EmitC ops.
          .Case<emitc::CallOp, emitc::CallOpaqueOp>(
              [&](auto op) { return printOperation(*this, op); })
          // LLVM ops.
          .Case<LLVM::ZeroOp, LLVM::ExtractValueOp>(
              [&](auto op) { return printOperation(*this, op); })
          // Vector ops.
          .Case<vector::PrintOp>(
              [&](auto op) { return printOperation(*this, op); })
          // Other operations are unknown/unsupported.
          .Default([&](Operation *) {
            return op.emitOpError("Kokkos emitter doesn't know how to output op of this type");
          });

  if (failed(status))
    return failure();
  if(!skipPrint) {
    os << (trailingSemicolon ? ";\n" : "\n");
  }
  // If op produced any DualView typed memrefs,
  // declare variables for its host and device views 
  for(auto result : op.getResults()) {
    if(auto memrefType = dyn_cast<MemRefType>(result.getType())) {
      if(kokkos::getMemSpace(result) == kokkos::MemorySpace::DualView) {
        if(skipPrint || !trailingSemicolon) {
          return op.emitOpError("op produced at least one DualView, but op was emitted in a context where we can't declare v_d and v_h views");
        }
        declareDeviceHostViews(result);
      }
    }
  }
  return success();
}

LogicalResult KokkosCppEmitter::emitInitAndFinalize(bool finalizeKokkos = true)
{
  os << "extern \"C\" void lapis_initialize()\n";
  os << "{\n";
  os.indent();
  os << "if (!Kokkos::is_initialized()) Kokkos::initialize();\n";
  //For each global view, that is not unused:
  // - allocate it
  // - if there is initializing data, copy into it. Otherwise leave uninitialized.
  for(auto& op : globalViews)
  {
    auto maybeValue = op.getInitialValue();
    MemRefType type = op.getType();
    auto space = kokkos::getMemSpace(op);
    os << "{\n";
    os.indent();
    if(maybeValue) {
      // We've already declared a global, mutable buffer with the initial data.
      // The host representation of op can be an unmanaged view of this buffer, since
      // we use LayoutRight for all views and the initial buffers are also LayoutRight.
      //
      // Only the device representation (if any) needs to be allocated now.
      if(space == kokkos::MemorySpace::Host) {
        os << op.getSymName() << " = ";
        if (failed(emitMemrefType(op.getLoc(), type, kokkos::MemorySpace::Host)))
          return failure();
        os << "(" << op.getSymName() << "_initial);\n";
      }
      else if(space == kokkos::MemorySpace::DualView) {
        if (failed(emitMemrefType(op.getLoc(), type, kokkos::MemorySpace::Host)))
          return failure();
        os << op.getSymName() << "_host(" << op.getSymName() << "_initial);\n";
        // Use the DualView constructor that takes a HostView.
        os << op.getSymName() << " = ";
        if (failed(emitMemrefType(op.getLoc(), type, kokkos::MemorySpace::DualView)))
          return failure();
        os << "(" << op.getSymName() << "_host);\n";
        // And sync it to device upfront
        os << op.getSymName() << ".sync_device();\n";
      }
      else if(space == kokkos::MemorySpace::Device) {
        // Create temporary unmanaged host view, and copy to a new device view.
        if (failed(emitMemrefType(op.getLoc(), type, kokkos::MemorySpace::Host)))
          return failure();
        os << op.getSymName() << "_host(" << op.getSymName() << "_initial);\n";
        os << op.getSymName() << " = ";
        if (failed(emitMemrefType(op.getLoc(), type, kokkos::MemorySpace::Device)))
          return failure();
        os << "(Kokkos::view_alloc(Kokkos::WithoutInitializing, \"" << op.getSymName() << "\"));\n";
        os << "Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), " << op.getSymName() << ", " << op.getSymName() << "_host);\n";
      }
    }
    else {
      // No initial data provided. Just allocate the view.
      os << op.getSymName() << " = ";
      if(space == kokkos::MemorySpace::DualView) {
        os << "(" << op.getSymName() << ")\n";
      }
      else {
        os << "(Kokkos::view_alloc(Kokkos::WithoutInitializing, \"" << op.getSymName() << "\"));\n";
      }
    }
    os.unindent();
    os << "}\n";
  }
  os.unindent();
  os << "}\n\n";
  os << "extern \"C\" void lapis_finalize()\n";
  os << "{\n";
  os.indent();
  // Free all global views
  for(auto& op : globalViews)
  {
    auto space = kokkos::getMemSpace(op);
    if(space == kokkos::MemorySpace::DualView) {
      os << op.getSymName() << ".deallocate();\n";
    }
    else {
      os << op.getSymName() << " = ";
      if(failed(emitMemrefType(op.getLoc(), op.getType(), space)))
        return failure();
      os << "();\n";
    }
  }
  // Free views returned to Python
  os << "LAPIS::alives.clear();\n";
  if(finalizeKokkos)
    os << "Kokkos::finalize();\n";
  os.unindent();
  os << "}\n";
  return success();
}

void KokkosCppEmitter::emitCppBoilerplate()
{
  // To edit the boilerplate to include in LAPIS generated programs:
  // - edit LAPISSupport.hpp (in this directory)
  // - run updateLAPISSupport.sh (turn the header into a C++ string literal)
  //
  // Note that this #include is in the emitter itself, and not in the emitted C++ file,
  // so LAPISSupportFormatted.hpp is not a dependency of emitted code.
  *this <<
    #include "LAPISSupportFormatted.hpp"
    ;
}


void KokkosCppEmitter::emitPythonBoilerplate()
{
  *py_os << "import ctypes\n";
  *py_os << "import numpy\n";
  *py_os << "from mlir import runtime as rt\n";
  *py_os << "class LAPISModule:\n";
  *py_os << "  def __init__(self, libPath):\n";
  //*py_os << "    print('Hello from LAPISModule.__init__!')\n";
  *py_os << "    self.libHandle = ctypes.CDLL(libPath)\n";
  // Do all initialization immediately
  //*py_os << "    print('Initializing module.')\n";
  *py_os << "    self.libHandle.lapis_initialize()\n";
  //*py_os << "    print('Done initializing module.')\n";
  *py_os << "  def __del__(self):\n";
  *py_os << "      self.libHandle.lapis_finalize()\n";
  //From here, only function wrappers are emitted.
  //These are class members so indent all of them now.
  py_os->indent();
}

LogicalResult KokkosCppEmitter::emitType(Location loc, Type type, bool forSparseRuntime) {
  if (auto iType = dyn_cast<IntegerType>(type)) {
    switch (iType.getWidth()) {
    case 1:
      return (*this << "bool"), success();
    case 8:
    case 16:
    case 32:
    case 64:
      if (shouldMapToUnsigned(iType.getSignedness()))
        return (*this << "uint" << iType.getWidth() << "_t"), success();
      else
        return (*this << "int" << iType.getWidth() << "_t"), success();
    default:
      return emitError(loc, "cannot emit integer type ") << type;
    }
  }
  if (auto fType = dyn_cast<FloatType>(type)) {
    switch (fType.getWidth()) {
    case 32:
      return (*this << "float"), success();
    case 64:
      return (*this << "double"), success();
    default:
      return emitError(loc, "cannot emit float type ") << type;
    }
  }
  if (auto iType = dyn_cast<IndexType>(type))
    return (*this << "size_t"), success();
  if (auto tType = dyn_cast<TensorType>(type)) {
    return emitError(loc, "cannot directly emit tensor type, should be lowered to memref");
  }
  if (auto tType = dyn_cast<TupleType>(type))
    return emitTupleType(loc, tType.getTypes());
  if (auto mrType = dyn_cast<MemRefType>(type)) {
    if (forSparseRuntime) {
      *this << "StridedMemRefType<";
      if (failed(emitType(loc, mrType.getElementType())))
        return failure();
      *this << ", " << mrType.getShape().size() << ">";
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
    *this << "void*";
    return success();
  }
  if (auto structType = dyn_cast<LLVM::LLVMStructType>(type)) {
    ensureTypeDeclared(loc, structType);
    *this << structTypes[structType];
    return success();
  }
  if (auto llvmArrayType = dyn_cast<LLVM::LLVMArrayType>(type)) {
    *this << "std::array<";
    if(failed(emitType(loc, llvmArrayType.getElementType(), false)))
      return failure();
    *this << ", " << llvmArrayType.getNumElements() << ">";
    return success();
  }
  return emitError(loc, "cannot emit type ") << type << "\n";
}

LogicalResult KokkosCppEmitter::emitTypes(Location loc, ArrayRef<Type> types, bool forSparseRuntime) {
  switch (types.size()) {
  case 0:
    *this << "void";
    return success();
  case 1:
    return emitType(loc, types.front(), forSparseRuntime);
  default:
    return emitTupleType(loc, types);
  }
}

void KokkosCppEmitter::ensureTypeDeclared(Location loc, Type t)
{
  // Recursively make sure t is declared in the C++ file.
  // If t is a:
  // - LLVMStructType -> make sure all members are declared, then declare the struct
  // - LLVMArrayType -> make sure element type declared
  // - MemRefType -> make sure element type declared
  // - Other -> primitive, nothing to do
  if(auto structType = dyn_cast<LLVM::LLVMStructType>(t)) {
    if(structTypes.contains(structType)) return;
    // Type is not registered yet - declare it
    for(auto memberType : structType.getBody())
      ensureTypeDeclared(loc, memberType);
    // Generate a unique name for the struct
    std::string name = "Struct" + std::to_string(structCount++);
    // Struct declarations go in the decl stream, which precedes the rest of the module
    selectDeclCppStream();
    *this << "struct " << name << " {\n";
    indent();
    // Define both default constructor and simple constructor taking all elements
    *this << name << "() = default;\n";
    *this << name << "(";
    int memIdx = 0;
    for(auto mem : structType.getBody()) {
      if(memIdx)
        *this << ", ";
      *this << "const ";
      (void) emitType(loc, mem, false);
      *this << "& m" << memIdx << "_";
      memIdx++;
    }
    *this << ")\n";
    *this << ": ";
    for(size_t i = 0; i < structType.getBody().size(); i++) {
      if(i)
        *this << ", ";
      *this << "m" << i << "(m" << i << "_)";
    }
    *this << " {}\n";
    memIdx = 0;
    for(auto mem : structType.getBody()) {
      (void) emitType(loc, mem, false);
      *this << " m" << memIdx << ";\n";
      memIdx++;
    }
    unindent();
    *this << "};\n";
    // Switch back to the default stream
    selectMainCppStream();
    // Register the type's name for future uses
    structTypes[structType] = name;
  }
  else if(auto arrayType = dyn_cast<LLVM::LLVMArrayType>(t)) {
    ensureTypeDeclared(loc, arrayType.getElementType());
  }
  else if (auto memrefType = dyn_cast<MemRefType>(t)) {
    ensureTypeDeclared(loc, memrefType.getElementType());
  }

  // Switch back to the main stream
  selectMainCppStream();
}

LogicalResult KokkosCppEmitter::emitFuncResultTypes(Location loc, ArrayRef<Type> types) {
  auto emitOneType =
    [&](Type type) {
      if (auto mrType = dyn_cast<MemRefType>(type)) {
        if(failed(emitMemrefType(loc, mrType, kokkos::MemorySpace::DualView)))
          return failure();
      }
      else if (auto umrType = dyn_cast<UnrankedMemRefType>(type)) {
        if(failed(emitMemrefType(loc, umrType, kokkos::MemorySpace::DualView)))
          return failure();
      }
      else {
        if(failed(emitType(loc, type, false)))
          return failure();
      }
      return success();
    };

  if(types.size() == size_t(0)) {
    *this << "void";
    return success();
  }
  if(types.size() > 1)
    *this << "std::tuple<";
  if (failed(interleaveCommaWithError(types, os, emitOneType)))
    return failure();
  if(types.size() > 1)
    *this << ">";
  return success();
}

LogicalResult KokkosCppEmitter::emitMemrefType(Location loc, MemRefType type, kokkos::MemorySpace space)
{
  if(space == kokkos::MemorySpace::DualView) {
    *this << "LAPIS::DualView<";
    if (failed(emitType(loc, type.getElementType())))
      return failure();
    for(auto extent : type.getShape()) {
      if(type.hasStaticShape()) {
          *this << '[' << extent << ']';
      }
      else {
          *this << '*';
      }
    }
    *this << ", Kokkos::LayoutRight>";
  }
  else {
    *this << "Kokkos::View<";
    if (failed(emitType(loc, type.getElementType())))
      return failure();
    for(auto extent : type.getShape()) {
      if(type.hasStaticShape()) {
          *this << '[' << extent << ']';
      }
      else {
          *this << '*';
      }
    }
    *this << ", Kokkos::LayoutRight, ";
    if(space == kokkos::MemorySpace::Device)
      *this << "Kokkos::DefaultExecutionSpace";
    else
      *this << "Kokkos::DefaultHostExecutionSpace";
    *this << ">";
  }
  return success();
}

LogicalResult KokkosCppEmitter::emitMemrefType(Location loc, UnrankedMemRefType type, kokkos::MemorySpace space)
{
  if(space == kokkos::MemorySpace::DualView) {
    *this << "LAPIS::DualView<";
    if (failed(emitType(loc, type.getElementType())))
      return failure();
    *this << "*, Kokkos::LayoutRight>";
  }
  else {
    *this << "Kokkos::View<";
    if (failed(emitType(loc, type.getElementType())))
      return failure();
    *this << "*, Kokkos::LayoutRight, ";
    if(space == kokkos::MemorySpace::Device)
      *this << "Kokkos::DefaultExecutionSpace";
    else
      *this << "Kokkos::DefaultHostExecutionSpace";
    *this << ">";
  }
  return success();
}

LogicalResult KokkosCppEmitter::emitStridedMemrefType(Location loc, MemRefType type, kokkos::MemorySpace space)
{
  if(space == kokkos::MemorySpace::DualView) {
    *this << "LAPIS::DualView<";
    if (failed(emitType(loc, type.getElementType())))
      return failure();
    for(auto extent : type.getShape()) {
      if(type.hasStaticShape()) {
          *this << '[' << extent << ']';
      }
      else {
          *this << '*';
      }
    }
    *this << ", Kokkos::LayoutStride>";
  }
  else {
    *this << "Kokkos::View<";
    if (failed(emitType(loc, type.getElementType())))
      return failure();
    for(auto extent : type.getShape()) {
      if(type.hasStaticShape()) {
          *this << '[' << extent << ']';
      }
      else {
          *this << '*';
      }
    }
    *this << ", Kokkos::LayoutStride, ";
    if(space == kokkos::MemorySpace::Device)
      *this << "Kokkos::DefaultExecutionSpace";
    else
      *this << "Kokkos::DefaultHostExecutionSpace";
    *this << ">";
  }
  return success();
}

LogicalResult KokkosCppEmitter::emitTupleType(Location loc, ArrayRef<Type> types) {
  *this << "LAPIS::tuple<";
  if (failed(interleaveCommaWithError(
          types, os, [&](Type type) { return emitType(loc, type); })))
    return failure();
  *this << ">";
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

//Version for when we are just emitting C++
LogicalResult kokkos::translateToKokkosCpp(Operation *op, raw_ostream &os) {
  //Uncomment to pause so you can attach debugger
  //pauseForDebugger();
  std::string cppDeclBuffer;
  std::string cppBuffer;
  llvm::raw_string_ostream cppDeclStream(cppDeclBuffer);
  llvm::raw_string_ostream cppStream(cppBuffer);
  KokkosCppEmitter emitter(cppDeclStream, cppStream);
  emitter.selectDeclCppStream();
  emitter.emitCppBoilerplate();
  emitter.selectMainCppStream();
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
LogicalResult kokkos::translateToKokkosCpp(Operation *op, raw_ostream &os, raw_ostream &py_os, bool isLastKernel) {
  //Uncomment to pause so you can attach debugger
  //pauseForDebugger();
  std::string cppDeclBuffer;
  std::string cppBuffer;
  llvm::raw_string_ostream cppDeclStream(cppDeclBuffer);
  llvm::raw_string_ostream cppStream(cppBuffer);
  KokkosCppEmitter emitter(cppDeclStream, cppStream, py_os);
  //Emit the C++ boilerplate to decl stream
  emitter.selectDeclCppStream();
  emitter.emitCppBoilerplate();
  emitter.selectMainCppStream();
  //Emit the ctypes boilerplate to py_os first - function wrappers need to come after this.
  emitter.emitPythonBoilerplate();
  //Global preamble.
  //Emit the actual module (global variables and functions)
  if(failed(emitter.emitOperation(*op, /*trailingSemicolon=*/false)))
    return failure();
  //Emit the init and finalize function definitions.
  if(failed(emitter.emitInitAndFinalize(isLastKernel)))
    return failure();
  os << cppDeclBuffer << cppBuffer;
  return success();
}


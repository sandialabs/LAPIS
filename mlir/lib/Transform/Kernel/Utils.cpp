#include <algorithm>
#include <csignal>
#include <iterator>
#include <stdlib.h>
#include "mlir-c/IR.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

// {{{ generic -> einsum

typedef struct EinsumArg {
  std::string spec;
  SmallVector<int> shape;
  int argIndex = -1;

  std::string stringify();

  void print(llvm::raw_fd_ostream &stream) { stream << stringify(); }
  void print() { print(llvm::errs()); }
} EinsumArg;

std::string EinsumArg::stringify() {
  std::string s = "";

  s += (spec + ": ");
  s += ("(");
  for (auto it = shape.begin(); it != shape.end(); it++) {
    s += std::to_string(*it);
    if (it != (shape.end() - 1))
      s += ", ";
  }
  s += ")";

  return s;
}

typedef struct EinsumSpecification {
  std::vector<char> reductionDims;
  std::vector<char> parallelDims;
  std::vector<char> allDims;

  std::map<unsigned int, linalg::LinalgOp> temporaries;

  std::vector<int> contractPath;

  linalg::LinalgOp definingOp;

  SmallVector<EinsumArg> inputs;
  EinsumArg output;

  std::string stringify();

  void print(llvm::raw_ostream &stream);
  void print();

  void setDimTypes();

  linalg::LinalgOp getDefiningOp() { return definingOp; }

} EinsumSpecification;

std::string EinsumSpecification::stringify() {
  std::string spec = "";
  for (auto input : inputs) {
    spec += input.spec;
    if (input.spec == (*(inputs.end() - 1)).spec)
      spec += "->";
    else
      spec += ",";
  }
  spec += output.spec;

  return spec;
}

void EinsumSpecification::print(llvm::raw_ostream &stream) {
  stream << stringify();
}

void EinsumSpecification::print() { print(llvm::errs()); }

typedef struct FusedEinsum {
  std::vector<EinsumSpecification> containedEinsums;
  std::vector<std::tuple<int>> contractPath;

  EinsumSpecification einsum;

  std::string stringify() { return einsum.stringify(); };

  void print() { print(llvm::errs()); }
  void print(llvm::raw_fd_ostream &stream) { einsum.print(stream); }

} FusedEinsum;

void printGenericAsEinsum(linalg::LinalgOp generic, llvm::raw_ostream &stream) {
  for (auto idx_map : generic.getIndexingMapsArray()) {
    stream << "(";
    for (auto res : idx_map.getResults()) {
      stream << res;
      if (res != *(idx_map.getResults().end() - 1))
        stream << ", ";
    }
    stream << ")";

    if (idx_map == *(generic.getIndexingMapsArray().end() - 2))
      stream << " -> ";
    else if (idx_map != *(generic.getIndexingMapsArray().end() - 1))
      stream << ", ";
    else
      stream << "\n";
  }
}

void printGenericAsEinsum(linalg::LinalgOp generic) {
  printGenericAsEinsum(generic, llvm::errs());
}

bool isConvertibleToEinsum(linalg::GenericOp generic) {

  // 1. must have a single result
  if (llvm::range_size(generic.getResults()) > 1) return false;

  // 2. must only contain additions or multiplications
  auto &block = generic.getRegion().front();
  for (auto &op : block) {
    if (
      !isa<arith::AddFOp>(op) && 
      !isa<arith::MulFOp>(op) &&
      !isa<linalg::YieldOp>(op)
    ) {
      op.dump();
      return false;
    }
  }

  // 3. must only contain a single addition and single multiplication
  auto addOps = block.getOps<arith::AddFOp>();
  auto mulOps = block.getOps<arith::MulFOp>();
  if (llvm::range_size(addOps) != 1) return false;
  if (llvm::range_size(mulOps) != 1) return false;

  // 4. accumulation must happen via addition in final block arg
  arith::AddFOp addition = *addOps.begin();
  auto accCheck = llvm::find(
    addition.getOperands(),
    block.getArguments().back()
  );
  if (accCheck == addition.getOperands().end()) return false;

  // 5. final block arg cannot be present in multiplication
  arith::MulFOp mul = *mulOps.begin();
  auto mulCheck = llvm::find(
    mul.getOperands(),
    block.getArguments().back()
  );
  if (mulCheck != mul.getOperands().end()) return false;

  return true;
}

SmallVector<EinsumArg> generateEinsumArgsFromGeneric(linalg::LinalgOp generic) {
  DenseMap<AffineExpr, char> affineToIndex;
  std::string chars = "zyxwvutsrqponmlkjihgfedcba";

  SmallVector<EinsumArg> args;
  for (size_t i = 0; i < generic.getIndexingMapsArray().size(); ++i) {
    EinsumArg einsum;
    auto idxMap = generic.getIndexingMapsArray()[i];

    std::string input = "";
    for (auto idx : idxMap.getResults()) {
      if (affineToIndex.find(idx) == affineToIndex.end()) {
        affineToIndex[idx] = chars.back();
        chars.pop_back();
      }

      input += affineToIndex[idx];
    }

    einsum.spec = input;

    auto genericArg = generic.getOpOperandsMatchingBBargs()[i]->get();
    if (genericArg && isa<BlockArgument>(genericArg)) {
      einsum.argIndex = dyn_cast<BlockArgument>(genericArg).getArgNumber();
    }

    args.push_back(einsum);
  }

  return args;
}

void EinsumSpecification::setDimTypes() {
  std::set<char> inputIndices;
  for (auto input : inputs) {
    for (char c : input.spec) {
      if (inputIndices.find(c) == inputIndices.end())
        inputIndices.insert(c);
    }
  }

  std::set<char> outputIndices;
  for (char c : output.spec) {
    if (outputIndices.find(c) == outputIndices.end())
      outputIndices.insert(c);
  }

  std::set_intersection(
    inputIndices.begin(), inputIndices.end(),
    outputIndices.begin(), outputIndices.end(),
    std::back_inserter(parallelDims)
  );

  std::set_difference(
    inputIndices.begin(), inputIndices.end(),
    outputIndices.begin(), outputIndices.end(),
    std::back_inserter(reductionDims)
  );

  std::set_union(
    inputIndices.begin(), inputIndices.end(),
    outputIndices.begin(), outputIndices.end(),
    std::back_inserter(allDims)
  );
}

EinsumSpecification genericToEinsumSpec(linalg::LinalgOp generic) {
  EinsumSpecification einsum;

  SmallVector<EinsumArg> args = generateEinsumArgsFromGeneric(generic);
  einsum.output = *(args.end() - 1);
  einsum.inputs.insert(einsum.inputs.begin(), args.begin(), args.end() - 1);

  // FIXME: ensure linalg.generic is expressible by an einsum
  unsigned int counter = 0;
  for (auto arg : generic.getOpOperandsMatchingBBargs()) {
    if (auto op = arg->get().getDefiningOp()) {
      if (isa<linalg::LinalgOp>(op)) {
        einsum.temporaries.insert(
          std::pair<unsigned int, linalg::LinalgOp>(counter, op));
      }
    }

    ++counter;
  }

  for (size_t i = 0; i < (generic.getDpsInputOperands().size()); ++i) {
    auto op = generic.getDpsInputOperands()[i];
    if (TensorType t = dyn_cast<TensorType>(op->get().getType())) {
      einsum.inputs[i].shape.insert(einsum.inputs[i].shape.begin(),
                                    t.getShape().begin(), t.getShape().end());
    } else {
      llvm::errs() << "Could not determine shape of inputs arguments\n";
      abort();
    }
  }

  if (!(generic.getDpsInits().size() == 1)) {
    llvm::errs() << "Only a single output operand is supported\n";
    abort();
  }

  auto op = generic.getDpsInitOperand(0);
  if (TensorType t = dyn_cast<TensorType>(op->get().getType())) {
    einsum.output.shape.insert(einsum.output.shape.begin(),
                               t.getShape().begin(), t.getShape().end());
  }

  einsum.definingOp = generic;
  einsum.setDimTypes();

  return einsum;
}

FusedEinsum fuseEinsums(std::vector<EinsumSpecification> einsums) {
  FusedEinsum fusedEinsum;

  std::vector<EinsumSpecification> fused_einsums;
  for (auto outer = einsums.rbegin(); outer != einsums.rend(); ++outer) {
    std::string availChars = "zyxwvutsrqponmlkjihgfedcba";
    for (auto input : outer->inputs) {
      for (auto idx : input.spec) {
        auto pos = availChars.find(idx);
        if (pos != std::string::npos)
          availChars.erase(pos, 1);
      }
    }

    for (auto inner = outer + 1; inner != einsums.rend(); ++inner) {
      for (auto inputToEinsum : outer->temporaries) {
        if (inputToEinsum.second == inner->getDefiningOp()) { // fusable
          EinsumArg outerInput = outer->inputs[inputToEinsum.first];

          if (outerInput.shape != inner->output.shape) {
            llvm::errs() << "ERROR: Shape mismatch\n";
            abort();
          }

          std::map<char, char> innerOutToOuterIn;
          for (size_t i = 0; i < inner->output.spec.size(); ++i)
            innerOutToOuterIn.insert(std::pair<char, char>(
              inner->output.spec[i], outerInput.spec[i]));

          std::string inputIndices = "";
          for (auto input : inner->inputs) {
            for (auto idx : input.spec) {
              auto pos = inputIndices.find(idx);
              if (pos == std::string::npos)
                inputIndices += idx;
            }
          }

          for (auto p : innerOutToOuterIn) {
            auto pos = inputIndices.find(p.first);
            if (pos != std::string::npos)
              inputIndices.erase(pos, 1);
          }

          for (auto idx : inputIndices) {
            auto p = innerOutToOuterIn.find(idx);
            if (p == innerOutToOuterIn.end()) {
              char newVal = *availChars.begin();
              availChars.erase(0, 1);
              innerOutToOuterIn.insert(std::pair<char, char>(idx, newVal));
            } else {
              char newVal = *availChars.begin();
              availChars.erase(0, 1);
              innerOutToOuterIn[p->first] = newVal;
            }
          }

          SmallVector<EinsumArg> newInputs;
          for (auto s : inner->inputs) {
            std::string innerSpec;
            for (auto idx : s.spec) {
              innerSpec += innerOutToOuterIn[idx];
            }

            EinsumArg newInput;
            newInput.spec = innerSpec;
            newInput.shape = s.shape;
            newInput.argIndex = s.argIndex;
            newInputs.push_back(newInput);
          }

          for (auto input : outer->inputs) {
            if (input.spec != outerInput.spec) {
              EinsumArg newInput;
              newInput.spec = input.spec;
              newInput.shape = input.shape;
              newInput.argIndex = input.argIndex;
              newInputs.push_back(newInput);
            }
          }

          EinsumSpecification fusedSpec;
          fusedSpec.inputs = newInputs;
          fusedSpec.output = outer->output;

          fusedEinsum.einsum = fusedSpec;
        }
      }
    }
  }

  fusedEinsum.einsum.setDimTypes();

  return fusedEinsum;
}

typedef struct EinsumSequence {
  std::vector<EinsumSpecification> sequence;
  SmallVector<SmallVector<int>> contractPath;
} EinsumSequence;

// }}}

// {{{ einsum -> generic

bool buildGenericsFromEinsums(func::FuncOp func, EinsumSequence optimalOrder) {
  if (llvm::range_size(func.getOps<linalg::GenericOp>()) == 0) return false;

  OpBuilder builder(func);
  builder.setInsertionPoint(func);

  mlir::MLIRContext *ctx = func.getContext();

  FunctionType funcType = func.getFunctionType();

  if (funcType.getNumResults() > 1) {
    llvm::errs() << "Only a single return value is supported at this time. ";
    llvm::errs() << "Generics will not be reordered.\n";
    return false;
  }

  func::FuncOp newFunc = builder.create<func::FuncOp>(
    func.getLoc(), builder.getStringAttr(func.getName().str() + "_reordered"),
    funcType);
  newFunc.addEntryBlock();

  RankedTensorType returnType =
    dyn_cast<mlir::RankedTensorType>(funcType.getResult(0));

  auto elementType = returnType.getElementType();

  SmallVector<Value> argList;
  for (BlockArgument arg : newFunc.getBody().getArguments())
  argList.push_back(Value(arg));

  size_t einsumCounter = 0;
  for (EinsumSpecification einsum : optimalOrder.sequence) {

    // get result shape and type 
    SmallVector<Type> typeVector;
    SmallVector<int64_t> outputShape;
    for (int shapeComponent : einsum.output.shape)
    outputShape.push_back((int64_t)shapeComponent);
    typeVector.push_back(mlir::RankedTensorType::get(outputShape, elementType));

    // get inputs and outputs
    SmallVector<Value> inputs;
    for (int idx : optimalOrder.contractPath[einsumCounter])
    inputs.push_back(argList[idx]);

    SmallVector<Value> outputs;
    builder.setInsertionPointToEnd(&newFunc.getBody().front());
    outputs.push_back(builder.create<tensor::EmptyOp>(
      newFunc.getLoc(), outputShape, elementType));

    // get indexing maps
    std::map<char, AffineExpr> allAffineDims;
    for (auto [idx, dim] : llvm::enumerate(einsum.allDims)) {
      allAffineDims.insert(
        std::pair<char, AffineExpr>(dim, mlir::getAffineDimExpr(idx, ctx)));
    }

    SmallVector<AffineMap> indexingMaps;
    for (EinsumArg input : einsum.inputs) {
      std::vector<AffineExpr> indexingMapOutputDims;
      for (char dim : input.spec)
      indexingMapOutputDims.push_back(allAffineDims[dim]);

      indexingMaps.push_back(
        AffineMap::get(einsum.allDims.size(), 0, indexingMapOutputDims, ctx));
    }

    std::vector<AffineExpr> indexingMapOutputDims;
    for (char dim : einsum.output.spec)
    indexingMapOutputDims.push_back(allAffineDims[dim]);
    indexingMaps.push_back(
      AffineMap::get(einsum.allDims.size(), 0, indexingMapOutputDims, ctx));

    // get iterator types
    SmallVector<utils::IteratorType> iteratorTypes;
    for (auto [dim, affineDim] : allAffineDims) {
      if (std::find(einsum.parallelDims.begin(), einsum.parallelDims.end(),
                    dim) != einsum.parallelDims.end())
        iteratorTypes.push_back(utils::IteratorType::parallel);
      else if (std::find(einsum.reductionDims.begin(),
                         einsum.reductionDims.end(),
                         dim) != einsum.reductionDims.end())
        iteratorTypes.push_back(utils::IteratorType::reduction);
    }

    SmallVector<NamedAttribute> attributes = {};

    linalg::GenericOp generic = builder.create<linalg::GenericOp>(
      newFunc.getLoc(),
      TypeRange(typeVector),
      ValueRange(inputs),
      ValueRange(outputs),
      indexingMaps,
      ArrayRef(iteratorTypes),
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        Value mul = nestedBuilder.create<arith::MulFOp>(
          nestedLoc, args[0], args[1]);
        Value add = nestedBuilder.create<arith::AddFOp>(
          nestedLoc, args[2], mul);
        nestedBuilder.create<linalg::YieldOp>(nestedLoc, add);
      },
      attributes
    );

    Value result = generic.getResult(0);
    int resultStoreIndex =
      *std::min_element(optimalOrder.contractPath[einsumCounter].begin(),
                        optimalOrder.contractPath[einsumCounter].end());

    for (int argIdx : optimalOrder.contractPath[einsumCounter]) {
      argList.erase(std::find(argList.begin(), argList.end(), argList[argIdx])); 
    }

    argList.insert(argList.begin() + resultStoreIndex, result);

    if (einsumCounter == (optimalOrder.sequence.size() - 1)) {
      builder.setInsertionPointToEnd(&newFunc.getBody().front());
      builder.create<func::ReturnOp>(newFunc.getLoc(), result);
    }

    ++einsumCounter;
  }

  // replace call to old function with call to new, optimized function
  ModuleOp module = dyn_cast<ModuleOp>(func->getParentOp());
  module.walk([&](Operation *op) {
    if (auto call = dyn_cast<func::CallOp>(op)) {
      if (call.getCallee() == func.getName()) {
        call.setCallee(newFunc.getName());
      }
    }
  });

  return true;
}

// }}}

// {{{ optimizers

typedef struct BruteForceOptimizer {
  FusedEinsum unoptimizedEinsum;
  EinsumSequence optimizedEinsumSequence;

  BruteForceOptimizer(FusedEinsum originalEinsum)
  : unoptimizedEinsum(originalEinsum) {}

  void optimize();

} BruteForceOptimizer;

std::vector<char> getSharedIndices(EinsumArg iarg, EinsumArg jarg) {
  std::set<char> iindices(iarg.spec.begin(), iarg.spec.end());
  std::set<char> jindices(jarg.spec.begin(), jarg.spec.end());

  std::vector<char> sharedIndices;
  std::set_intersection(
    iindices.begin(), iindices.end(), 
    jindices.begin(), jindices.end(), 
    std::back_inserter(sharedIndices)
  );

  return sharedIndices;
}

double estimateCost(EinsumArg iarg, EinsumArg jarg,
                    std::vector<char> sharedIndices) {
  double cost = 1.0;
  for (char sharedIdx : sharedIndices)
    cost *= iarg.shape[iarg.spec.find(sharedIdx)];

  std::set<char> allIndices(iarg.spec.begin(), iarg.spec.end());
  allIndices.insert(jarg.spec.begin(), jarg.spec.end());

  std::vector<char> disjointIndices;
  std::set_difference(
    allIndices.begin(), allIndices.end(), 
    sharedIndices.begin(), sharedIndices.end(),
    std::back_inserter(disjointIndices)
  );

  for (char idx : disjointIndices) {
    if (iarg.spec.find(idx) != std::string::npos)
      cost *= iarg.shape[iarg.spec.find(idx)];
    else if (jarg.spec.find(idx) != std::string::npos)
      cost *= jarg.shape[jarg.spec.find(idx)];
  }

  return cost;
}

SmallVector<int> getResultShape(EinsumArg iarg, EinsumArg jarg,
                                std::string resultIndices) {
  SmallVector<int> shape;
  for (char resultIdx : resultIndices) {
    auto ipos = iarg.spec.find(resultIdx);
    if (ipos != std::string::npos) {
      shape.push_back(iarg.shape[ipos]);
      continue; // avoid double-counting parallel axes
    }

    auto jpos = iarg.spec.find(resultIdx);
    if (jpos != std::string::npos)
      shape.push_back(iarg.shape[ipos]);
  }

  return shape;
}

std::string getResultIndices(EinsumArg iarg, EinsumArg jarg,
                             std::vector<char> sharedIndices,
                             std::vector<char> reductionDims) {

  std::vector<char> sharedReductionIndices;
  std::set_intersection(reductionDims.begin(), reductionDims.end(),
                        sharedIndices.begin(), sharedIndices.end(),
                        std::back_inserter(sharedReductionIndices));

  std::set<char> allIndices;
  allIndices.insert(iarg.spec.begin(), iarg.spec.end());
  allIndices.insert(jarg.spec.begin(), jarg.spec.end());
  for (char sharedReductionIndex : sharedReductionIndices) {
    if (allIndices.find(sharedReductionIndex) != allIndices.end())
      allIndices.erase(sharedReductionIndex);
  }

  std::string spec(allIndices.begin(), allIndices.end());
  return spec;
}

bool checkLegalContraction(EinsumArg iarg, EinsumArg jarg,
                           std::vector<char> sharedIndices,
                           std::vector<char> reductionDims) {

  std::vector<char> reducedIndices;
  std::set_intersection(
    sharedIndices.begin(), sharedIndices.end(),
    reductionDims.begin(), reductionDims.end(),
    std::back_inserter(reducedIndices)
  );

  if (reducedIndices.size() == 0) return false;

  for (char idx : reducedIndices) {
    auto ipos = iarg.spec.find(idx);
    auto jpos = jarg.spec.find(idx);

    if ((ipos != std::string::npos) && (jpos != std::string::npos)) {
      if (iarg.shape[ipos] != jarg.shape[jpos]) return false;
    }
    if ((ipos != std::string::npos && jpos == std::string::npos)) return false;
    if ((ipos == std::string::npos && jpos != std::string::npos)) return false;
  }

  return true;
}

void BruteForceOptimizer::optimize() {
  EinsumSpecification einsum = unoptimizedEinsum.einsum;
  SmallVector<EinsumArg> inputs = einsum.inputs;
  EinsumArg output = einsum.output;

  while (inputs.size() > 1) {
    double bestCost = std::numeric_limits<double>::max();
    EinsumArg smallestTemporary;

    int imin = -1;
    int jmin = -1;
    for (size_t i = 0; i < inputs.size(); ++i) {
      for (size_t j = i + 1; j < inputs.size(); ++j) {
        EinsumArg iarg = inputs[i];
        EinsumArg jarg = inputs[j];

        std::vector<char> sharedIndices = getSharedIndices(iarg, jarg);

        if (!checkLegalContraction(iarg, jarg, sharedIndices,
                                   einsum.reductionDims))
          continue;

        std::string resultIndices =
          getResultIndices(iarg, jarg, sharedIndices, einsum.reductionDims);

        SmallVector<int> resultShape =
          getResultShape(iarg, jarg, resultIndices);

        double estimatedCost = estimateCost(iarg, jarg, sharedIndices);
        if (estimatedCost <= bestCost) {
          bestCost = estimatedCost;
          imin = i;
          jmin = j;

          smallestTemporary.spec = resultIndices;
          smallestTemporary.shape = resultShape;
        }
      }
    }

    EinsumSpecification einsumPart;
    einsumPart.inputs.push_back(inputs[imin]);
    einsumPart.inputs.push_back(inputs[jmin]);

    SmallVector<int> path;
    for (EinsumArg input : einsumPart.inputs) {
      if (input.argIndex != -1)
        path.push_back(input.argIndex);
    }

    optimizedEinsumSequence.contractPath.push_back(path);
    smallestTemporary.argIndex = *std::min_element(path.begin(), path.end());

    inputs.erase(inputs.begin() + imin);
    inputs.erase(inputs.begin() + jmin);
    inputs.push_back(smallestTemporary);

    einsumPart.output = smallestTemporary;
    einsumPart.setDimTypes();
    optimizedEinsumSequence.sequence.push_back(einsumPart);
  }
}

// }}}

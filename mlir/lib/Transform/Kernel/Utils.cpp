#include <csignal>
#include <stdlib.h>
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Linalg -> einsum helpers
//===----------------------------------------------------------------------===//

// {{{ generic -> einsum

typedef struct EinsumArg {
  std::string spec;
  SmallVector<int> shape;

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
  SmallVector<unsigned int> reductionDims;
  SmallVector<unsigned int> parallelDims;
  std::map<unsigned int, linalg::LinalgOp> argsProducedByEinsum;

  linalg::LinalgOp definingOp;

  SmallVector<EinsumArg> inputs;
  EinsumArg output;

  std::string stringify();

  void print(llvm::raw_ostream &stream);
  void print();

  void dumpToFile(std::string filename);
  void dumpToFile();

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

void EinsumSpecification::dumpToFile(std::string filename) {
  std::error_code EC;
  llvm::raw_fd_ostream fstream(filename, EC,
                               llvm::sys::fs::OpenFlags::OF_Delete);
  fstream << this->stringify() << "\n";
  for (auto arg : inputs)
    fstream << arg.stringify() << "\n";
  fstream << output.stringify() << "\n";
}

void EinsumSpecification::dumpToFile() { dumpToFile("./unoptimized.einsum"); }

typedef struct FusedEinsum {
  std::vector<EinsumSpecification> containedEinsums;
  std::vector<std::tuple<int>> contractPath;

  EinsumSpecification einsum;

  std::string stringify() { return einsum.stringify(); };

  void print() { print(llvm::errs()); }
  void print(llvm::raw_fd_ostream &stream) { einsum.print(stream); }

  void dumpToFile() { dumpToFile("unoptimizedFused.einsum"); }
  void dumpToFile(std::string filename) { einsum.dumpToFile(filename); }
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

SmallVector<EinsumArg> generateEinsumArgsFromGeneric(linalg::LinalgOp generic) {
  DenseMap<AffineExpr, char> affineToIndex;
  std::string chars = "zyxwvutsrqponmlkjihgfedcba";

  SmallVector<EinsumArg> args;
  for (auto idxMap : generic.getIndexingMapsArray()) {
    std::string input = "";
    for (auto idx : idxMap.getResults()) {
      if (affineToIndex.find(idx) == affineToIndex.end()) {
        affineToIndex[idx] = chars.back();
        chars.pop_back();
      }

      input += affineToIndex[idx];
    }
    EinsumArg einsum;
    einsum.spec = input;
    args.push_back(einsum);
  }

  return args;
}

EinsumSpecification genericToEinsumSpec(linalg::LinalgOp generic) {
  EinsumSpecification einsum;
  generic.getParallelDims(einsum.parallelDims);
  generic.getReductionDims(einsum.reductionDims);

  SmallVector<EinsumArg> args = generateEinsumArgsFromGeneric(generic);
  einsum.output = *(args.end() - 1);
  einsum.inputs.insert(einsum.inputs.begin(), args.begin(), args.end() - 1);

  // determine whether args were produced by another einsum
  // TODO: verify that all linalgops can be converted to generics
  // (fairly certain this is true)
  unsigned int counter = 0;
  for (auto arg : generic.getOpOperandsMatchingBBargs()) {
    if (auto op = arg->get().getDefiningOp())
      if (isa<linalg::LinalgOp>(op))
        einsum.argsProducedByEinsum.insert(
            std::pair<unsigned int, linalg::LinalgOp>(counter, op));

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

  return einsum;
}

FusedEinsum fuseEinsums(std::vector<EinsumSpecification> einsums) {
  /*----------------------------------------------------------===//
   * We assume a consumer einsum can be written as a composition
   * of producer einsums. For example, assume we are performing
   * ABx. Let f(A, B) = AB and g(C, x) = Cx. Writing as einsums
   *
   *   f(A, B) = "ik,kj->ij"
   *   g(f(A, B), x) = "ij,j->i"
   *                 = "ik,kj,j->i"
   *
   * Hence, by "remembering" which argument positions are einsums,
   * we can validate that shapes match and *assume* that the axes
   * are correctly ordered in the producer einsum.
   *
   * NOTE: the current implementation will fuse einsums regardless
   * of the contained operations in the original linalg.generic.
   * This can be restricted, but  must be done so elsewhere and
   * is still WIP.
   *
  //===---------------------------------------------------------*/
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
      for (auto inputToEinsum : outer->argsProducedByEinsum) {
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
            newInputs.push_back(newInput);
          }

          for (auto input : outer->inputs) {
            if (input.spec != outerInput.spec) {
              EinsumArg newInput;
              newInput.spec = input.spec;
              newInput.shape = input.shape;
              newInputs.push_back(newInput);
            }
          }

          EinsumSpecification fusedSpec;
          fusedSpec.inputs = newInputs;
          fusedSpec.output = outer->output;

          fusedSpec.dumpToFile();

          fusedEinsum.einsum = fusedSpec;
        }
      }
    }
  }

  return fusedEinsum;
}

// }}}

// {{{ optimizers

typedef struct PythonOptimizer {
  FusedEinsum unoptimizedEinsum;
  std::string optimizer;
  std::string input_filename;
  std::string output_filename;

  PythonOptimizer(FusedEinsum einsum, std::string opt, std::string in_file,
                  std::string out_file)
      : unoptimizedEinsum(einsum), optimizer(opt), input_filename(in_file),
        output_filename(out_file) {}

  PythonOptimizer(FusedEinsum einsum, std::string opt, std::string in_file)
      : unoptimizedEinsum(einsum), optimizer(opt), input_filename(in_file),
        output_filename("optimized.einsum") {}

  PythonOptimizer(FusedEinsum einsum)
      : unoptimizedEinsum(einsum), optimizer("opt_einsum"),
        input_filename("unoptimized.einsum"),
        output_filename("optimized.einsum") {}

  void optimize();

} PythonOptimizer;

void PythonOptimizer::optimize() {
  std::string cmd = "";
  std::string lapis_src = getenv("LAPIS_SRC");
  cmd += "python3 " + lapis_src + "/mlir/lib/Transform/Kernel/einsums.py ";
  cmd += "-i " + input_filename + " ";
  cmd += "-o " + output_filename + " ";
  cmd += "-f ";
  cmd += (optimizer == "cotengra") ? "-c" : "";

  int ret = system(cmd.c_str());
  if (ret) {
    llvm::errs() << "ERROR: Python script was unable to execute. Return code: ";
    llvm::errs() << ret << "\n";
  }
}

// }}}

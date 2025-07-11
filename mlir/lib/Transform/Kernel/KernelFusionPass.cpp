#include <map>
#include <set>
#include <deque>

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Casting.h"

namespace mlir {
namespace kernel {

using AllocTensorOp = bufferization::AllocTensorOp;
using EmptyOp = tensor::EmptyOp;
using FuncOp = func::FuncOp;
using CallOp = func::CallOp;
using ReturnOp = func::ReturnOp;
using LinalgOp = linalg::LinalgOp;
using FuncVec = std::vector<CallOp>;
using CallMap = std::map<CallOp, FuncVec>;

using FusionSet = std::vector<CallOp>;
using FusionSetVector = std::vector<FusionSet>;
using ArgsToIdxTy = std::vector<std::pair<Value, int>>;

CallMap getCallMap(FuncOp mainFuncOp) {
  SmallVector<CallOp> calls;

  mainFuncOp.walk([&calls](func::CallOp call) {
    calls.push_back(call);
  });

  CallMap callMap;
  for (auto keyCall : calls) {
    callMap[keyCall] = FuncVec();
    for (auto valueCall : calls) {
      if (keyCall == valueCall)
        continue;

      bool valueCallMapped = false;

      for (auto keyOp : keyCall.getOperands()) {
        if (valueCallMapped)
          break;
        for (auto valueOp : valueCall.getOperands()) {
          if (keyOp == valueOp) {
            callMap[keyCall].push_back(valueCall);
            valueCallMapped = true;
            break;
          }
        }
      }

      for (auto keyResult : keyCall.getResults()) {
        if (valueCallMapped)
          break;
        for (auto valueOp : valueCall.getOperands()) {
          if (keyResult == valueOp) {
            callMap[keyCall].push_back(valueCall);
            valueCallMapped = true;
            break;
          }
        }
      }
    }
  }

  return callMap;
}

bool parallelIterationSpacesMatch(ModuleOp module, CallOp firstCall,
                                  CallOp secondCall) {
  FuncOp firstCallee = dyn_cast<FuncOp>(SymbolTable::lookupNearestSymbolFrom(
      module, firstCall.getCallableForCallee().get<SymbolRefAttr>()));
  FuncOp secondCallee = dyn_cast<FuncOp>(SymbolTable::lookupNearestSymbolFrom(
      module, secondCall.getCallableForCallee().get<SymbolRefAttr>()));

  IRMapping firstCalleeToCall;
  IRMapping secondCalleeToCall;
  Block &firstBody = firstCallee.getFunctionBody().front();
  Block &secondBody = secondCallee.getFunctionBody().front();
  firstCalleeToCall.map(firstCallee.getArguments(), firstCall.getOperands());
  secondCalleeToCall.map(secondCallee.getArguments(), secondCall.getOperands());

  SmallVector<LinalgOp> firstLAOps(firstBody.getOps<LinalgOp>());
  SmallVector<LinalgOp> secondLAOps(secondBody.getOps<LinalgOp>());

  for (LinalgOp firstLAOp : firstLAOps) {
    SmallVector<unsigned int> firstParDims;
    firstLAOp.getParallelDims(firstParDims);
    for (LinalgOp secondLAOp : secondLAOps) {
      SmallVector<unsigned int> secondParDims;
      secondLAOp.getParallelDims(secondParDims);
      
      bool firstIncludesSecond = std::includes(
        firstParDims.begin(), firstParDims.end(),
        secondParDims.begin(), secondParDims.end()
      );

      if (!firstIncludesSecond) return false;
    }
  }

  return true;
}

bool markedForFusion(CallOp keyKernel, CallOp valKernel) {
  if (!keyKernel->getAttr("fuse_with") || !valKernel->getAttr("fuse_with"))
    return false;

  SmallVector<StringRef> keyKernelFuseWithStrings;
  SmallVector<StringRef> valKernelFuseWithStrings;
  StringRef keyKernelFuseWithString =
      dyn_cast<StringAttr>(keyKernel->getAttr("fuse_with")).strref();
  StringRef valKernelFuseWithString =
      dyn_cast<StringAttr>(valKernel->getAttr("fuse_with")).strref();
  keyKernelFuseWithString.split(keyKernelFuseWithStrings, ",");
  valKernelFuseWithString.split(valKernelFuseWithStrings, ",");

  bool fuseWithFlag1 = false;
  for (StringRef fuseWithString : keyKernelFuseWithStrings) {
    if (valKernel.getCallee() == fuseWithString.trim(' ')) {
      fuseWithFlag1 = true;
      break;
    }
  }
  bool fuseWithFlag2 = false;
  for (StringRef fuseWithString : valKernelFuseWithStrings) {
    if (keyKernel.getCallee() == fuseWithString.trim(' ')) {
      fuseWithFlag2 = true;
      break;
    }
  }

  return fuseWithFlag1 && fuseWithFlag2;
}

FusionSetVector createFusionSets(mlir::ModuleOp module, FuncOp mainFuncOp,
                                 CallMap callMap) {
  std::deque<CallOp> kernelsToFuse;
  std::set<CallOp> kernelSet;
  FusionSetVector fusionSets;

  mainFuncOp.walk([&kernelsToFuse, &kernelSet](func::CallOp call) {
    if (kernelSet.find(call) == kernelSet.end()) {
      kernelSet.insert(call);
      kernelsToFuse.push_back(call);
    }
  });

  int fusionSetIndex = 0;
  while (!kernelsToFuse.empty()) {
    auto kernelToFuse = kernelsToFuse.front();
    kernelsToFuse.pop_front();

    auto kernelCheck = kernelSet.find(kernelToFuse);
    if (kernelCheck == kernelSet.end())
      continue;

    kernelSet.extract(kernelToFuse);

    if (fusionSetIndex != int(fusionSets.size() - 1))
      fusionSets.push_back(std::vector<CallOp>());
    fusionSets[fusionSetIndex].push_back(kernelToFuse);

    bool fusionLegal = false;
    for (auto val : callMap[kernelToFuse]) {
      fusionLegal = parallelIterationSpacesMatch(module, kernelToFuse, val);
      if (fusionLegal && markedForFusion(kernelToFuse, val)) {
        kernelSet.extract(val);
        fusionSets[fusionSetIndex].push_back(val);
        fusionLegal = false; // reset flag
      }
    }

    if (fusionSets[fusionSetIndex].size() == 1) {
      fusionSets[fusionSetIndex].pop_back();
      continue;
    }

    fusionSetIndex += 1;
  }
  return fusionSets;
}

void buildArgsToIndexMap(FusionSet &fusionSet, CallOp kernel,
                         SmallVector<Value> &newArgs,
                         SmallVector<Value> &newResults,
                         DenseMap<Value, int> &argsToIndexMap,
                         int &fusedKernelArgIndex) {
  for (auto arg : kernel.getOperands()) {
    auto argCheck = std::find(newArgs.begin(), newArgs.end(), arg);
    if (argCheck != newArgs.end())
      continue;

    auto producer = arg.getDefiningOp();
    if (producer) {
      auto producerCheck =
          std::find(fusionSet.begin(), fusionSet.end(), producer);
      if (producerCheck != fusionSet.end())
        continue;
    }

    newArgs.push_back(arg);
    argsToIndexMap[arg] = fusedKernelArgIndex;
    fusedKernelArgIndex++;
  }
}

void buildResultsToIndexMap(FusionSet &fusionSet, CallOp kernel,
                            SmallVector<Value> &newResults,
                            DenseMap<Value, int> &resultsToIndexMap,
                            int &fusedKernelResultIndex) {
  for (auto res : kernel.getResults()) {
    // TODO: mark subkernels that have results that are used outside of the
    // context of the fused kernel (used to determine whether we need to return
    // intermediate results or not)

    // auto resCheck = std::find(newResults.begin(), newResults.end(), res);
    // if (resCheck != newResults.end())
    //   continue;
    //
    // // check if the result is used by someone else in the fusion set
    // bool userInFusionSet = false;
    // for (auto user : res.getUsers()) {
    //   auto userCheck =
    //       std::find(fusionSet.begin(), fusionSet.end(), user);
    //   if (userCheck != fusionSet.end()) {
    //     userInFusionSet = true;
    //     break;
    //   }
    // }
    //
    // if (userInFusionSet)
    //   continue;

    newResults.push_back(res);
    resultsToIndexMap[res] = fusedKernelResultIndex;
    fusedKernelResultIndex++;
  }
}

FuncOp buildFusedKernelOp(OpBuilder &builder, ModuleOp &module,
                          FusionSet &fusionSet, SmallVector<Value> &newArgs,
                          SmallVector<Value> &newResults,
                          int fusedKernelCounter) {
  TypeRange newArgTypes(newArgs);
  TypeRange newResultTypes(newResults);
  FunctionType fusedKernelType =
      builder.getFunctionType(newArgTypes, newResultTypes);

  builder.setInsertionPointToStart(module.getBody());

  std::string fusedKernelName = "";
  for (auto kernel : fusionSet)
    fusedKernelName += (kernel.getCallee() + "_").str();
  fusedKernelName += std::to_string(fusedKernelCounter);

  FuncOp fusedKernelOp = builder.create<FuncOp>(
      module.getLoc(), fusedKernelName, fusedKernelType);

  fusedKernelOp.addEntryBlock();

  return fusedKernelOp;
}

void insertSubkernelCallsIntoFusedKernel(OpBuilder &builder, ModuleOp module,
                                         FusionSet &fusionSet,
                                         FuncOp fusedKernelOp,
                                         DenseMap<Value, int> &argsToIndexMap) {
  builder.setInsertionPointToStart(&fusedKernelOp.front());

  DenseMap<CallOp, ArgsToIdxTy> callsToIntermediateValues;
  for (CallOp kernel : fusionSet)
    callsToIntermediateValues[kernel] = ArgsToIdxTy();

  for (CallOp kernel : fusionSet) {
    FuncOp callee = dyn_cast<FuncOp>(SymbolTable::lookupNearestSymbolFrom(
        module, kernel.getCallableForCallee().get<SymbolRefAttr>()));

    if (!callee.isPrivate())
      callee.setPrivate();

    SmallVector<Value> args;
    for (auto arg : kernel.getOperands())
      args.push_back(fusedKernelOp.getArgument(argsToIndexMap[arg]));

    auto newCallHandle =
        builder.create<CallOp>(fusedKernelOp.getLoc(), callee, args);

    for (auto result : kernel.getResults()) {
      for (auto user : result.getUsers()) {
        auto userCheck = std::find(fusionSet.begin(), fusionSet.end(), user);
        if (userCheck != fusionSet.end()) {
          int argIndex = 0;
          CallOp userCall = *userCheck;
          for (auto arg : userCall.getOperands()) {
            if (arg == result) {
              auto newArg = newCallHandle.getResult(argIndex);
              callsToIntermediateValues[userCall].push_back(
                  std::make_pair(newArg, argIndex));
            }
            argIndex++;
          }
        }
      }
    }

    if (!callsToIntermediateValues[kernel].empty())
      for (auto argIndexPair : callsToIntermediateValues[kernel])
        newCallHandle.setOperand(argIndexPair.second, argIndexPair.first);

    newCallHandle.getOperation()->setAttr("inline",
                                          builder.getStringAttr("true"));
  }
}

void insertReturnOpToFusedKernelOp(OpBuilder &builder, FuncOp fusedKernelOp) {
  SmallVector<Value> returnOperands;
  for (auto newCall : fusedKernelOp.getOps<CallOp>())
    for (auto result : newCall.getResults())
      returnOperands.push_back(result);

  builder.create<ReturnOp>(fusedKernelOp.getLoc(),
                                 ValueRange(returnOperands));
}

void buildFusedKernelCallAndReplaceSubkernelUses(
    OpBuilder &builder, FusionSet &fusionSet, FuncOp mainFuncOp,
    FuncOp fusedKernelOp, SmallVector<Value> newArgs,
    DenseMap<Value, int> resultsToIndexMap) {

  builder.setInsertionPoint(*fusionSet.rbegin());
  CallOp fusedKernelCallHandle =
      builder.create<CallOp>(mainFuncOp.getLoc(), fusedKernelOp, newArgs);
  fusedKernelCallHandle->setAttr("noinline", builder.getUnitAttr());

  for (auto kernelCall : fusionSet) {
    for (auto result : kernelCall.getResults()) {
      auto newResult =
          fusedKernelCallHandle.getResult(resultsToIndexMap[result]);
      result.replaceAllUsesWith(newResult);
      kernelCall.erase();
    }
  } // end loop over kernels in fusionSet
}

#define GEN_PASS_DEF_KERNELFUSIONPASS
#include "Transform/Kernel/KernelPasses.h.inc"

struct KernelFusionPass : impl::KernelFusionPassBase<KernelFusionPass> {
  using KernelFusionPassBase::KernelFusionPassBase;

  void runOnOperation() override {

    // get the module and main function
    OpBuilder builder(getOperation()->getContext());
    ModuleOp module = dyn_cast<ModuleOp>(getOperation());
    FuncOp mainFuncOp;
    bool foundMainFunc = false;
    for (FuncOp funcOp : module.getOps<FuncOp>()) {
      if (funcOp.getSymName() == "main") {
        mainFuncOp = funcOp;
        foundMainFunc = true;
        break;
      }
    }

    if (!foundMainFunc)
      return;

    CallMap callMap = getCallMap(mainFuncOp);
    FusionSetVector fusionSets = createFusionSets(module, mainFuncOp, callMap);

    int fusedKernelCounter = 0;
    for (auto fusionSet : fusionSets) {
      if (fusionSet.empty())
        continue;

      SmallVector<Value> newArgs;
      SmallVector<Value> newResults;
      SmallVector<Value> intermediates;
      DenseMap<Value, int> argsToIndexMap;
      DenseMap<Value, int> resultsToIndexMap;
      int fusedKernelArgIndex = 0;
      int fusedKernelResultIndex = 0;
      for (auto kernel : fusionSet) {
        buildArgsToIndexMap(fusionSet, kernel, newArgs, newResults,
                            argsToIndexMap, fusedKernelArgIndex);

        buildResultsToIndexMap(fusionSet, kernel, newResults,
                               resultsToIndexMap, fusedKernelResultIndex);
      }

      FuncOp fusedKernelOp = buildFusedKernelOp(
          builder, module, fusionSet, newArgs, newResults, fusedKernelCounter);

      insertSubkernelCallsIntoFusedKernel(builder, module, fusionSet,
                                          fusedKernelOp, argsToIndexMap);

      insertReturnOpToFusedKernelOp(builder, fusedKernelOp);

      buildFusedKernelCallAndReplaceSubkernelUses(builder, fusionSet,
                                                  mainFuncOp, fusedKernelOp,
                                                  newArgs, resultsToIndexMap);

      fusedKernelCounter += 1;
    } // end loop over fusionSets
  } // end runOnOperation
};
} // namespace kernel
} // namespace mlir

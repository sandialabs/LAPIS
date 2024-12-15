/* TODO: check for 'fuse_with' attribute
 * TODO: verify parallel iteration spaces match
 *  - KernelDomainFusion expects this to be done; legality is checked in KDF
 */

#include <map>
#include <set>
#include <deque>

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
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
using FusionSetVector = std::vector<std::vector<CallOp>>;

CallMap getCallMap(FuncOp mainFuncOp) {
  auto calls = mainFuncOp.getOps<CallOp>();
  CallMap callMap;
  for (auto keyCall : calls) {
    callMap[keyCall] = FuncVec();
    for (auto valueCall : calls) {
      if (keyCall == valueCall)
        continue;

      // mapped flag
      bool valueCallMapped = false;

      // check for shared arguments
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

      // check for read-write dependence
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

FailureOr<OperandRange> inferShape(Value tensor) {
  if (Operation *op = tensor.getDefiningOp()) {
    if (EmptyOp emptyOp = dyn_cast<EmptyOp>(op))
      return emptyOp.getOperands();
    if (AllocTensorOp allocTensorOp = dyn_cast<AllocTensorOp>(op))
      return allocTensorOp.getOperands();
  }
  return failure();
}

// NOTE: we are fusing kernel calls using kernel definitions, use operand/result
// definitions in the call to map them to the operations in the kernel
bool parallelIterationSpacesMatch(mlir::ModuleOp module, CallOp firstCall,
                                  CallOp secondCall) {
  // get the FuncOps
  FuncOp firstCallee = dyn_cast<FuncOp>(SymbolTable::lookupNearestSymbolFrom(
      module, firstCall.getCallableForCallee().get<SymbolRefAttr>()));
  FuncOp secondCallee = dyn_cast<FuncOp>(SymbolTable::lookupNearestSymbolFrom(
      module, secondCall.getCallableForCallee().get<SymbolRefAttr>()));

  // map FuncOp args -> CallOp operands
  IRMapping firstCalleeToCall;
  IRMapping secondCalleeToCall;
  Block &firstBody = firstCallee.getFunctionBody().front();
  Block &secondBody = secondCallee.getFunctionBody().front();
  firstCalleeToCall.map(firstCallee.getArguments(), firstCall.getOperands());
  secondCalleeToCall.map(secondCallee.getArguments(), secondCall.getOperands());

  // check args of each LinalgOp; match => check iteration domains
  SmallVector<LinalgOp> firstLAOps(firstBody.getOps<LinalgOp>());
  SmallVector<LinalgOp> secondLAOps(secondBody.getOps<LinalgOp>());

  // no built-in support for shape analysis => do some digging
  for (LinalgOp firstLAOp : firstLAOps) {
    SmallVector<OpOperand *> fOperands =
        firstLAOp.getOpOperandsMatchingBBargs();

    for (LinalgOp secondLAOp : secondLAOps) {
      SmallVector<OpOperand *> sOperands =
          secondLAOp.getOpOperandsMatchingBBargs();

      // if we can't figure out loop bounds, look at axes of matching operands
      for (OpOperand *fOp : fOperands) {
        Value fArg = firstCalleeToCall.lookupOrNull(fOp->get());
        for (OpOperand *sOp : sOperands) {
          Value sArg = secondCalleeToCall.lookupOrNull(sOp->get());

          // check that axes match 
          if (fArg == sArg && isa<TensorType>(fArg.getType())) {
          }
        }
      }
    }
  }

  return true;
}

bool markedForFusion(CallOp keyKernel, CallOp valKernel) {
  // FIXME: segfault if one of the kernels does not have this attribute 
  if (!keyKernel->hasAttr("fuse_with") ||
      !valKernel->hasAttr("fuse_with")) {
    return false;
  }

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
  std::set<CallOp> kernelSet; // full set of unique kernels
  FusionSetVector fusionSets; // sets of kernels to be fused
  for (auto call : mainFuncOp.getOps<CallOp>()) {
    // attempt to find the call in the kernelSet
    if (kernelSet.find(call) != kernelSet.end()) {
      continue;
    }
    kernelSet.insert(call);
    kernelsToFuse.push_back(call); // preserve call order within blocks
  }

  // create sets of fusions
  int fusionSetIndex = 0;
  while (!kernelsToFuse.empty()) {
    auto kernelToFuse = kernelsToFuse.front();
    kernelsToFuse.pop_front();

    // if the kernel has already been fused, then don't fuse again
    auto kernelCheck = kernelSet.find(kernelToFuse);
    if (kernelCheck == kernelSet.end()) {
      continue;
    }

    // add to the current fusion set, remove from kernelSet
    kernelSet.extract(kernelToFuse);

    // we may have not found any kernels to fuse on the last iteration
    if (fusionSetIndex != int(fusionSets.size() - 1))
      fusionSets.push_back(std::vector<CallOp>());
    fusionSets[fusionSetIndex].push_back(kernelToFuse);

    // process edges in the graph
    bool fusionLegal = true; // FIXME: this is not properly checked right now
    for (auto val : callMap[kernelToFuse]) {
      // fusion legal -> update everything
      fusionLegal = parallelIterationSpacesMatch(module, kernelToFuse, val); 
      if (fusionLegal && markedForFusion(kernelToFuse, val)) {
        kernelSet.extract(val);
        fusionSets[fusionSetIndex].push_back(val);
        fusionLegal = true; // reset flag
      }
    }

    // skip if there aren't any kernels to fuse with current kernel
    if (fusionSets[fusionSetIndex].size() == 1) {
      fusionSets[fusionSetIndex].pop_back();
      continue;
    }

    fusionSetIndex += 1;
  }
  return fusionSets;
}

#define GEN_PASS_DEF_KERNELFUSIONPASS
#include "Transform/Kernel/KernelPasses.h.inc"

struct KernelFusionPass : impl::KernelFusionPassBase<KernelFusionPass> {
  using KernelFusionPassBase::KernelFusionPassBase;

  void runOnOperation() override {

    // get the module and main function
    ModuleOp module = dyn_cast<ModuleOp>(getOperation());
    FuncOp mainFuncOp;
    for (FuncOp funcOp : module.getOps<FuncOp>()) {
      if (funcOp.getSymName() == "main") {
        mainFuncOp = funcOp;
        break;
      }
    }

    // 1. identify candidate kernels 
    CallMap callMap = getCallMap(mainFuncOp);

    // 2. create (profitable) fusion sets
    FusionSetVector fusionSets = createFusionSets(module, mainFuncOp, callMap);

    // 3. for each fusion set, create a new kernel that calls each subkernel
    // NOTE: we preserve MLIR ordering to respect dependences (probably not
    // permanent; probably not a good idea but :shrug: early implementation) 
    int fusedKernelCounter = 0;
    for (auto fusionSet : fusionSets) {
      if (fusionSet.empty())
        continue;

      // determine unique arguments/results included in each original kernel
      // call so that we can determine a type for the new kernel
      SmallVector<Value> newArgs;
      SmallVector<Value> newResults;
      SmallVector<Value> intermediates;
      DenseMap<Value, int> argsToIndexMap;
      DenseMap<Value, int> resultsToIndexMap;
      int fusedKernelArgIndex = 0;
      int fusedKernelResultIndex = 0;
      for (auto kernel : fusionSet) {
        // arguments
        for (auto arg : kernel.getOperands()) {
          auto argCheck = std::find(newArgs.begin(), newArgs.end(), arg);
          if (argCheck != newArgs.end())
            continue;

          // check if arg is produced by another kernel in the fusion set
          auto producer = arg.getDefiningOp();
          if (producer) {
            auto producerCheck =
                std::find(fusionSet.begin(), fusionSet.end(), producer);
            if (producerCheck != fusionSet.end())
              continue;
          }

          // add the operand to the type of the fused kernel
          newArgs.push_back(arg);
          argsToIndexMap[arg] = fusedKernelArgIndex;
          fusedKernelArgIndex++;
        }

        // results
        for (auto res : kernel.getResults()) {
          auto resCheck = std::find(newResults.begin(), newResults.end(), res);
          if (resCheck != newResults.end())
            continue;

          // check if the result is used by someone else in the fusion set
          bool userInFusionSet = false;
          for (auto user : res.getUsers()) {
            auto userCheck =
                std::find(fusionSet.begin(), fusionSet.end(), user);
            if (userCheck != fusionSet.end()) {
              userInFusionSet = true;
              break;
            }
          }

          if (userInFusionSet)
            continue;

          // add result to the type of the fused kernel 
          newResults.push_back(res);
          resultsToIndexMap[res] = fusedKernelResultIndex;
          fusedKernelResultIndex++;
        }
      }

      // build a new FuncOp for the fused kernel
      OpBuilder builder(getOperation()->getContext());
      TypeRange newArgTypes(newArgs);
      TypeRange newResultTypes(newResults);
      FunctionType fusedKernelType =
          builder.getFunctionType(newArgTypes, newResultTypes);

      // create a FuncOp and insert at the top of the module
      builder.setInsertionPointToStart(
          dyn_cast<ModuleOp>(getOperation()).getBody());
      StringRef fusedKernelName =
          "fusedKernel_" + std::to_string(fusedKernelCounter);
      FuncOp fusedKernelOp = builder.create<FuncOp>(
          module.getLoc(), fusedKernelName, fusedKernelType);

      // call each sub-kernel in the fused kernel
      Block *entry = fusedKernelOp.addEntryBlock();
      builder.setInsertionPointToStart(entry);

      // used to map results from new calls -> args of new calls
      using ArgsToIdxTy = std::vector<std::pair<Value, int>>;
      DenseMap<CallOp, ArgsToIdxTy>
          callsToIntermediateValues;

      // initialize the map
      for (CallOp kernel : fusionSet)
        callsToIntermediateValues[kernel] = ArgsToIdxTy();

      for (CallOp kernel : fusionSet) {
        // summon the kernel definition with evil LLVM magic
        FuncOp callee =
            dyn_cast<FuncOp>(SymbolTable::lookupNearestSymbolFrom(
                getOperation(),
                kernel.getCallableForCallee().get<SymbolRefAttr>()));

        // set the callee to private so inlining deletes the definition later
        if (!callee.isPrivate())
          callee.setPrivate();

        // get the arguments in the new kernel using the call from main
        SmallVector<Value> args;
        for (auto arg : kernel.getOperands())
          args.push_back(fusedKernelOp.getArgument(argsToIndexMap[arg]));

        // create the call
        auto newCallHandle =
            builder.create<CallOp>(fusedKernelOp.getLoc(), callee, args);

        // determine if results of old call are used by other kernels in set
        for (auto result : kernel.getResults()) {
          for (auto user : result.getUsers()) {
            auto userCheck =
                std::find(fusionSet.begin(), fusionSet.end(), user);
            if (userCheck != fusionSet.end()) {
              int argIndex = 0;
              CallOp userCall = *userCheck;
              for (auto arg : userCall.getOperands()) {
                if (arg == result) {
                  auto newArg = newCallHandle.getResult(argIndex);
                  callsToIntermediateValues[userCall].push_back(
                    std::make_pair(newArg, argIndex) 
                  );
                }
                argIndex++;
              }
            }
          }
        }

        // update arguments with intermediate results if necessary
        if (!callsToIntermediateValues[kernel].empty())
          for (auto argIndexPair : callsToIntermediateValues[kernel])
            newCallHandle.setOperand(argIndexPair.second, argIndexPair.first);

        // mark the new call as safe to inline
        newCallHandle.getOperation()->setAttr("inline",
                                              builder.getStringAttr("true"));
      }

      // construct a returnOp for the function; result has no uses => return it
      SmallVector<Value> returnOperands;
      for (auto newCall : fusedKernelOp.getOps<CallOp>()) {
        for (auto result : newCall.getResults()) {
          // compute the number of users
          int numUsers = 0;
          for (auto user : result.getUsers())
            numUsers++;

          if (numUsers == 0)
            returnOperands.push_back(result);
        }
      }
      builder.create<ReturnOp>(fusedKernelOp.getLoc(),
                                     ValueRange(returnOperands));

      // call the built funcOp
      builder.setInsertionPoint(*fusionSet.rbegin());
      CallOp fusedKernelCallHandle = builder.create<CallOp>(
          mainFuncOp.getLoc(), fusedKernelOp, newArgs);

      // update SSA values to make sense in the new kernel
      for (auto kernelCall : fusionSet) {
        for (auto result : kernelCall.getResults()) {
          auto newResult =
              fusedKernelCallHandle.getResult(resultsToIndexMap[result]);
          result.replaceAllUsesWith(newResult);
          kernelCall.erase();
        }
      } // end loop over kernels in fusionSet
      fusedKernelCounter += 1;
    } // end loop over fusionSets
  } // end runOnOperation
};
} // namespace kernel
} // namespace mlir

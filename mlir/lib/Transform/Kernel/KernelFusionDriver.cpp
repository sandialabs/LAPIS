/* Top level routine for automatic kernel fusion. Creates "fusion sets", i.e.
 * sets of subkernels that are to be fused together. Major steps:
 */

#include "Transform/Kernel/KernelPasses.h"
#include "Utils.cpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace kernel {

#define GEN_PASS_DEF_KERNELFUSIONDRIVER
#include "Transform/Kernel/KernelPasses.h.inc"

struct KernelFusionDriver : impl::KernelFusionDriverBase<KernelFusionDriver> {
  using KernelFusionDriverBase::KernelFusionDriverBase;

  EinsumSequence 
  getOptimalContractionOrder(func::FuncOp func) {
    std::vector<EinsumSpecification> einsums;
    for (linalg::LinalgOp laOp : func.getOps<linalg::LinalgOp>())
      einsums.push_back(genericToEinsumSpec(laOp));

    FusedEinsum fused = fuseEinsums(einsums);
    BruteForceOptimizer optimizer(fused);
    optimizer.optimize();

    return optimizer.optimizedEinsumSequence;
  }

  // TODO:
  void reorderGenerics(func::FuncOp func) {
    EinsumSequence optimalOrder =
        getOptimalContractionOrder(func);
    buildGenericsFromEinsums(func, optimalOrder);
  }

  void runOnOperation() override {
    mlir::ModuleOp module = dyn_cast<ModuleOp>(getOperation());
    OpPassManager driveKernelFusionPass;

    // find and move related calls into a new kernel
    driveKernelFusionPass.addPass(createKernelFusionPass());

    // inline the calls using a custom inlining pass
    driveKernelFusionPass.addPass(createFusedKernelInliningPass());

    // run the pipeline
    if (failed(runPipeline(driveKernelFusionPass, module)))
      return signalPassFailure();

    // reorder linalg.generics in each fused kernel
    for (func::FuncOp f :
         llvm::make_early_inc_range(module.getOps<func::FuncOp>())) {
      if (f.getSymName() == "main")
        continue;
      reorderGenerics(f);

      f.erase();
    }
  }
};
} // namespace kernel
} // namespace mlir

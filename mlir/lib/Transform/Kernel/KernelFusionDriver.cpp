/* Top level routine for automatic kernel fusion. Creates "fusion sets", i.e.
 * sets of subkernels that are to be fused together. Major steps:  
 */

#include "Transform/Kernel/KernelPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace kernel {

#define GEN_PASS_DEF_KERNELFUSIONDRIVER
#include "Transform/Kernel/KernelPasses.h.inc"

struct KernelFusionDriver : impl::KernelFusionDriverBase<KernelFusionDriver> {
  using KernelFusionDriverBase::KernelFusionDriverBase;

  void runOnOperation() override {
    mlir::ModuleOp module = dyn_cast<ModuleOp>(getOperation());
    OpPassManager driveKernelFusionPass;

    // find and move related calls into a new kernel
    driveKernelFusionPass.addPass(createKernelFusionPass());

    // inline the calls using a custom inlining pass
    driveKernelFusionPass.addPass(createFusedKernelInliningPass());

    // run the pipelin
    if (failed(runPipeline(driveKernelFusionPass, module)))
      return signalPassFailure();
  }
};
} // namespace kernel
} // namespace mlir

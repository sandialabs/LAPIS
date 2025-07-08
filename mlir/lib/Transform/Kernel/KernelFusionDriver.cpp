/* Top level routine for automatic kernel fusion. Creates "fusion sets", i.e.
 * sets of subkernels that are to be fused together. Major steps:
 */

#include "Transform/Kernel/KernelPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
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

  void runOnOperation() override {
    mlir::ModuleOp module = dyn_cast<ModuleOp>(getOperation());
    OpPassManager driveKernelFusionPass;

    // find and move related calls into a new kernel
    driveKernelFusionPass.addPass(createKernelFusionPass());

    // inline the calls using a custom inlining pass
    driveKernelFusionPass.addPass(createFusedKernelInliningPass());

    // reorder linalg generics to minimize temp size/computational cost
    // driveKernelFusionPass.addPass(createLinalgGenericReorderingPass());

    // run the pipeline
    if (failed(runPipeline(driveKernelFusionPass, module)))
      return signalPassFailure();

  }
};
} // namespace kernel
} // namespace mlir

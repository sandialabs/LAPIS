#ifndef KERNEL_FUSION_PASS
#define KERNEL_FUSION_PASS

include "mlir/Pass/PassBase.td"

def KernelFusionPass: Pass<"kernel-fusion-pass"> {
  let summary = "Fuses related subkernels into a single kernel.";
  let description = [{
    Fuses kernels by examining the arguments and results of kernel calls.
    Multiple calls to the same kernel are treated as unique instantiations of a
    kernel. 

    Once related calls are identified, they are put into sets. Once these sets
    are created, each subkernel call is moved into a new kernel. Afterward, a
    custom inling pass is run *only* over the fused kernels and the unused
    kernel definitions are removed.

    Once this pass is finished, the new fused kernels are ready for optimization
    and further lowering.
  }];
}

def FusedKernelInliningPass: Pass<"fused-kernel-inlining-pass"> {
  let summary = "Run a custom inlining pass over fused kernels";
  let constructor = "createFusedKernelInliningPass()";
  let description = [{
    Runs a restricted version of the MLIR inliner. During the KernelFusion pass,
    subkernel calls are tagged with an "inline" attribute. This inlining pass 
    *only* inlines calls with this attribute and *always* inlines them.

    This pass is meant to run directly after KernelFusion. However, it will work
    on any calls tagged with "inline", so it can be used outside of the context
    of kernel fusion.
  }];
}

def KernelDomainFusion: Pass<"fused-kernel-domain-fusion-pass"> {
  let summary = "Fuse the parallel loops of subkernels in a fused kernel";
  let constructor = "createKernelDomainFusionPass()";
  let description = [{
    Fuses the domains (i.e. the parallel loops) of subkernels in a fused kernel.
    Uses a custom version of the existing SCF parallel loop fusion pass. This
    custom pass *does not* consider aliasing. Aliasing is explicitly prohibited
    for now.
  }];
}

def KernelFusionDriver: Pass<"drive-kernel-fusion"> {
  let summary = "Drive a kernel fusion pass from unfused to optimized";
  let description = [{
    Runs all relevant passes to fuse, lower, and run a default optimization pass
    over a program containing computational kernels.
  }];
  let dependentDialects = [
    "arith::ArithDialect",
    "LLVM::LLVMDialect",
    "sparse_tensor::SparseTensorDialect",
    "bufferization::BufferizationDialect"
  ];
}

#endif

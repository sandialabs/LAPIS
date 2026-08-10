#ifndef FUNC_CUSTOM_PASSES_H
#define FUNC_CUSTOM_PASSES_H

#include "Transform/Kernel/KernelFusionPass.h"
#include "Transform/Kernel/KernelFusionDriver.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"

namespace mlir {
namespace kernel {

#define GEN_PASS_DECL_FUSEDKERNELINLININGPASS
std::unique_ptr<Pass> createFusedKernelInliningPass();

#define GEN_PASS_DECL_KERNELDOMAINFUSIONPASS
std::unique_ptr<Pass> createKernelDomainFusionPass();

#define GEN_PASS_DECL_LINALGGENERICREORDERINGPASS
std::unique_ptr<Pass> createLinalgGenericReorderingPass();

#define GEN_PASS_REGISTRATION
#include "Transform/Kernel/KernelPasses.h.inc"

}
}

#endif

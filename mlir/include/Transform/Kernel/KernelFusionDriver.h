#ifndef KERNEL_FUSION_DRIVER_H 
#define KERNEL_FUSION_DRIVER_H 

#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"

namespace mlir {
namespace kernel {

#define GEN_PASS_DECL_KERNELFUSIONDRIVER
#include "Transform/Kernel/KernelPasses.h.inc"

}
}


#endif

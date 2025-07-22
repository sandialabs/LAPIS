#ifndef KERNEL_FUSION_PASS_H
#define KERNEL_FUSION_PASS_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace kernel {

#define GEN_PASS_DECL_KERNELFUSIONPASS
#include "Transform/Kernel/KernelPasses.h.inc"

}
}

#endif

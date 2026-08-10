set(LLVM_TARGET_DEFINITIONS KernelPasses.td)
mlir_tablegen(KernelPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRKernelPassesIncGen)

add_mlir_library(MLIRKernelPasses
    KernelFusionPass.cpp    
    KernelFusionDriver.cpp
    KernelDomainFusion.cpp
    FusedKernelInliningPass.cpp
    LinalgGenericReordering.cpp
    Utils.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Transform/Kernel

    DEPENDS
    MLIRKernelPassesIncGen 
    MLIRFuncInlinerExtension

    LINK_LIBS PUBLIC
)

add_mlir_library(MLIRLapisTransforms
  AlgebraicKernelFusion.cpp
  ContractionPlanner.cpp
  Einsum.cpp
  FuseAlgebraicKernelLoops.cpp
  OutlineAlgebraicKernels.cpp

  ADDITIONAL_HEADER_DIRS
  ${LAPIS_SOURCE_DIR}/mlir/include/lapis/Transform

  DEPENDS
  MLIRLapisTransformPassIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRFuncDialect
  MLIRArithDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRRewrite
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTransforms
)

add_mlir_dialect_library(MLIRKokkosTransforms
  KokkosPasses.cpp
  ParallelUnitStep.cpp
  KokkosLoopMapping.cpp
  KokkosMemorySpaceAssignment.cpp
  KokkosDualViewManagement.cpp
  KokkosMdrangeIterationPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Kokkos

  DEPENDS
  MLIRKokkosPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRComplexDialect
  MLIREmitCDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRSCFUtils
  MLIRKokkosDialect
  MLIRTensorDialect
  MLIRTransforms
  MLIRVectorDialect
)

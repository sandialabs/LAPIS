IF(LAPIS_ENABLE_PART_TENSOR)
  SET(PartTensorLibs "MLIRPartTensor;MLIRPartTensorTransforms")
ENDIF()

add_mlir_dialect_library(MLIRKokkosPipelines
  KokkosPipelines.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Kokkos
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SparseTensor

  LINK_LIBS PUBLIC
  MLIRArithTransforms
  MLIRAffineToStandard
  MLIRBufferizationTransforms
  MLIRComplexToLLVM
  MLIRComplexToLibm
  MLIRComplexToStandard
  MLIRFuncTransforms
  MLIRLinalgTransforms
  MLIRMathToLibm
  MLIRMathToLLVM
  MLIRMemRefToLLVM
  MLIRPass
  MLIRReconcileUnrealizedCasts
  MLIRSCFToControlFlow
  MLIRKokkosDialect
  MLIRKokkosTransforms
  MLIRSparseTensorDialect
  MLIRSparseTensorTransforms
  MLIRTensorTransforms
  MLIRVectorToLLVM
  MLIRVectorTransforms
  ${PartTensorLibs}
)

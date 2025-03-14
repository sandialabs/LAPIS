SET(PartTensorLibs "")
IF(LAPIS_ENABLE_PART_TENSOR)
  SET(PartTensorLibs "MLIRPartTensor;MLIRPartTensorTransforms")
ENDIF()

SET(TorchMLIRLibs "")
IF(LAPIS_HAS_TORCH_MLIR)
  SET(TorchMLIRLibs "TorchMLIRTMTensorPasses;TorchMLIRRefBackend")
ENDIF()

add_mlir_dialect_library(MLIRKokkosPipelines
  KokkosPipelines.cpp

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
  MLIRShapeToStandard
  MLIRSparseTensorDialect
  MLIRSparseTensorTransforms
  MLIRTensorTransforms
  MLIRVectorToLLVM
  MLIRVectorTransforms
  ${PartTensorLibs}
  ${TorchMLIRLibs}
)

# Not needed?
#  ADDITIONAL_HEADER_DIRS
#  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Kokkos
#  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SparseTensor

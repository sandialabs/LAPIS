add_mlir_translation_library(MLIRTargetKokkosCpp
  TranslateRegistration.cpp
  TranslateToKokkosCpp.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIREmitCDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRSCFDialect
  MLIRSupport
  MLIRTranslateLib
  MLIRVectorDialect
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_executable(lapis-opt lapis-opt.cpp)
llvm_update_compile_flags(lapis-opt)
target_link_libraries(lapis-opt
  PRIVATE
  ${dialect_libs}
  ${translation_libs}
  ${test_libs}
  MLIRIR
  MLIRParser
  MLIRSPIRVDialect
  MLIRSupport
  MLIROptLib
  MLIRKernelPasses
  )

mlir_check_link_libraries(lapis-opt)
install(TARGETS lapis-opt)

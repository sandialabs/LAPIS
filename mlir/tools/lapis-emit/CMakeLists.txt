set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_llvm_executable(lapis-emit
  lapis-emit.cpp
  )
llvm_update_compile_flags(lapis-emit)
target_link_libraries(lapis-emit
  PRIVATE
  ${dialect_libs}
  ${translation_libs}
  ${test_libs}
  MLIRIR
  MLIRParser
  MLIRSupport
  )

mlir_check_link_libraries(lapis-emit)
install(TARGETS lapis-emit)

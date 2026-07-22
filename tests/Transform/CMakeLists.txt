find_package(GTest REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_executable(LAPISTransformTests
  AlgebraicKernelFusionTest.cpp
  ContractionPlannerTest.cpp
  EinsumTest.cpp
)
add_dependencies(LAPISTransformTests MLIRLapisTransformPassIncGen)
llvm_update_compile_flags(LAPISTransformTests)

target_include_directories(LAPISTransformTests
  PRIVATE
  ${LAPIS_SOURCE_DIR}/mlir/include
  ${LAPIS_BINARY_DIR}/mlir/include
)

target_link_libraries(LAPISTransformTests
  PRIVATE
  GTest::gtest_main
  LLVMSupport
  MLIRParser
  MLIRTensorDialect
  MLIRLapisTransforms
)

add_test(NAME LAPISTransformTests COMMAND LAPISTransformTests)

add_test(
  NAME AlgebraicKernelFusionNumerical
  COMMAND
    ${CMAKE_COMMAND} -E env
      "PATH=${LAPIS_TOOL_BIN}:$ENV{PATH}"
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/AlgebraicKernelFusionNumericalTest.py
      --lapis-opt $<TARGET_FILE:lapis-opt>
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/AlgebraicKernelFusionNumerical
)
set_tests_properties(
  AlgebraicKernelFusionNumerical
  PROPERTIES
    SKIP_RETURN_CODE 77
    TIMEOUT 180
)

find_package(GTest REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_executable(LAPISTransformTests
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
  MLIRLapisTransforms
)

add_test(NAME LAPISTransformTests COMMAND LAPISTransformTests)

add_test(
  NAME AlgebraicKernelFusionNumerical
  COMMAND
    ${CMAKE_COMMAND} -E env
      "PATH=${LAPIS_TOOL_BIN}:$ENV{PATH}"
      ${Python3_EXECUTABLE}
      ${LAPIS_SOURCE_DIR}/benchmarks/algebraic_kernel_fusion/run.py
      --case abx
      --case pcg
      --case linear_attention
      --warmup 1
      --iterations 1
      --lapis-opt $<TARGET_FILE:lapis-opt>
      --lapis-translate $<TARGET_FILE:lapis-translate>
      --build-dir ${CMAKE_CURRENT_BINARY_DIR}/AlgebraicKernelFusionNumerical
      --skip-if-unavailable
)
set_tests_properties(
  AlgebraicKernelFusionNumerical
  PROPERTIES
    SKIP_RETURN_CODE 77
    TIMEOUT 180
)

add_executable(LAPISTransformTests
  ContractionPlannerTest.cpp
  EinsumTest.cpp
  TransformTestMain.cpp
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
  LLVMSupport
  MLIRLapisTransforms
)

add_test(NAME LAPISTransformTests COMMAND LAPISTransformTests)

set(algebraic_kernel_fusion_numerical_args
  --case abx
  --case batched_linear_attention
  --case burgers
  --case pcg
  --case linear_attention
  --warmup 1
  --iterations 1
  --lapis-opt $<TARGET_FILE:lapis-opt>
  --lapis-translate $<TARGET_FILE:lapis-translate>
  --build-dir ${CMAKE_CURRENT_BINARY_DIR}/AlgebraicKernelFusionNumerical
  --skip-if-unavailable
)
if(LAPIS_BENCHMARK_BACKEND)
  list(APPEND algebraic_kernel_fusion_numerical_args
    --backend ${LAPIS_BENCHMARK_BACKEND})
endif()
if(LAPIS_BENCHMARK_KOKKOS_ROOT)
  list(APPEND algebraic_kernel_fusion_numerical_args
    --kokkos-root ${LAPIS_BENCHMARK_KOKKOS_ROOT})
endif()
if(LAPIS_BENCHMARK_CXX)
  list(APPEND algebraic_kernel_fusion_numerical_args
    --cxx ${LAPIS_BENCHMARK_CXX})
endif()
if(LAPIS_BENCHMARK_SUPPORT_LIB)
  list(APPEND algebraic_kernel_fusion_numerical_args
    --support-lib ${LAPIS_BENCHMARK_SUPPORT_LIB})
endif()

add_test(
  NAME AlgebraicKernelFusionNumerical
  COMMAND
    $<TARGET_FILE:lapis-algebraic-kernel-fusion-benchmark>
      ${algebraic_kernel_fusion_numerical_args}
)
set_tests_properties(
  AlgebraicKernelFusionNumerical
  PROPERTIES
    SKIP_RETURN_CODE 77
    RUN_SERIAL TRUE
    TIMEOUT 300
)

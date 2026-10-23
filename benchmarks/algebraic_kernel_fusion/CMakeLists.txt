add_executable(lapis-algebraic-kernel-fusion-benchmark
  BenchmarkRunner.cpp
)

set(LAPIS_BENCHMARK_BACKEND "$ENV{LAPIS_BENCHMARK_BACKEND}" CACHE STRING
  "Expected Kokkos backend for algebraic-fusion numerical tests")
set_property(CACHE LAPIS_BENCHMARK_BACKEND PROPERTY STRINGS
  serial openmp cuda hip sycl)
set(LAPIS_BENCHMARK_KOKKOS_ROOT "$ENV{KOKKOS_ROOT}" CACHE PATH
  "Kokkos package used by algebraic-fusion numerical tests")
set(LAPIS_BENCHMARK_CXX "$ENV{CXX}" CACHE FILEPATH
  "Kokkos-compatible compiler used by algebraic-fusion numerical tests")
set(LAPIS_BENCHMARK_SUPPORT_LIB "$ENV{SUPPORT_LIB}" CACHE FILEPATH
  "MLIR C runner support library used by algebraic-fusion numerical tests")

set(LLVM_REQUIRES_EH ON)
set(LLVM_REQUIRES_RTTI ON)
llvm_update_compile_flags(lapis-algebraic-kernel-fusion-benchmark)
unset(LLVM_REQUIRES_EH)
unset(LLVM_REQUIRES_RTTI)

target_compile_definitions(lapis-algebraic-kernel-fusion-benchmark
  PRIVATE
  LAPIS_ALGEBRAIC_FUSION_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
  LAPIS_REPOSITORY_DIR="${LAPIS_SOURCE_DIR}"
)

target_link_libraries(lapis-algebraic-kernel-fusion-benchmark
  PRIVATE
  LLVMSupport
)

add_dependencies(lapis-algebraic-kernel-fusion-benchmark
  lapis-opt
  lapis-translate
)

set_target_properties(lapis-algebraic-kernel-fusion-benchmark
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LLVM_RUNTIME_OUTPUT_INTDIR}
)

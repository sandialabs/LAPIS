project(pytaco_cpp_driver)
cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

find_package(Kokkos REQUIRED
  PATHS "$ENV{KOKKOS_ROOT}")

if(NOT IS_DIRECTORY $ENV{LLVM_SOURCE_ROOT})
  message(FATAL_ERROR "Set environment variable LLVM_SOURCE_ROOT to the absolute path of the LLVM source directory: /path/to/llvm-project")
endif()

add_library(mlir_c_runner_utils_static
  STATIC
  $ENV{LLVM_SOURCE_ROOT}/mlir/lib/ExecutionEngine/CRunnerUtils.cpp
  $ENV{LLVM_SOURCE_ROOT}/mlir/lib/ExecutionEngine/Float16bits.cpp
  $ENV{LLVM_SOURCE_ROOT}/mlir/lib/ExecutionEngine/SparseTensorRuntime.cpp
  $ENV{LLVM_SOURCE_ROOT}/mlir/lib/ExecutionEngine/SparseTensor/File.cpp
  $ENV{LLVM_SOURCE_ROOT}/mlir/lib/ExecutionEngine/SparseTensor/NNZ.cpp
  $ENV{LLVM_SOURCE_ROOT}/mlir/lib/ExecutionEngine/SparseTensor/PermutationRef.cpp
  $ENV{LLVM_SOURCE_ROOT}/mlir/lib/ExecutionEngine/SparseTensor/Storage.cpp)

set_property(TARGET mlir_c_runner_utils_static PROPERTY CXX_STANDARD 17)

add_executable(pytaco_cpp main.cpp)

include_directories($ENV{LLVM_SOURCE_ROOT}/mlir/include)

target_link_libraries(pytaco_cpp Kokkos::kokkos)
target_link_libraries(pytaco_cpp mlir_c_runner_utils_static)

cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(lapis_package)
find_package(Kokkos REQUIRED)

add_subdirectory(batched_gemm)
add_subdirectory(gemm)
add_subdirectory(MALA)
add_subdirectory(resnet)
add_subdirectory(spmv)


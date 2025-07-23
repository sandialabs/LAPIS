cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(MALA_Batch_Test)
find_package(Kokkos REQUIRED)
add_executable(MALA_Batch main.cpp)
target_link_libraries(MALA_Batch Kokkos::kokkos)


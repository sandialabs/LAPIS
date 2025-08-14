cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(maxpool_team)
find_package(Kokkos REQUIRED)
add_executable(maxpool_nchw_teamlevel main.cpp maxpool.cpp)
target_link_libraries(maxpool_nchw_teamlevel Kokkos::kokkos)


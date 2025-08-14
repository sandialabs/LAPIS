cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(cloudfrac_model)
find_package(Kokkos REQUIRED)
add_executable(cloudfrac main.cpp cloudfrac.cpp)
target_link_libraries(cloudfrac Kokkos::kokkos)


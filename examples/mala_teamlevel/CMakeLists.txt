cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(MALA_TeamLevel_Test)
find_package(Kokkos REQUIRED)
add_executable(MALA_TeamLevel main.cpp forward_snap.cpp)
target_link_libraries(MALA_TeamLevel Kokkos::kokkos)


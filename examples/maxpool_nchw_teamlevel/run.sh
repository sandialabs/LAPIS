#!/bin/bash

set -e

echo "Lowering with team-level pipeline..."
lapis-opt --team-compiler-kokkos maxpool.mlir -o maxpool_lowered.mlir
echo "Emitting Kokkos..."
lapis-translate maxpool_lowered.mlir --team-level -o maxpool.cpp --hpp maxpool.hpp

mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
echo "Building executable..."
make -j2
cd ..
./build/maxpool_nchw_teamlevel


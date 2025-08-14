#!/bin/bash

set -e

echo "Lowering with team-level pipeline..."
lapis-opt --team-compiler-kokkos cloudfrac.mlir -o cloudfrac_lowered.mlir
echo "Emitting Kokkos..."
lapis-translate cloudfrac_lowered.mlir --team-level -o cloudfrac.cpp --hpp cloudfrac.hpp

mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
echo "Building executable..."
make -j2
cd ..
./build/cloudfrac


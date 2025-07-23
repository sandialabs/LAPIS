#!/bin/bash

set -e

echo "Lowering with team-level pipeline..."
lapis-opt --sparse-compiler-kokkos forward_snap_batch.mlir -o forward_snap_lowered.mlir
echo "Emitting Kokkos..."
lapis-translate forward_snap_lowered.mlir -o forward_snap.hpp

mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
echo "Building executable..."
make
cd ..
./build/MALA_Batch


#!/bin/bash

set -e

echo "Lowering with team-level pipeline..."
lapis-opt --team-compiler-kokkos forward_snap.mlir -o forward_snap_lowered.mlir
echo "Emitting Kokkos..."
lapis-translate forward_snap_lowered.mlir --team-level -o forward_snap.cpp --hpp forward_snap.hpp

mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
echo "Building executable..."
make -j2
cd ..
./build/MALA_TeamLevel


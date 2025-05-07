NOTE: this repository is not a CMake subdirectory of
LAPIS, but is its own project. This way, LLVM/MLIR
and LAPIS don't need to be built on every test machine.

Instead, LAPIS can be built on one test machine to generate C++ files,
and then this benchmark directory can be copied to each test
machine to build and run the benchmarks.

Inside each of the five subdirectories (batched_gemm, gemm, MALA, resnet, spmv),
there is another README.txt describing how to use LAPIS to compile each model or
kernel to Kokkos C++. They also describe the input data (if any).

After generating the five C++ files, copy this directory to the test machine(s).
These should have Kokkos installed with the correct backend as described in the
AD appendix. On each machine, environment variable KOKKOS_ROOT should point to the
installed location, and Kokkos_DIR should point to the directory inside $KOKKOS_ROOT
containing KokkosConfig.cmake.

Then compile the benchmarks:
  mkdir build
  cd build
  cmake -DCMAKE_CXX_FLAGS="-Wno-c++11-narrowing" ..
  make
  cd ..

and run them from this directory:

Assuming that the 7 spmv matrices are in $MATRIX_DIR.
  build/batched_gemm/batched_gemm
  build/gemm/gemm
  build/resnet/resnet
  build/MALA/MALA

  build/spmv/spmv $MATRIX_DIR/StocF-1465.mtx
  build/spmv/spmv $MATRIX_DIR/PFlow_742.mtx
  build/spmv/spmv $MATRIX_DIR/Emilia_923.mtx
  build/spmv/spmv $MATRIX_DIR/Elasticity3D_60.mtx
  build/spmv/spmv $MATRIX_DIR/Hook_1498.mtx
  build/spmv/spmv $MATRIX_DIR/Serena.mtx
  build/spmv/spmv $MATRIX_DIR/audikw_1.mtx

This will print out the averaged running times that are then plotted for the paper's figures.
To reproduce the actual figures, replace the times in these files with those printed by the benchmarks:
  * batched_gemm.csv
  * gemm.csv
  * plotMALA.py
  * plotResnet.py
  * spmv.csv

then install matplotlib and run the plot***.py scripts to generate the figures.

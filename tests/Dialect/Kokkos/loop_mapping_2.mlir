// RUN: %lapis-opt %s --kokkos-loop-mapping | diff %s.gold -

// spmv-like kernel: 2D nested loop where the inner loop does a reduction.
// The outer loop becomes a kokkos.thread_parallel.
// The inner loop becomes a ThreadVector range_parallel.

module {
  func.func @spmv(%arg0: memref<?xf64>, %arg1: memref<?xindex>, %arg2: memref<?xindex>, %arg3: memref<?xf64>, %arg4: memref<?xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cN = memref.dim %arg0, %c0 : memref<?xf64>
    scf.parallel (%arg5) = (%c0) to (%cN) step (%c1) {
      %0 = memref.load %arg0[%arg5] : memref<?xf64>
      %1 = memref.load %arg1[%arg5] : memref<?xindex>
      %2 = arith.addi %arg5, %c1 : index
      %3 = memref.load %arg1[%2] : memref<?xindex>
      %4 = arith.subi %3, %1 : index
      %5 = scf.parallel (%arg6) = (%c0) to (%4) step (%c1) init (%0) -> f64 {
        %6 = arith.addi %1, %arg6 : index
        %7 = memref.load %arg2[%6] : memref<?xindex>
        %8 = memref.load %arg3[%6] : memref<?xf64>
        %9 = memref.load %arg4[%7] : memref<?xf64>
        %10 = arith.mulf %8, %9 : f64
        scf.reduce(%10 : f64) {
        ^bb0(%arg7: f64, %arg8: f64):
          %11 = arith.addf %arg7, %arg8 : f64
          scf.reduce.return %11 : f64
        }
      }
      memref.store %5, %arg0[%arg5] : memref<?xf64>
      scf.reduce
    }
    return
  }
}


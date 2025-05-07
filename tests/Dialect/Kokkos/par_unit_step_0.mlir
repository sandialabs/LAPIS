// RUN: %lapis-opt %s --parallel-unit-step | diff %s.gold -
module {
  func.func @myfunc(%arg0: memref<?x?x?xf64>, %arg1: memref<?xindex>, %arg2: memref<?xf64>, %arg3: memref<?xf64>) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c5 = arith.constant 5 : index
    // This loop already counts from 0 and has unit stride,
    // so the pass should leave it alone
    scf.parallel (%arg4) = (%c0) to (%c5) step (%c1) {
      %0 = memref.load %arg0[%arg4, %c0, %c1] : memref<?x?x?xf64>
      %1 = memref.load %arg1[%arg4] : memref<?xindex>
      %2 = arith.addi %arg4, %c1 : index
      %3 = memref.load %arg1[%2] : memref<?xindex>
      // But this loop counts from %1 so the
      // pass should shift the iteration range so it starts at 0
      %4 = scf.parallel (%arg5) = (%1) to (%3) step (%c1) init (%0) -> f64 {
        %5 = memref.load %arg1[%arg5] : memref<?xindex>
        %6 = memref.load %arg2[%arg5] : memref<?xf64>
        %7 = memref.load %arg3[%5] : memref<?xf64>
        %8 = arith.mulf %6, %7 : f64
        scf.reduce(%8 : f64) {
        ^bb0(%arg7: f64, %arg8: f64):
          %9 = arith.addf %arg7, %arg8 : f64
          scf.reduce.return %9 : f64
        }
      }
      memref.store %4, %arg0[%arg4, %arg4, %arg4] : memref<?x?x?xf64>
      scf.reduce
    }
    return
  }
}

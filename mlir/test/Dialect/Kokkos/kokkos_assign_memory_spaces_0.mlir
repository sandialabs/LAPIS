// RUN: %lapis-opt %s --kokkos-assign-memory-spaces
module {
// CHECK-LABEL:   func.func @myfunc(
// CHECK-SAME:                      %[[VAL_0:.*]]: memref<?x?x?xf64>,
// CHECK-SAME:                      %[[VAL_1:.*]]: memref<?xindex>,
// CHECK-SAME:                      %[[VAL_2:.*]]: memref<?xf64>,
// CHECK-SAME:                      %[[VAL_3:.*]]: memref<?xf64>) {
// CHECK:           %c1 = arith.constant 1 : index
// CHECK:           %c0 = arith.constant 0 : index
// CHECK:           %c5 = arith.constant 5 : index
// CHECK:           scf.parallel (%arg4) = (%c0) to (%c5) step (%c1) {
// CHECK:             %0 = memref.load %arg0[%arg4, %c0, %c1] {memorySpace = 1 : i64} : memref<?x?x?xf64>
// CHECK:             %1 = memref.load %arg1[%arg4] {memorySpace = 1 : i64} : memref<?xindex>
// CHECK:             %2 = arith.addi %arg4, %c1 : index
// CHECK:             %3 = memref.load %arg1[%2] {memorySpace = 1 : i64} : memref<?xindex>
// CHECK:             %4 = scf.parallel (%arg5) = (%1) to (%3) step (%c1) init (%0) -> f64 {
// CHECK:               %5 = memref.load %arg1[%arg5] {memorySpace = 1 : i64} : memref<?xindex>
// CHECK:               %6 = memref.load %arg2[%arg5] {memorySpace = 1 : i64} : memref<?xf64>
// CHECK:               %7 = memref.load %arg3[%5] {memorySpace = 1 : i64} : memref<?xf64>
// CHECK:               %8 = arith.mulf %6, %7 : f64
// CHECK:               scf.reduce(%8)  : f64 {
// CHECK:               ^bb0(%arg6: f64, %arg7: f64):
// CHECK:                 %9 = arith.addf %arg6, %arg7 : f64
// CHECK:                 scf.reduce.return %9 : f64
// CHECK:               }
// CHECK:               scf.yield
// CHECK:             }
// CHECK:             memref.store %4, %arg0[%arg4, %arg4, %arg4] {memorySpace = 1 : i64} : memref<?x?x?xf64>
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           return
// CHECK:         }
  func.func @myfunc(%arg0: memref<?x?x?xf64>, %arg1: memref<?xindex>, %arg2: memref<?xf64>, %arg3: memref<?xf64>) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c5 = arith.constant 5 : index
    scf.parallel (%arg4) = (%c0) to (%c5) step (%c1) {
      %0 = memref.load %arg0[%arg4, %c0, %c1] : memref<?x?x?xf64>
      %1 = memref.load %arg1[%arg4] : memref<?xindex>
      %2 = arith.addi %arg4, %c1 : index
      %3 = memref.load %arg1[%2] : memref<?xindex>
      %4 = scf.parallel (%arg5) = (%1) to (%3) step (%c1) init (%0) -> f64 {
        %5 = memref.load %arg1[%arg5] : memref<?xindex>
        %6 = memref.load %arg2[%arg5] : memref<?xf64>
        %7 = memref.load %arg3[%5] : memref<?xf64>
        %8 = arith.mulf %6, %7 : f64
        scf.reduce(%8)  : f64 {
        ^bb0(%arg7: f64, %arg8: f64):
          %9 = arith.addf %arg7, %arg8 : f64
          scf.reduce.return %9 : f64
        }
        scf.yield
      }
      memref.store %4, %arg0[%arg4, %arg4, %arg4] : memref<?x?x?xf64>
      scf.yield
    }
    return
  }
}

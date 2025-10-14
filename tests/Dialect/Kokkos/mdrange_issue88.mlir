// RUN: %lapis-opt %s --kokkos-mdrange-iteration | diff %s.gold -
module {
  func.func private @sparseCoordinates32(!llvm.ptr, index) -> memref<?xi32> attributes {llvm.emit_c_interface}
  func.func private @sparsePositions32(!llvm.ptr, index) -> memref<?xi32> attributes {llvm.emit_c_interface}
  func.func private @sparseLvlSize(!llvm.ptr, index) -> index
  func.func private @sparseValuesF64(!llvm.ptr) -> memref<?xf64> attributes {llvm.emit_c_interface}
  func.func @spmv(%arg0: !llvm.ptr, %arg1: memref<?xf64>, %arg2: memref<?xf64>) -> memref<?xf64> {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = call @sparseValuesF64(%arg0) : (!llvm.ptr) -> memref<?xf64>
    %1 = call @sparseLvlSize(%arg0, %c0) : (!llvm.ptr, index) -> index
    %2 = call @sparsePositions32(%arg0, %c1) : (!llvm.ptr, index) -> memref<?xi32>
    %3 = call @sparseCoordinates32(%arg0, %c1) : (!llvm.ptr, index) -> memref<?xi32>
    scf.parallel (%arg3) = (%c0) to (%1) step (%c1) {
      %4 = memref.load %arg2[%arg3] : memref<?xf64>
      %5 = memref.load %2[%arg3] : memref<?xi32>
      %6 = arith.extui %5 : i32 to i64
      %7 = arith.index_cast %6 : i64 to index
      %8 = arith.addi %arg3, %c1 : index
      %9 = memref.load %2[%8] : memref<?xi32>
      %10 = arith.extui %9 : i32 to i64
      %11 = arith.index_cast %10 : i64 to index
      %12 = scf.parallel (%arg4) = (%7) to (%11) step (%c1) init (%4) -> f64 {
        %13 = memref.load %3[%arg4] : memref<?xi32>
        %14 = arith.extui %13 : i32 to i64
        %15 = arith.index_cast %14 : i64 to index
        %16 = memref.load %0[%arg4] : memref<?xf64>
        %17 = memref.load %arg1[%15] : memref<?xf64>
        %18 = arith.mulf %16, %17 : f64
        scf.reduce(%18 : f64) {
        ^bb0(%arg5: f64, %arg6: f64):
          %19 = arith.addf %arg5, %arg6 : f64
          scf.reduce.return %19 : f64
        }
      } {"Emitted from" = "linalg.generic"}
      memref.store %12, %arg2[%arg3] : memref<?xf64>
      scf.reduce 
    } {"Emitted from" = "linalg.generic"}
    return %arg2 : memref<?xf64>
  }
}

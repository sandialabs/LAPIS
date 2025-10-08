module attributes {torch.debug_module_name = "MaxPool2d"} {
  func.func @forward(%arg0: memref<1x3x6x10xf32>, %arg1: memref<1x3x20x30xf32>) {
    %c30 = arith.constant 30 : index
    %c20 = arith.constant 20 : index
    %c2 = arith.constant 2 : index
    %c10 = arith.constant 10 : index
    %c6 = arith.constant 6 : index
    %c32 = arith.constant 32 : index
    %c22 = arith.constant 22 : index
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0xFF800000 : f32
    %0 = "kokkos.alloc_scratch"() <{offset = 0 : index}> : () -> memref<1x3x22x32xf32>
    kokkos.range_parallel (%arg2, %arg3, %arg4, %arg5) -> (%c1, %c3, %c22, %c32) {
      memref.store %cst, %0[%arg2, %arg3, %arg4, %arg5] : memref<1x3x22x32xf32>
      kokkos.yield
    } {executionSpace = 2 : i64, parallelLevel = 1 : i64}
    "kokkos.team_barrier"() : () -> ()
    %reinterpret_cast = memref.reinterpret_cast %0 to offset: [33], sizes: [1, 3, 20, 30], strides: [2112, 704, 32, 1] : memref<1x3x22x32xf32> to memref<1x3x20x30xf32, strided<[2112, 704, 32, 1], offset: 33>>
    kokkos.range_parallel (%arg2, %arg3, %arg4, %arg5) -> (%c1, %c3, %c20, %c30) {
      %1 = memref.load %arg1[%arg2, %arg3, %arg4, %arg5] : memref<1x3x20x30xf32>
      memref.store %1, %reinterpret_cast[%arg2, %arg3, %arg4, %arg5] : memref<1x3x20x30xf32, strided<[2112, 704, 32, 1], offset: 33>>
      kokkos.yield
    } {executionSpace = 2 : i64, parallelLevel = 1 : i64}
    "kokkos.team_barrier"() : () -> ()
    kokkos.range_parallel (%arg2, %arg3, %arg4, %arg5) -> (%c1, %c3, %c6, %c10) {
      memref.store %cst, %arg0[%arg2, %arg3, %arg4, %arg5] : memref<1x3x6x10xf32>
      kokkos.yield
    } {executionSpace = 2 : i64, parallelLevel = 1 : i64}
    "kokkos.team_barrier"() : () -> ()
    kokkos.range_parallel (%arg2, %arg3, %arg4, %arg5) -> (%c1, %c3, %c6, %c10) {
      scf.for %arg6 = %c0 to %c3 step %c1 {
        scf.for %arg7 = %c0 to %c3 step %c1 {
          %1 = arith.muli %arg4, %c3 : index
          %2 = arith.muli %arg6, %c2 : index
          %3 = arith.addi %1, %2 : index
          %4 = arith.muli %arg5, %c3 : index
          %5 = arith.muli %arg7, %c2 : index
          %6 = arith.addi %4, %5 : index
          %7 = memref.load %0[%arg2, %arg3, %3, %6] : memref<1x3x22x32xf32>
          %8 = memref.load %arg0[%arg2, %arg3, %arg4, %arg5] : memref<1x3x6x10xf32>
          %9 = arith.maximumf %8, %7 : f32
          memref.store %9, %arg0[%arg2, %arg3, %arg4, %arg5] : memref<1x3x6x10xf32>
        }
      }
      kokkos.yield
    } {executionSpace = 2 : i64, parallelLevel = 1 : i64}
    "kokkos.team_barrier"() : () -> ()
    return
  }
}


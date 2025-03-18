// RUN: %lapis-opt %s --parallel-unit-step | diff %s.gold -
module {
  func.func @myfunc(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: index) -> memref<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c5 = arith.constant 5 : index
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc(%arg2, %arg2) {alignment = 64 : i64} : memref<?x?xf32>
    // This 2D loop is already unit stride in both dimensions,
    // so the pass should leave it alone
    scf.parallel (%i, %j) = (%c0, %c0) to (%c5, %arg2) step (%c1, %c1) {
      %0 = memref.load %arg0[%i, %j] : memref<?x?xf32>
      %1 = memref.load %arg1[%j, %j] : memref<?x?xf32>
      %2 = arith.addf %0, %1 : f32
      memref.store %2, %alloc[%j, %i] : memref<?x?xf32>
      scf.reduce
    }
    return %alloc : memref<?x?xf32>
  }
}

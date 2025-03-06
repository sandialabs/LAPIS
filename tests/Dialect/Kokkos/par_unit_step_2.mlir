// RUN: %lapis-opt %s --parallel-unit-step | diff %s.gold -
module {
  func.func @myfunc(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?xf32>, %arg2: index, %arg3: index, %arg4: index) -> memref<?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cPI = arith.constant 3.14159 : f32
    // All three loop dimensions should be updated by the pass:
    // i counts from 1, so it should be shifted
    // j counts from 0 but has non-unit stride
    // k counts from 1 and also has non-unit stride
    // so the pass should leave it alone
    scf.parallel (%i, %j, %k) = (%c1, %c0, %c1) to (%arg4, %arg4, %arg4) step (%c1, %arg3, %arg3) {
      %0 = memref.load %arg0[%i, %j, %k] : memref<?x?x?xf32>
      %2 = arith.addf %0, %cPI : f32
      memref.store %2, %arg0[%i, %j, %k] : memref<?x?x?xf32>
      scf.reduce
    }
    return %arg0 : memref<?x?x?xf32>
  }
}

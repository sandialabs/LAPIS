// RUN: %lapis-opt %s --kokkos-loop-mapping | diff %s.gold -
module {
  func.func @myfunc(%arg0: memref<?x?x?xf64>, %arg1: memref<?xindex>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c5 = arith.constant 5 : index
    // This 1D loop should be replaced with a 2D kokkos.range_parallel on host
    // (it cannot be offloaded to device because it contains an allocation)
    scf.parallel (%i, %j) = (%c0, %c0) to (%c5, %arg4) step (%c1, %c1) {
      %alloc = memref.alloc(%c5) : memref<?xf32>
      // Do something with %alloc so it's not eliminated by canonicalization
      %cPI = arith.constant 3.14159 : f32
      scf.for %k = %c0 to %c5 step %c1 {
        memref.store %cPI, %alloc[%k] : memref<?xf32>
        scf.yield
      }
      scf.reduce
    }
    return
  }
}

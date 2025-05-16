module {
  func.func @example_function(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // like 1c, but dynamic extents
    // store needs to be reversed, load is okay as-is, store is modeled
    // as more expensive, so loop should be reversed
    scf.parallel (%i, %j) = (%c0, %c0) to (%arg2, %arg2) step (%c1, %c1) {
      %val = memref.load %arg0[%j, %i] : memref<?x?xf32>
      memref.store %val, %arg1[%i, %j] : memref<?x?xf32>
      scf.reduce
    }
    return
  }
}
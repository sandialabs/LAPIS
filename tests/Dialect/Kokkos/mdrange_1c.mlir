module {
  func.func @example_function(%arg0: memref<20x20xf32>, %arg1: memref<?x20xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c20 = arith.constant 20 : index

    // store needs to be reversed, load is okay as-is, store is modeled
    // as more expensive, so loop should be reversed
    scf.parallel (%i, %j) = (%c0, %c0) to (%c20, %c20) step (%c1, %c1) {
      %val = memref.load %arg0[%j, %i] : memref<20x20xf32>
      memref.store %val, %arg1[%i, %j] : memref<?x20xf32>
      scf.reduce
    }
    return
  }
}
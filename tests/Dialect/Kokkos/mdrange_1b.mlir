module {
  func.func @example_function(%arg0: memref<10x20xf32>, %arg1: memref<?x20xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %c20 = arith.constant 20 : index

    // okay as-is
    scf.parallel (%i, %j) = (%c0, %c0) to (%c20, %c10) step (%c1, %c1) {
      %val = memref.load %arg0[%j, %i] : memref<10x20xf32>
      memref.store %val, %arg1[%j, %i] : memref<?x20xf32>
      scf.reduce
    }
    return
  }
}
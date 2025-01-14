module {
  func.func @example_function(%arg0: memref<10x20xf32>, %arg1: memref<10x20xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %c20 = arith.constant 20 : index

    scf.parallel (%i, %j) = (%c0, %c0) to (%c10, %c20) step (%c1, %c1) {
      %val = memref.load %arg0[%i, %j] : memref<10x20xf32>
      memref.store %val, %arg1[%i, %j] : memref<10x20xf32>
    }
    return
  }
}
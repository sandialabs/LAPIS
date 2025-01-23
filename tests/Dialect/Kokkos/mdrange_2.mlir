module {
  func.func @example_function(%arg0: memref<10x20xf32>, %arg1: memref<?x20xf32>, %loop_bound_i: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    scf.parallel (%i, %j) = (%c0, %c0) to (%loop_bound_i, %c2) step (%c1, %c1) {
      %val = memref.load %arg0[%i, %j] : memref<10x20xf32>
      memref.store %val, %arg1[%i, %j] : memref<?x20xf32>
    }
    return
  }
}

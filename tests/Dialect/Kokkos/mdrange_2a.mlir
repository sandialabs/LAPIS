module {
  func.func @example_function(%arg0: memref<?x?xf32>,
                              %ub_i: index,
                              %ub_j: index,
                              %ub_k: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    scf.parallel (%i, %j) = (%c0, %c0) to (%ub_i, %ub_j) step (%c1, %c1) {

      // this loop wants the outer loop indices to be reversed
      scf.parallel (%k) = (%c0) to (%ub_k) step (%c1) {
        %val = memref.load %arg0[%i, %j] : memref<?x?xf32>
        memref.store %val, %arg0[%i, %j] : memref<?x?xf32>
        scf.reduce
      }

      scf.reduce
    }
    return
  }
}

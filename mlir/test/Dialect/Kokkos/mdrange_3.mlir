module {
  func.func @nested_parallel(%arg0: index, %arg1: index, %arg2: index) {
    %0 = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
    %1 = memref.alloc(%arg1, %arg2) : memref<?x?xf32>
    %2 = memref.alloc(%arg0, %arg2) : memref<?x?xf32>

    %3 = arith.addi %arg0, %arg1 : index
    %4 = arith.addi %arg1, %arg2 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c10 = arith.constant 10 : index
    // Outer parallel loop with dynamic bounds and stride
    scf.parallel (%i, %j) = (%arg0, %arg1) to (%3, %4) step (%arg1, %c2) {
      // Inner parallel loop with constant bounds and stride
      scf.parallel (%k) = (%c0) to (%c10) step (%c1) {
        %val0 = memref.load %0[%i, %j] : memref<?x?xf32>
        %val1 = memref.load %1[%j, %k] : memref<?x?xf32>
        %result = arith.addf %val0, %val1 : f32
        memref.store %result, %2[%i, %k] : memref<?x?xf32>
      }
    }

    memref.dealloc %0 : memref<?x?xf32>
    memref.dealloc %1 : memref<?x?xf32>
    memref.dealloc %2 : memref<?x?xf32>
    return
  }
}
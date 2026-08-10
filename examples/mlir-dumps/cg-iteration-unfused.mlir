module {
  func.func @gemv(%arg0: memref<128x128xf64>, %arg1: memref<128xf64>, %arg2: memref<128xf64>) -> memref<128xf64> {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    scf.parallel (%arg3) = (%c0) to (%c128) step (%c1) {
      scf.for %arg4 = %c0 to %c128 step %c1 {
        %0 = memref.load %arg0[%arg3, %arg4] : memref<128x128xf64>
        %1 = memref.load %arg1[%arg4] : memref<128xf64>
        %2 = memref.load %arg2[%arg3] : memref<128xf64>
        %3 = arith.mulf %0, %1 : f64
        %4 = arith.addf %2, %3 : f64
        memref.store %4, %arg2[%arg3] : memref<128xf64>
      }
      scf.reduce 
    }
    return %arg2 : memref<128xf64>
  }
  func.func @xpy(%arg0: memref<128xf64>, %arg1: memref<128xf64>, %arg2: memref<128xf64>) -> memref<128xf64> {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    scf.parallel (%arg3) = (%c0) to (%c128) step (%c1) {
      %0 = memref.load %arg0[%arg3] : memref<128xf64>
      %1 = memref.load %arg1[%arg3] : memref<128xf64>
      %2 = arith.addf %0, %1 : f64
      memref.store %2, %arg2[%arg3] : memref<128xf64>
      scf.reduce 
    }
    return %arg2 : memref<128xf64>
  }
  func.func @dot(%arg0: memref<128xf64>, %arg1: memref<128xf64>, %arg2: memref<f64>) -> memref<f64> {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    scf.for %arg3 = %c0 to %c128 step %c1 {
      %0 = memref.load %arg0[%arg3] : memref<128xf64>
      %1 = memref.load %arg1[%arg3] : memref<128xf64>
      %2 = memref.load %arg2[] : memref<f64>
      %3 = arith.mulf %0, %1 : f64
      %4 = arith.addf %2, %3 : f64
      memref.store %4, %arg2[] : memref<f64>
    }
    return %arg2 : memref<f64>
  }
  func.func @dscal(%arg0: memref<f64>, %arg1: memref<128xf64>, %arg2: memref<128xf64>) -> memref<128xf64> {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    scf.parallel (%arg3) = (%c0) to (%c128) step (%c1) {
      %0 = memref.load %arg0[] : memref<f64>
      %1 = memref.load %arg1[%arg3] : memref<128xf64>
      %2 = arith.mulf %0, %1 : f64
      memref.store %2, %arg2[%arg3] : memref<128xf64>
      scf.reduce 
    }
    return %arg2 : memref<128xf64>
  }
  func.func @div(%arg0: memref<f64>, %arg1: memref<f64>, %arg2: memref<f64>) -> memref<f64> {
    %0 = memref.load %arg0[] : memref<f64>
    %1 = memref.load %arg1[] : memref<f64>
    %2 = arith.divf %0, %1 : f64
    memref.store %2, %arg2[] : memref<f64>
    return %arg2 : memref<f64>
  }
  func.func @neg(%arg0: memref<f64>, %arg1: memref<f64>) -> memref<f64> {
    %0 = memref.load %arg0[] : memref<f64>
    %1 = arith.negf %0 : f64
    memref.store %1, %arg1[] : memref<f64>
    return %arg1 : memref<f64>
  }
  func.func @main(%arg0: memref<128x128xf64>, %arg1: memref<128xf64>, %arg2: memref<128xf64>, %arg3: memref<128xf64>) -> memref<128xf64> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<f64>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<128xf64>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<f64>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<128xf64>
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<128xf64>
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<f64>
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<128xf64>
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<128xf64>
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<f64>
    %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<128xf64>
    %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<128xf64>
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<f64>
    %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<f64>


    %0 = call @dot(%arg2, %arg2, %alloc) {fuse_with = "gemv"} : (memref<128xf64>, memref<128xf64>, memref<f64>) -> memref<f64>
    %1 = call @gemv(%arg0, %arg3, %alloc_0) {fuse_with = "dot"} : (memref<128x128xf64>, memref<128xf64>, memref<128xf64>) -> memref<128xf64>
    %2 = call @dot(%1, %arg3, %alloc_1) : (memref<128xf64>, memref<128xf64>, memref<f64>) -> memref<f64>

    %3 = call @div(%0, %2, %alloc_10) {fuse_with = ""} : (memref<f64>, memref<f64>, memref<f64>) -> memref<f64>
    %4 = call @dscal(%3, %arg3, %alloc_2) {fuse_with = ""} : (memref<f64>, memref<128xf64>, memref<128xf64>) -> memref<128xf64>
    %5 = call @xpy(%arg1, %4, %alloc_3) {fuse_with = ""} : (memref<128xf64>, memref<128xf64>, memref<128xf64>) -> memref<128xf64>
    %6 = call @neg(%3, %alloc_4) {fuse_with = ""} : (memref<f64>, memref<f64>) -> memref<f64>
    %7 = call @dscal(%6, %1, %alloc_5) {fuse_with = ""} : (memref<f64>, memref<128xf64>, memref<128xf64>) -> memref<128xf64>
    %8 = call @xpy(%arg2, %7, %alloc_6) {fuse_with = ""} : (memref<128xf64>, memref<128xf64>, memref<128xf64>) -> memref<128xf64>
    %9 = call @dot(%8, %8, %alloc_7) {fuse_with = ""} : (memref<128xf64>, memref<128xf64>, memref<f64>) -> memref<f64>
    %10 = call @div(%9, %0, %alloc_11) {fuse_with = ""} : (memref<f64>, memref<f64>, memref<f64>) -> memref<f64>
    %11 = call @dscal(%10, %arg3, %alloc_8) {fuse_with = ""} : (memref<f64>, memref<128xf64>, memref<128xf64>) -> memref<128xf64>
    %12 = call @xpy(%8, %11, %alloc_9) {fuse_with = ""} : (memref<128xf64>, memref<128xf64>, memref<128xf64>) -> memref<128xf64>
    return %12 : memref<128xf64>
  }
}


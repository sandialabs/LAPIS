// RUN: %lapis-opt %s --dense-linalg-to-parallel-loops | diff %s.gold -
module {
  func.func @dot(%arg0: memref<?xf64>, %arg1: memref<?xf64>) -> f64 {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<f64>
    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>], iterator_types = ["reduction"]} ins(%arg0, %arg1 : memref<?xf64>, memref<?xf64>) outs(%alloc : memref<f64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %1 = arith.mulf %in, %in_0 : f64
      %2 = arith.addf %out, %1 : f64
      linalg.yield %2 : f64
    }
    %0 = memref.load %alloc[] : memref<f64>
    return %0 : f64
  }
}

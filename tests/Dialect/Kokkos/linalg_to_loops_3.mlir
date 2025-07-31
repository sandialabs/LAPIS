// RUN: %lapis-opt %s --dense-linalg-to-parallel-loops | diff %s.gold -
module {
  func.func @gemm(%arg0: memref<4096x4096xf32>, %arg1: memref<4096x4096xf32>, %arg2: memref<4096x4096xf32>) -> memref<4096x4096xf32> {
    linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : memref<4096x4096xf32>, memref<4096x4096xf32>) outs(%arg2 : memref<4096x4096xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.mulf %in, %in_0 : f32
      %1 = arith.addf %out, %0 : f32
      linalg.yield %1 : f32
    }
    return %arg2 : memref<4096x4096xf32>
  }
}

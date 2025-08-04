// RUN: %lapis-opt %s --dense-linalg-to-parallel-loops | diff %s.gold -
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @gemv(%arg0: memref<?x?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) -> memref<?xf32> {
    linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "reduction"]} ins(%arg0, %arg1 : memref<?x?xf32>, memref<?xf32>) outs(%arg2 : memref<?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.mulf %in, %in_0 : f32 
      %1 = arith.addf %out, %0 : f32 
      linalg.yield %1 : f32 
    }   
    return %arg2 : memref<?xf32>
  }
}


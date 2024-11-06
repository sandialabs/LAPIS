#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @gemv(%arg0: tensor<?x?x?xf64>, %arg1: tensor<?x?xf64>, %arg2: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<?x?x?xf64>, tensor<?x?xf64>) outs(%arg2 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %1 = arith.mulf %in, %in_0 : f64
      %2 = arith.addf %out, %1 : f64
      linalg.yield %2 : f64
    } -> tensor<?x?xf64>
    return %0 : tensor<?x?xf64>
  }
  func.func @axpy(%arg0: tensor<?x?xf64>, %arg1: tensor<?x?xf64>, %arg2: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %0 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xf64>, tensor<?x?xf64>) outs(%arg2 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %1 = arith.addf %in, %in_0 : f64
      linalg.yield %1 : f64
    } -> tensor<?x?xf64>
    return %0 : tensor<?x?xf64>
  }
  func.func @main(%batch_size : index, %n : index) -> tensor<?x?xf64> {
    %0 = tensor.empty(%batch_size, %n, %n) : tensor<?x?x?xf64>
    %1 = tensor.empty(%batch_size, %n) : tensor<?x?xf64>
    %2 = tensor.empty(%batch_size, %n) : tensor<?x?xf64>
    %3 = tensor.empty(%batch_size, %n) : tensor<?x?xf64>
    %4 = call @gemv(%0, %1, %3) { fuse_with = "axpy" } : (tensor<?x?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) -> tensor<?x?xf64>
    %5 = tensor.empty(%batch_size, %n) : tensor<?x?xf64>
    %6 = call @axpy(%4, %2, %5) { fuse_with = "gemv" } : (tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) -> tensor<?x?xf64>
    return %6 : tensor<?x?xf64>
  }
}


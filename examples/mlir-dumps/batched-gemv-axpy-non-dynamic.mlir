#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @gemv(%arg0: tensor<10x10x10xf64>, %arg1: tensor<10x10xf64>, %arg2:
  tensor<10x10xf64>) -> tensor<10x10xf64> attributes { noinline } {
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<10x10x10xf64>, tensor<10x10xf64>) outs(%arg2 : tensor<10x10xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %1 = arith.mulf %in, %in_0 : f64
      %2 = arith.addf %out, %1 : f64
      linalg.yield %2 : f64
    } -> tensor<10x10xf64>
    return %0 : tensor<10x10xf64>
  }
  func.func @axpy(%arg0: tensor<10x10xf64>, %arg1: tensor<10x10xf64>, %arg2:
  tensor<10x10xf64>) -> tensor<10x10xf64> attributes { noinline } {
    %0 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<10x10xf64>, tensor<10x10xf64>) outs(%arg2 : tensor<10x10xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %1 = arith.addf %in, %in_0 : f64
      linalg.yield %1 : f64
    } -> tensor<10x10xf64>
    return %0 : tensor<10x10xf64>
  }
  func.func @main() -> tensor<10x10xf64> {
    %0 = tensor.empty() : tensor<10x10x10xf64>
    %1 = tensor.empty() : tensor<10x10xf64>
    %2 = tensor.empty() : tensor<10x10xf64>
    %3 = tensor.empty() : tensor<10x10xf64>
    %4 = call @gemv(%0, %1, %3) { noinline } : (tensor<10x10x10xf64>, tensor<10x10xf64>, tensor<10x10xf64>) -> tensor<10x10xf64>
    %5 = tensor.empty() : tensor<10x10xf64>
    %6 = call @axpy(%4, %2, %5) { noinline } : (tensor<10x10xf64>, tensor<10x10xf64>, tensor<10x10xf64>) -> tensor<10x10xf64>
    return %6 : tensor<10x10xf64>
  }
}

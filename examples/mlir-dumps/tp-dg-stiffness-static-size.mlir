#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d1, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d4)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>
module {
  func.func @compute_reference_dx(%arg0: tensor<100x100x100x100xf64>, %arg1: tensor<100x100xf64>, %arg2: tensor<100x100x100x100xf64>) -> tensor<100x100x100x100xf64> {
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<100x100x100x100xf64>, tensor<100x100xf64>) outs(%arg2 : tensor<100x100x100x100xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %1 = arith.mulf %in_0, %in : f64
      %2 = arith.addf %out, %1 : f64
      linalg.yield %2 : f64
    } -> tensor<100x100x100x100xf64>
    return %0 : tensor<100x100x100x100xf64>
  }

  func.func @compute_reference_dy(%arg0: tensor<100x100x100x100xf64>, %arg1: tensor<100x100xf64>, %arg2: tensor<100x100x100x100xf64>) -> tensor<100x100x100x100xf64> {
    %0 = linalg.generic {indexing_maps = [#map3, #map4, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<100x100x100x100xf64>, tensor<100x100xf64>) outs(%arg2 : tensor<100x100x100x100xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %1 = arith.mulf %in_0, %in : f64
      %2 = arith.addf %out, %1 : f64
      linalg.yield %2 : f64
    } -> tensor<100x100x100x100xf64>
    return %0 : tensor<100x100x100x100xf64>
  }

  func.func @compute_reference_dz(%arg0: tensor<100x100x100x100xf64>, %arg1: tensor<100x100xf64>, %arg2: tensor<100x100x100x100xf64>) -> tensor<100x100x100x100xf64> {
    %0 = linalg.generic {indexing_maps = [#map5, #map6, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<100x100x100x100xf64>, tensor<100x100xf64>) outs(%arg2 : tensor<100x100x100x100xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %1 = arith.mulf %in_0, %in : f64
      %2 = arith.addf %out, %1 : f64
      linalg.yield %2 : f64
    } -> tensor<100x100x100x100xf64>
    return %0 : tensor<100x100x100x100xf64>
  }

  func.func @main(%arg0: index, %arg1: index) -> (tensor<100x100x100x100xf64>,
  tensor<100x100x100x100xf64>, tensor<100x100x100x100xf64>) attributes {llvm.emit_c_interface} {
    %0 = tensor.empty() : tensor<100x100x100x100xf64>
    %1 = tensor.empty() : tensor<100x100xf64>
    %2 = tensor.empty() : tensor<100x100x100x100xf64>
    %3 = tensor.empty() : tensor<100x100x100x100xf64>
    %4 = tensor.empty() : tensor<100x100x100x100xf64>

    %5 = call @compute_reference_dx(%0, %1, %2) { fuse_with =
    "compute_reference_dy, compute_reference_dz" } : (tensor<100x100x100x100xf64>, tensor<100x100xf64>, tensor<100x100x100x100xf64>) -> tensor<100x100x100x100xf64>
    %6 = call @compute_reference_dy(%0, %1, %3) { fuse_with =
    "compute_reference_dx, compute_reference_dz" } : (tensor<100x100x100x100xf64>, tensor<100x100xf64>, tensor<100x100x100x100xf64>) -> tensor<100x100x100x100xf64>
    %7 = call @compute_reference_dz(%0, %1, %4) { fuse_with =
    "compute_reference_dx, compute_reference_dy" }: (tensor<100x100x100x100xf64>, tensor<100x100xf64>, tensor<100x100x100x100xf64>) -> tensor<100x100x100x100xf64>

    return %5, %6, %7 :
      tensor<100x100x100x100xf64>, tensor<100x100x100x100xf64>, tensor<100x100x100x100xf64>
  }
}


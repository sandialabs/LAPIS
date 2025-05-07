#map = affine_map<(i, j, k) -> (i, k)>
#map1 = affine_map<(i, j, k) -> (k, j)>
#map2 = affine_map<(i, j, k) -> (i, j)>

#map3 = affine_map<(i, j) -> (i, j)>
#map4 = affine_map<(i, j) -> (j)>
#map5 = affine_map<(i, j) -> (i)>

module {
  func.func @mmv(%a: tensor<32x10xf64>,
                 %b: tensor<10x17xf64>,
                 %x: tensor<17xf64>,
                 %y_out: tensor<32xf64>) -> tensor<32xf64>
  {
    %c_init = tensor.empty() : tensor<32x17xf64>
    %c = linalg.generic {
      indexing_maps = [#map, #map1, #map2],
      iterator_types = ["parallel", "parallel", "reduction"]
    }
      ins(%a, %b: tensor<32x10xf64>, tensor<10x17xf64>)
      outs(%c_init: tensor<32x17xf64>)
    {
      ^bb0(%in: f64, %in_0: f64, %out: f64):
        %1 = arith.mulf %in, %in_0 : f64
        %2 = arith.addf %out, %1 : f64
        linalg.yield %2 : f64
    } -> tensor<32x17xf64>

    %y = linalg.generic {
      indexing_maps = [#map3, #map4, #map5],
      iterator_types = ["parallel", "reduction"]
    } ins(%c, %x : tensor<32x17xf64>, tensor<17xf64>)
      outs(%y_out: tensor<32xf64>)
    {
      ^bb0(%cij : f64, %xj : f64, %yi : f64):
        %1 = arith.mulf %cij, %xj : f64
        %2 = arith.addf %yi, %1 : f64
        linalg.yield %2 : f64
    } -> tensor<32xf64>

    return %y : tensor<32xf64>
  }

  func.func @main() -> tensor<32xf64> {
    %a = tensor.empty() : tensor<32x10xf64>
    %b = tensor.empty() : tensor<10x17xf64>
    %x = tensor.empty() : tensor<17xf64>
    %y = tensor.empty() : tensor<32xf64>

    %y_out = call @mmv(%a, %b, %x, %y) : (tensor<32x10xf64>, tensor<10x17xf64>,
    tensor<17xf64>, tensor<32xf64>) -> tensor<32xf64>

    return %y_out : tensor<32xf64>
  }
}

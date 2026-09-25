module {
  // S=512, T=2048, D=64, E=64: 134,217,728 -> 10,485,760 multiplies.
  func.func @linear_attention(
      %q: tensor<512x64xf32>, %k: tensor<2048x64xf32>,
      %v: tensor<2048x64xf32>) -> tensor<512x64xf32> {
    %score_zero = arith.constant dense<0.0> : tensor<512x2048xf32>
    %scores = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%q, %k : tensor<512x64xf32>, tensor<2048x64xf32>)
      outs(%score_zero : tensor<512x2048xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<512x2048xf32>

    %output_zero = arith.constant dense<0.0> : tensor<512x64xf32>
    %output = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d2, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%scores, %v : tensor<512x2048xf32>, tensor<2048x64xf32>)
      outs(%output_zero : tensor<512x64xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<512x64xf32>
    return %output : tensor<512x64xf32>
  }
}

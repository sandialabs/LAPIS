module {
  // B=8, S=256, T=1024, D=64, E=64:
  // 268,435,456 -> 41,943,040 multiplications.
  func.func @batched_linear_attention(
      %q: tensor<8x256x64xf32>, %k: tensor<8x1024x64xf32>,
      %v: tensor<8x1024x64xf32>) -> tensor<8x256x64xf32> {
    %score_zero = arith.constant dense<0.0> : tensor<8x256x1024xf32>
    %scores = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
    } ins(%q, %k : tensor<8x256x64xf32>, tensor<8x1024x64xf32>)
      outs(%score_zero : tensor<8x256x1024xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<8x256x1024xf32>

    %output_zero = arith.constant dense<0.0> : tensor<8x256x64xf32>
    %output = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>],
      iterator_types = ["parallel", "parallel", "reduction", "parallel"]
    } ins(%scores, %v : tensor<8x256x1024xf32>, tensor<8x1024x64xf32>)
      outs(%output_zero : tensor<8x256x64xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<8x256x64xf32>
    return %output : tensor<8x256x64xf32>
  }
}

module {
  // Unnormalized linear-attention core with S=256, T=1024, D=64, E=64.
  // Source association: (Q K^T) V.
  //   source work    = S*T*D + S*T*E = 33,554,432 multiplications
  //   reordered work = T*D*E + S*D*E =  5,242,880 multiplications
  func.func @linear_attention(
      %q: tensor<256x64xf32>, %k: tensor<1024x64xf32>,
      %v: tensor<1024x64xf32>) -> tensor<256x64xf32> {
    %score_zero = arith.constant dense<0.0> : tensor<256x1024xf32>
    %scores = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%q, %k : tensor<256x64xf32>, tensor<1024x64xf32>)
      outs(%score_zero : tensor<256x1024xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<256x1024xf32>

    %output_zero = arith.constant dense<0.0> : tensor<256x64xf32>
    %output = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d2, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%scores, %v : tensor<256x1024xf32>, tensor<1024x64xf32>)
      outs(%output_zero : tensor<256x64xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<256x64xf32>
    return %output : tensor<256x64xf32>
  }
}

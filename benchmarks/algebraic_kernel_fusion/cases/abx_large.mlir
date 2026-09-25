module {
  // M=256, K=512, N=2048: 268,959,744 -> 1,179,648 multiplications.
  func.func @abx(%a: tensor<256x512xf32>, %b: tensor<512x2048xf32>,
                 %x: tensor<2048xf32>) -> tensor<256xf32> {
    %matrix_zero = arith.constant dense<0.0> : tensor<256x2048xf32>
    %ab = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d2, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%a, %b : tensor<256x512xf32>, tensor<512x2048xf32>)
      outs(%matrix_zero : tensor<256x2048xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<256x2048xf32>

    %vector_zero = arith.constant dense<0.0> : tensor<256xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%ab, %x : tensor<256x2048xf32>, tensor<2048xf32>)
      outs(%vector_zero : tensor<256xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<256xf32>
    return %result : tensor<256xf32>
  }
}

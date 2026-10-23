module {
  // Source association: (A B) x.
  // M=128, K=256, N=1024:
  //   source work    = M*K*N + M*N = 33,685,504 multiplications
  //   reordered work = K*N + M*K   =    294,912 multiplications
  func.func @abx(%a: tensor<128x256xf32>, %b: tensor<256x1024xf32>,
                 %x: tensor<1024xf32>) -> tensor<128xf32> {
    %matrix_zero = arith.constant dense<0.0> : tensor<128x1024xf32>
    %ab = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d2, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%a, %b : tensor<128x256xf32>, tensor<256x1024xf32>)
      outs(%matrix_zero : tensor<128x1024xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<128x1024xf32>

    %vector_zero = arith.constant dense<0.0> : tensor<128xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%ab, %x : tensor<128x1024xf32>, tensor<1024xf32>)
      outs(%vector_zero : tensor<128xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<128xf32>
    return %result : tensor<128xf32>
  }
}

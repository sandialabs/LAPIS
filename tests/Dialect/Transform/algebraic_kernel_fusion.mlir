// RUN: %lapis-opt %s --algebraic-kernel-fusion | diff -B %s.gold -
// RUN: %lapis-opt %s '--sparse-compiler-kokkos=parallelization-strategy=any-storage-any-loop decompose-sparse-tensors algebraic-kernel-fusion' | grep lapis.algebraic_kernel > /dev/null

// ABx exercises extraction, region composition, exact reassociation, plan
// materialization, fusion-set marking, and the integrated Kokkos pipeline.
module {
  func.func @abx(%a: tensor<2x3xf32>, %b: tensor<3x5xf32>,
                 %x: tensor<5xf32>) -> tensor<2xf32> {
    %matrix_zero = arith.constant dense<0.0> : tensor<2x5xf32>
    %ab = linalg.generic {
      indexing_maps = [affine_map<(i, j, k) -> (i, k)>,
                       affine_map<(i, j, k) -> (k, j)>,
                       affine_map<(i, j, k) -> (i, j)>],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%a, %b : tensor<2x3xf32>, tensor<3x5xf32>)
      outs(%matrix_zero : tensor<2x5xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<2x5xf32>

    %vector_zero = arith.constant dense<0.0> : tensor<2xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (j)>,
                       affine_map<(i, j) -> (i)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%ab, %x : tensor<2x5xf32>, tensor<5xf32>)
      outs(%vector_zero : tensor<2xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<2xf32>
    return %result : tensor<2xf32>
  }

  // These pointwise operations have the same iteration space expressed in
  // opposite orders. They are deliberately not contractions: this stresses
  // fusion-set selection and semantic iterator alignment independently of the
  // einsum planner.
  func.func @permuted_pointwise(%input: tensor<2x3xf32>)
      -> tensor<3x2xf32> {
    %producer_zero = arith.constant dense<0.0> : tensor<2x3xf32>
    %producer = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%input : tensor<2x3xf32>)
      outs(%producer_zero : tensor<2x3xf32>) {
    ^bb0(%value: f32, %unused: f32):
      linalg.yield %value : f32
    } -> tensor<2x3xf32>

    %consumer_zero = arith.constant dense<0.0> : tensor<3x2xf32>
    %consumer = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (j, i)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%producer : tensor<2x3xf32>)
      outs(%consumer_zero : tensor<3x2xf32>) {
    ^bb0(%value: f32, %unused: f32):
      linalg.yield %value : f32
    } -> tensor<3x2xf32>
    return %consumer : tensor<3x2xf32>
  }
}

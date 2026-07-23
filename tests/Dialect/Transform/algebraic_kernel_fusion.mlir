// RUN: %lapis-opt %s --algebraic-kernel-fusion | diff -B %s.gold -
// RUN: %lapis-opt %s '--sparse-compiler-kokkos=parallelization-strategy=any-storage-any-loop decompose-sparse-tensors algebraic-kernel-fusion' > %t.kokkos
// RUN: grep lapis.algebraic_kernel %t.kokkos > /dev/null
// RUN: sed -n '/func.func private @fan_in__lapis_algebraic_kernel/,/func.func @fan_in/p' %t.kokkos > %t.fan_in
// RUN: grep -c kokkos.range_parallel %t.fan_in | grep '^1$'
// RUN: ! grep memref.alloc %t.fan_in
// RUN: sed -n '/func.func private @storage_choice__lapis_algebraic_kernel/,/func.func @storage_choice/p' %t.kokkos > %t.storage_choice
// RUN: grep -c kokkos.range_parallel %t.storage_choice | grep '^1$'
// RUN: ! grep memref.alloc %t.storage_choice
// RUN: %lapis-opt %S/../../../benchmarks/algebraic_kernel_fusion/cases/pcg.mlir --algebraic-kernel-fusion > %t.pcg
// RUN: grep -c lapis.algebraic_fusion_group %t.pcg | grep '^3$'
// RUN: grep 'iterator_types = \["reduction"\].*lapis.algebraic_fusion_group' %t.pcg > /dev/null
// RUN: ! grep 'iterator_types = \["parallel", "reduction"\].*lapis.algebraic_fusion_group' %t.pcg
// RUN: %lapis-opt %S/../../../benchmarks/algebraic_kernel_fusion/cases/burgers.mlir '--sparse-compiler-kokkos=parallelization-strategy=any-storage-any-loop decompose-sparse-tensors algebraic-kernel-fusion' > %t.burgers
// RUN: sed -n '/func.func private @burgers__lapis_algebraic_kernel/,/func.func @burgers/p' %t.burgers > %t.burgers_kernels
// RUN: grep -c kokkos.range_parallel %t.burgers_kernels | grep '^2$'
// RUN: ! grep memref.alloc %t.burgers_kernels

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

  // Three sibling producers share an input and meet at one consumer. No
  // producer-prefix is connected, so this specifically exercises complete-set
  // fan-in selection rather than chain fusion.
  func.func @fan_in(%input: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %empty0 = tensor.empty() : tensor<2x3xf32>
    %twice = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%input : tensor<2x3xf32>)
      outs(%empty0 : tensor<2x3xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %doubled = arith.addf %value, %value : f32
      linalg.yield %doubled : f32
    } -> tensor<2x3xf32>

    %empty1 = tensor.empty() : tensor<2x3xf32>
    %negated = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%input : tensor<2x3xf32>)
      outs(%empty1 : tensor<2x3xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %negation = arith.negf %value : f32
      linalg.yield %negation : f32
    } -> tensor<2x3xf32>

    %empty2 = tensor.empty() : tensor<2x3xf32>
    %squared = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%input : tensor<2x3xf32>)
      outs(%empty2 : tensor<2x3xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %square = arith.mulf %value, %value : f32
      linalg.yield %square : f32
    } -> tensor<2x3xf32>

    %result_empty = tensor.empty() : tensor<2x3xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%twice, %negated, %squared
          : tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>)
      outs(%result_empty : tensor<2x3xf32>) {
    ^bb0(%positive: f32, %negative: f32, %square: f32, %unused: f32):
      %cancelled = arith.addf %positive, %negative : f32
      %sum = arith.addf %cancelled, %square : f32
      linalg.yield %sum : f32
    } -> tensor<2x3xf32>
    return %result : tensor<2x3xf32>
  }

  // Either adjacent boundary can be removed, but the complete chain has no
  // common fusion domain. Prefer eliminating the 4x8 expanded tensor over the
  // four-element row reduction.
  func.func @storage_choice(%input: tensor<4x8xf32>) -> tensor<f32> {
    %row_zero = arith.constant dense<0.0> : tensor<4xf32>
    %row_sum = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%input : tensor<4x8xf32>) outs(%row_zero : tensor<4xf32>) {
    ^bb0(%value: f32, %accumulator: f32):
      %sum = arith.addf %accumulator, %value : f32
      linalg.yield %sum : f32
    } -> tensor<4xf32>

    %expanded_empty = tensor.empty() : tensor<4x8xf32>
    %expanded = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%row_sum : tensor<4xf32>)
      outs(%expanded_empty : tensor<4x8xf32>) {
    ^bb0(%value: f32, %unused: f32):
      linalg.yield %value : f32
    } -> tensor<4x8xf32>

    %scalar_zero = arith.constant dense<0.0> : tensor<f32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> ()>],
      iterator_types = ["reduction", "reduction"]
    } ins(%expanded : tensor<4x8xf32>) outs(%scalar_zero : tensor<f32>) {
    ^bb0(%value: f32, %accumulator: f32):
      %sum = arith.addf %accumulator, %value : f32
      linalg.yield %sum : f32
    } -> tensor<f32>
    return %result : tensor<f32>
  }
}

// RUN: %lapis-opt %s --algebraic-kernel-fusion | diff -B %s.gold -

module {
  // The denominator-only form is a legal einsum region. Both associations
  // perform 8 * 8 + 8 scalar multiplications, so the primary work objective
  // ties and the secondary algebraic-kernel-count objective selects fusion.
  func.func @pcg_denominator_only(%a: tensor<8x8xf64>,
                                  %p: tensor<8xf64>) -> tensor<f64> {
    %ap_zero = arith.constant dense<0.0> : tensor<8xf64>
    %ap = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%a, %p : tensor<8x8xf64>, tensor<8xf64>)
      outs(%ap_zero : tensor<8xf64>) {
    ^bb0(%lhs: f64, %rhs: f64, %acc: f64):
      %product = arith.mulf %lhs, %rhs : f64
      %sum = arith.addf %acc, %product : f64
      linalg.yield %sum : f64
    } -> tensor<8xf64>

    %pap_zero = arith.constant dense<0.0> : tensor<f64>
    %pap = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]
    } ins(%p, %ap : tensor<8xf64>, tensor<8xf64>)
      outs(%pap_zero : tensor<f64>) {
    ^bb0(%lhs: f64, %rhs: f64, %acc: f64):
      %product = arith.mulf %lhs, %rhs : f64
      %sum = arith.addf %acc, %product : f64
      linalg.yield %sum : f64
    } -> tensor<f64>
    return %pap : tensor<f64>
  }

  // This mirrors the paper's fused SpMV+dot kernel interface: PCG needs both
  // Ap and p^T Ap. Discovery represents Ap as a required region output. The
  // constrained source and optimized plans both perform 72 multiplications,
  // then the secondary algebraic-kernel-count objective selects fusion.
  func.func @pcg_denominator_and_ap(%a: tensor<8x8xf64>,
                                    %p: tensor<8xf64>)
      -> (tensor<8xf64>, tensor<f64>) {
    %ap_zero = arith.constant dense<0.0> : tensor<8xf64>
    %ap = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%a, %p : tensor<8x8xf64>, tensor<8xf64>)
      outs(%ap_zero : tensor<8xf64>) {
    ^bb0(%lhs: f64, %rhs: f64, %acc: f64):
      %product = arith.mulf %lhs, %rhs : f64
      %sum = arith.addf %acc, %product : f64
      linalg.yield %sum : f64
    } -> tensor<8xf64>

    %pap_zero = arith.constant dense<0.0> : tensor<f64>
    %pap = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]
    } ins(%p, %ap : tensor<8xf64>, tensor<8xf64>)
      outs(%pap_zero : tensor<f64>) {
    ^bb0(%lhs: f64, %rhs: f64, %acc: f64):
      %product = arith.mulf %lhs, %rhs : f64
      %sum = arith.addf %acc, %product : f64
      linalg.yield %sum : f64
    } -> tensor<f64>
    return %ap, %pap : tensor<8xf64>, tensor<f64>
  }
}

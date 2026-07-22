// RUN: %lapis-opt %s --split-input-file --algebraic-kernel-fusion | diff -B %s.gold -

// ABx is reassociated from (AB)x to A(Bx), reducing the intermediate from
// 2x5 to 3 elements. Both replacement contractions form one fusion group.
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
}

// -----

// PCG needs both Ap and p^T Ap. The pass preserves Ap as an externally visible
// result while selecting the equal-work two-operation region for fusion.
module {
  func.func @pcg_denominator_and_ap(%a: tensor<8x8xf64>,
                                    %p: tensor<8xf64>)
      -> (tensor<8xf64>, tensor<f64>) {
    %ap_zero = arith.constant dense<0.0> : tensor<8xf64>
    %ap = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (j)>,
                       affine_map<(i, j) -> (i)>],
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
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>,
                       affine_map<(i) -> ()>],
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

// -----

// The ordered selector chooses map-map-reduce and the independent two-map
// chain as distinct fusion groups.
module {
  func.func @fusion_partition(
      %a: tensor<4xf32>, %b: tensor<4xf32>,
      %c: tensor<4xf32>, %d: tensor<4xf32>)
      -> (tensor<f32>, tensor<4xf32>) {
    %zero0 = arith.constant dense<0.0> : tensor<4xf32>
    %lhs = linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>,
                       affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]
    } ins(%a, %b : tensor<4xf32>, tensor<4xf32>)
      outs(%zero0 : tensor<4xf32>) {
    ^bb0(%x: f32, %y: f32, %unused: f32):
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
    } -> tensor<4xf32>

    %zero1 = arith.constant dense<0.0> : tensor<4xf32>
    %rhs = linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>,
                       affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]
    } ins(%c, %d : tensor<4xf32>, tensor<4xf32>)
      outs(%zero1 : tensor<4xf32>) {
    ^bb0(%x: f32, %y: f32, %unused: f32):
      %difference = arith.subf %x, %y : f32
      linalg.yield %difference : f32
    } -> tensor<4xf32>

    %scalar_zero = arith.constant dense<0.0> : tensor<f32>
    %dot = linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>,
                       affine_map<(i) -> ()>],
      iterator_types = ["reduction"]
    } ins(%lhs, %rhs : tensor<4xf32>, tensor<4xf32>)
      outs(%scalar_zero : tensor<f32>) {
    ^bb0(%x: f32, %y: f32, %acc: f32):
      %product = arith.mulf %x, %y : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<f32>

    %zero2 = arith.constant dense<0.0> : tensor<4xf32>
    %tail0 = linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>,
                       affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]
    } ins(%a, %d : tensor<4xf32>, tensor<4xf32>)
      outs(%zero2 : tensor<4xf32>) {
    ^bb0(%x: f32, %y: f32, %unused: f32):
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
    } -> tensor<4xf32>

    %zero3 = arith.constant dense<0.0> : tensor<4xf32>
    %tail1 = linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>,
                       affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]
    } ins(%tail0, %b : tensor<4xf32>, tensor<4xf32>)
      outs(%zero3 : tensor<4xf32>) {
    ^bb0(%x: f32, %y: f32, %unused: f32):
      %difference = arith.subf %x, %y : f32
      linalg.yield %difference : f32
    } -> tensor<4xf32>
    return %dot, %tail1 : tensor<f32>, tensor<4xf32>
  }
}

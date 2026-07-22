// RUN: %lapis-opt %s --algebraic-kernel-fusion | diff -B %s.gold -

module {
  func.func @abx(%a: tensor<2x3xf32>, %b: tensor<3x5xf32>,
                 %x: tensor<5xf32>) -> tensor<2xf32> {
    %matrix_zero = arith.constant dense<0.0> : tensor<2x5xf32>
    %ab = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d2, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
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
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d1)>,
                       affine_map<(d0, d1) -> (d0)>],
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

  // The middle ABx result escapes and feeds a final dot product. Preserving
  // that output constrains the tree to contain the {A, B, x} subset, while
  // still permitting ABx itself to be reassociated from (AB)x to A(Bx).
  func.func @profitable_multi_output(
      %a: tensor<2x3xf32>, %b: tensor<3x5xf32>, %x: tensor<5xf32>,
      %z: tensor<2xf32>) -> (tensor<2xf32>, tensor<f32>) {
    %zero0 = arith.constant dense<0.0> : tensor<2x5xf32>
    %ab = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d2, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%a, %b : tensor<2x3xf32>, tensor<3x5xf32>)
      outs(%zero0 : tensor<2x5xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<2x5xf32>

    %zero1 = arith.constant dense<0.0> : tensor<2xf32>
    %abx = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%ab, %x : tensor<2x5xf32>, tensor<5xf32>)
      outs(%zero1 : tensor<2xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<2xf32>

    %zero2 = arith.constant dense<0.0> : tensor<f32>
    %dot = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]
    } ins(%abx, %z : tensor<2xf32>, tensor<2xf32>)
      outs(%zero2 : tensor<f32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<f32>

    return %abx, %dot : tensor<2xf32>, tensor<f32>
  }

  // The escaping y result no longer splits this into two regions. It becomes
  // a required output of one larger tree while both profitable subtrees are
  // still reassociated.
  func.func @dependent_profitable_regions(
      %a: tensor<2x3xf32>, %b: tensor<3x5xf32>, %x: tensor<5xf32>,
      %d: tensor<2x5xf32>, %e: tensor<5xf32>)
      -> (tensor<2xf32>, tensor<f32>) {
    %zero0 = arith.constant dense<0.0> : tensor<2x5xf32>
    %ab = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d2, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%a, %b : tensor<2x3xf32>, tensor<3x5xf32>)
      outs(%zero0 : tensor<2x5xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<2x5xf32>

    %zero1 = arith.constant dense<0.0> : tensor<2xf32>
    %y = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%ab, %x : tensor<2x5xf32>, tensor<5xf32>)
      outs(%zero1 : tensor<2xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<2xf32>

    %zero2 = arith.constant dense<0.0> : tensor<5xf32>
    %yd = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d1)>],
      iterator_types = ["reduction", "parallel"]
    } ins(%y, %d : tensor<2xf32>, tensor<2x5xf32>)
      outs(%zero2 : tensor<5xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<5xf32>

    %zero3 = arith.constant dense<0.0> : tensor<f32>
    %denominator = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]
    } ins(%yd, %e : tensor<5xf32>, tensor<5xf32>)
      outs(%zero3 : tensor<f32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<f32>

    return %y, %denominator : tensor<2xf32>, tensor<f32>
  }

  // Every pointwise association has equal work. The primary objective ties,
  // so the secondary algebraic-kernel-count objective selects the region; the
  // deterministic planner tie-break still determines its materialized order.
  func.func @equal_work(%a: tensor<8xf32>, %b: tensor<8xf32>,
                        %c: tensor<8xf32>) -> tensor<8xf32> {
    %zero0 = arith.constant dense<0.0> : tensor<8xf32>
    %ab = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%a, %b : tensor<8xf32>, tensor<8xf32>)
      outs(%zero0 : tensor<8xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<8xf32>

    %zero1 = arith.constant dense<0.0> : tensor<8xf32>
    %abc = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%ab, %c : tensor<8xf32>, tensor<8xf32>)
      outs(%zero1 : tensor<8xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<8xf32>
    return %abc : tensor<8xf32>
  }

  // Initializers do not impose artificial fusion cuts. The exact ordered
  // partition selects map-map-reduce as one terminal-reduction group, then
  // selects the two remaining pointwise operations as a second group.
  func.func @fusion_partition(
      %a: tensor<4xf32>, %b: tensor<4xf32>,
      %c: tensor<4xf32>, %d: tensor<4xf32>)
      -> (tensor<f32>, tensor<4xf32>) {
    %zero0 = arith.constant dense<0.0> : tensor<4xf32>
    %lhs = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%a, %b : tensor<4xf32>, tensor<4xf32>)
      outs(%zero0 : tensor<4xf32>) {
    ^bb0(%x: f32, %y: f32, %unused: f32):
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
    } -> tensor<4xf32>

    %zero1 = arith.constant dense<0.0> : tensor<4xf32>
    %rhs = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%c, %d : tensor<4xf32>, tensor<4xf32>)
      outs(%zero1 : tensor<4xf32>) {
    ^bb0(%x: f32, %y: f32, %unused: f32):
      %difference = arith.subf %x, %y : f32
      linalg.yield %difference : f32
    } -> tensor<4xf32>

    %scalar_zero = arith.constant dense<0.0> : tensor<f32>
    %dot = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> ()>],
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
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%a, %d : tensor<4xf32>, tensor<4xf32>)
      outs(%zero2 : tensor<4xf32>) {
    ^bb0(%x: f32, %y: f32, %unused: f32):
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
    } -> tensor<4xf32>

    %zero3 = arith.constant dense<0.0> : tensor<4xf32>
    %tail1 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
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

module {
  // One dense preconditioned conjugate-gradient iteration at N=512. Ap and
  // znext remain externally visible while also feeding dot products, stressing
  // multi-output planning and equal-work fusion selection.
  func.func @pcg_step(
      %a: tensor<512x512xf64>, %p: tensor<512xf64>, %r: tensor<512xf64>,
      %z: tensor<512xf64>, %x: tensor<512xf64>, %dinv: tensor<512xf64>)
      -> (tensor<512xf64>, tensor<512xf64>, tensor<512xf64>,
          tensor<512xf64>, tensor<512xf64>, tensor<f64>, tensor<f64>) {
    %vector_zero = arith.constant dense<0.0> : tensor<512xf64>
    %scalar_zero = arith.constant dense<0.0> : tensor<f64>

    %rz = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]
    } ins(%r, %z : tensor<512xf64>, tensor<512xf64>)
      outs(%scalar_zero : tensor<f64>) {
    ^bb0(%lhs: f64, %rhs: f64, %acc: f64):
      %product = arith.mulf %lhs, %rhs : f64
      %sum = arith.addf %acc, %product : f64
      linalg.yield %sum : f64
    } -> tensor<f64>

    %ap = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%a, %p : tensor<512x512xf64>, tensor<512xf64>)
      outs(%vector_zero : tensor<512xf64>) {
    ^bb0(%lhs: f64, %rhs: f64, %acc: f64):
      %product = arith.mulf %lhs, %rhs : f64
      %sum = arith.addf %acc, %product : f64
      linalg.yield %sum : f64
    } -> tensor<512xf64>

    %pap = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]
    } ins(%p, %ap : tensor<512xf64>, tensor<512xf64>)
      outs(%scalar_zero : tensor<f64>) {
    ^bb0(%lhs: f64, %rhs: f64, %acc: f64):
      %product = arith.mulf %lhs, %rhs : f64
      %sum = arith.addf %acc, %product : f64
      linalg.yield %sum : f64
    } -> tensor<f64>

    %rz_value = tensor.extract %rz[] : tensor<f64>
    %pap_value = tensor.extract %pap[] : tensor<f64>
    %alpha = arith.divf %rz_value, %pap_value : f64

    %xnext = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%x, %p : tensor<512xf64>, tensor<512xf64>)
      outs(%vector_zero : tensor<512xf64>) {
    ^bb0(%x_value: f64, %p_value: f64, %unused: f64):
      %scaled = arith.mulf %alpha, %p_value : f64
      %updated = arith.addf %x_value, %scaled : f64
      linalg.yield %updated : f64
    } -> tensor<512xf64>

    %rnext = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%r, %ap : tensor<512xf64>, tensor<512xf64>)
      outs(%vector_zero : tensor<512xf64>) {
    ^bb0(%r_value: f64, %ap_value: f64, %unused: f64):
      %scaled = arith.mulf %alpha, %ap_value : f64
      %updated = arith.subf %r_value, %scaled : f64
      linalg.yield %updated : f64
    } -> tensor<512xf64>

    %znext = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%rnext, %dinv : tensor<512xf64>, tensor<512xf64>)
      outs(%vector_zero : tensor<512xf64>) {
    ^bb0(%lhs: f64, %rhs: f64, %acc: f64):
      %product = arith.mulf %lhs, %rhs : f64
      %sum = arith.addf %acc, %product : f64
      linalg.yield %sum : f64
    } -> tensor<512xf64>

    %rznext = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]
    } ins(%rnext, %znext : tensor<512xf64>, tensor<512xf64>)
      outs(%scalar_zero : tensor<f64>) {
    ^bb0(%lhs: f64, %rhs: f64, %acc: f64):
      %product = arith.mulf %lhs, %rhs : f64
      %sum = arith.addf %acc, %product : f64
      linalg.yield %sum : f64
    } -> tensor<f64>

    %rznext_value = tensor.extract %rznext[] : tensor<f64>
    %beta = arith.divf %rznext_value, %rz_value : f64
    %pnext = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%znext, %p : tensor<512xf64>, tensor<512xf64>)
      outs(%vector_zero : tensor<512xf64>) {
    ^bb0(%z_value: f64, %p_value: f64, %unused: f64):
      %scaled = arith.mulf %beta, %p_value : f64
      %updated = arith.addf %z_value, %scaled : f64
      linalg.yield %updated : f64
    } -> tensor<512xf64>

    return %xnext, %rnext, %pnext, %znext, %ap, %pap, %rznext
        : tensor<512xf64>, tensor<512xf64>, tensor<512xf64>,
          tensor<512xf64>, tensor<512xf64>, tensor<f64>, tensor<f64>
  }
}

#pointwise = affine_map<(d0, d1) -> (d0, d1)>

module {
  // One forward-Euler stage for the two-dimensional scalar viscous Burgers
  // equation
  //
  //   u_t = -u (u_x + u_y) + nu (u_xx + u_yy).
  //
  // The tensor contains the 128x128 interior grid. Padding supplies homogeneous
  // Dirichlet boundary values. The three derivative primitives deliberately
  // remain separate so the fusion pass must discover their fan-in at the RHS.
  func.func private @burgers_euler_stage(
      %u: tensor<128x128xf64>, %destination: tensor<128x128xf64>)
      -> tensor<128x128xf64> {
    %zero = arith.constant 0.0 : f64
    %inv_two_dx = arith.constant 64.5 : f64
    %inv_dx_squared = arith.constant 16641.0 : f64
    %viscosity = arith.constant 0.01 : f64
    %dt = arith.constant 0.0005 : f64
    %minus_four = arith.constant -4.0 : f64

    %padded = tensor.pad %u low[1, 1] high[1, 1] {
    ^bb0(%row: index, %column: index):
      tensor.yield %zero : f64
    } : tensor<128x128xf64> to tensor<130x130xf64>

    %north = tensor.extract_slice %padded[0, 1] [128, 128] [1, 1]
        : tensor<130x130xf64> to tensor<128x128xf64>
    %west = tensor.extract_slice %padded[1, 0] [128, 128] [1, 1]
        : tensor<130x130xf64> to tensor<128x128xf64>
    %center = tensor.extract_slice %padded[1, 1] [128, 128] [1, 1]
        : tensor<130x130xf64> to tensor<128x128xf64>
    %east = tensor.extract_slice %padded[1, 2] [128, 128] [1, 1]
        : tensor<130x130xf64> to tensor<128x128xf64>
    %south = tensor.extract_slice %padded[2, 1] [128, 128] [1, 1]
        : tensor<130x130xf64> to tensor<128x128xf64>

    %ux_empty = tensor.empty() : tensor<128x128xf64>
    %ux = linalg.generic {
      indexing_maps = [#pointwise, #pointwise, #pointwise],
      iterator_types = ["parallel", "parallel"]
    } ins(%west, %east : tensor<128x128xf64>, tensor<128x128xf64>)
      outs(%ux_empty : tensor<128x128xf64>) {
    ^bb0(%west_value: f64, %east_value: f64, %unused: f64):
      %difference = arith.subf %east_value, %west_value : f64
      %derivative = arith.mulf %difference, %inv_two_dx : f64
      linalg.yield %derivative : f64
    } -> tensor<128x128xf64>

    %uy_empty = tensor.empty() : tensor<128x128xf64>
    %uy = linalg.generic {
      indexing_maps = [#pointwise, #pointwise, #pointwise],
      iterator_types = ["parallel", "parallel"]
    } ins(%north, %south : tensor<128x128xf64>, tensor<128x128xf64>)
      outs(%uy_empty : tensor<128x128xf64>) {
    ^bb0(%north_value: f64, %south_value: f64, %unused: f64):
      %difference = arith.subf %south_value, %north_value : f64
      %derivative = arith.mulf %difference, %inv_two_dx : f64
      linalg.yield %derivative : f64
    } -> tensor<128x128xf64>

    %laplacian_empty = tensor.empty() : tensor<128x128xf64>
    %laplacian = linalg.generic {
      indexing_maps = [#pointwise, #pointwise, #pointwise, #pointwise,
                       #pointwise, #pointwise],
      iterator_types = ["parallel", "parallel"]
    } ins(%north, %west, %center, %east, %south
          : tensor<128x128xf64>, tensor<128x128xf64>, tensor<128x128xf64>,
            tensor<128x128xf64>, tensor<128x128xf64>)
      outs(%laplacian_empty : tensor<128x128xf64>) {
    ^bb0(%north_value: f64, %west_value: f64, %center_value: f64,
         %east_value: f64, %south_value: f64, %unused: f64):
      %horizontal = arith.addf %west_value, %east_value : f64
      %vertical = arith.addf %north_value, %south_value : f64
      %neighbors = arith.addf %horizontal, %vertical : f64
      %center_scaled = arith.mulf %minus_four, %center_value : f64
      %difference = arith.addf %neighbors, %center_scaled : f64
      %result = arith.mulf %difference, %inv_dx_squared : f64
      linalg.yield %result : f64
    } -> tensor<128x128xf64>

    %rhs_empty = tensor.empty() : tensor<128x128xf64>
    %rhs = linalg.generic {
      indexing_maps = [#pointwise, #pointwise, #pointwise, #pointwise,
                       #pointwise],
      iterator_types = ["parallel", "parallel"]
    } ins(%u, %ux, %uy, %laplacian
          : tensor<128x128xf64>, tensor<128x128xf64>, tensor<128x128xf64>,
            tensor<128x128xf64>)
      outs(%rhs_empty : tensor<128x128xf64>) {
    ^bb0(%u_value: f64, %ux_value: f64, %uy_value: f64,
         %laplacian_value: f64, %unused: f64):
      %gradient_sum = arith.addf %ux_value, %uy_value : f64
      %advection = arith.mulf %u_value, %gradient_sum : f64
      %diffusion = arith.mulf %viscosity, %laplacian_value : f64
      %result = arith.subf %diffusion, %advection : f64
      linalg.yield %result : f64
    } -> tensor<128x128xf64>

    %updated = linalg.generic {
      indexing_maps = [#pointwise, #pointwise, #pointwise],
      iterator_types = ["parallel", "parallel"]
    } ins(%u, %rhs : tensor<128x128xf64>, tensor<128x128xf64>)
      outs(%destination : tensor<128x128xf64>) {
    ^bb0(%u_value: f64, %rhs_value: f64, %unused: f64):
      %increment = arith.mulf %dt, %rhs_value : f64
      %result = arith.addf %u_value, %increment : f64
      linalg.yield %result : f64
    } -> tensor<128x128xf64>
    return %updated : tensor<128x128xf64>
  }

  // Twenty SSP-RK2 steps. Each step evaluates the three-input derivative fan-in
  // twice, then averages the original state with the second Euler stage.
  func.func @burgers(%initial: tensor<128x128xf64>)
      -> tensor<128x128xf64> {
    %begin = arith.constant 0 : index
    %end = arith.constant 20 : index
    %step = arith.constant 1 : index
    %half = arith.constant 0.5 : f64
    %current_storage = tensor.empty() : tensor<128x128xf64>
    %current_initial = linalg.copy ins(%initial : tensor<128x128xf64>)
        outs(%current_storage : tensor<128x128xf64>)
        -> tensor<128x128xf64>
    %stage1_initial = tensor.empty() : tensor<128x128xf64>
    %stage2_initial = tensor.empty() : tensor<128x128xf64>
    %result:3 = scf.for %time = %begin to %end step %step
        iter_args(%current = %current_initial,
                  %stage1_destination = %stage1_initial,
                  %stage2_destination = %stage2_initial)
        -> (tensor<128x128xf64>, tensor<128x128xf64>, tensor<128x128xf64>) {
      %stage1 = func.call @burgers_euler_stage(%current, %stage1_destination)
          : (tensor<128x128xf64>, tensor<128x128xf64>)
            -> tensor<128x128xf64>
      %stage2 = func.call @burgers_euler_stage(%stage1, %stage2_destination)
          : (tensor<128x128xf64>, tensor<128x128xf64>)
            -> tensor<128x128xf64>

      %next = linalg.generic {
        indexing_maps = [#pointwise, #pointwise, #pointwise],
        iterator_types = ["parallel", "parallel"]
      } ins(%current, %stage2 : tensor<128x128xf64>, tensor<128x128xf64>)
        outs(%current : tensor<128x128xf64>) {
      ^bb0(%current_value: f64, %stage2_value: f64, %unused: f64):
        %sum = arith.addf %current_value, %stage2_value : f64
        %average = arith.mulf %half, %sum : f64
        linalg.yield %average : f64
      } -> tensor<128x128xf64>
      scf.yield %next, %stage1, %stage2
          : tensor<128x128xf64>, tensor<128x128xf64>, tensor<128x128xf64>
    }
    return %result#0 : tensor<128x128xf64>
  }
}

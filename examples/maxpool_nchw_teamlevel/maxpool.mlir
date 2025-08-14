module attributes {torch.debug_module_name = "MaxPool2d"} {
  func.func @forward(%arg0: tensor<1x3x20x30xf32>) -> tensor<1x3x6x10xf32> {
    %cst = arith.constant 0xFF800000 : f32
    %padded = tensor.pad %arg0 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst : f32
    } : tensor<1x3x20x30xf32> to tensor<1x3x22x32xf32>
    %0 = tensor.empty() : tensor<1x3x6x10xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x3x6x10xf32>) -> tensor<1x3x6x10xf32>
    %2 = tensor.empty() : tensor<3x3xf32>
    %3 = linalg.pooling_nchw_max {dilations = dense<2> : vector<2xi64>, strides = dense<3> : vector<2xi64>} ins(%padded, %2 : tensor<1x3x22x32xf32>, tensor<3x3xf32>) outs(%1 : tensor<1x3x6x10xf32>) -> tensor<1x3x6x10xf32>
    return %3 : tensor<1x3x6x10xf32>
  }
}


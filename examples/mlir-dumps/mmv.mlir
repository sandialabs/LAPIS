module {
  func.func private @matmul(
    %a: tensor<4096x4096xf64>,
    %b: tensor<4096x4096xf64>,
    %c_out: tensor<4096x4096xf64>
  ) -> tensor<4096x4096xf64> {
    %c = linalg.matmul ins(%a, %b: tensor<4096x4096xf64>, tensor<4096x4096xf64>)
      outs(%c_out: tensor<4096x4096xf64>) -> tensor<4096x4096xf64>
    return %c : tensor<4096x4096xf64>
  }

  func.func private @matvec(
    %a: tensor<4096x4096xf64>,
    %x: tensor<4096xf64>,
    %y_out: tensor<4096xf64>
  ) -> tensor<4096xf64> {
    %y = linalg.matvec ins(%a, %x: tensor<4096x4096xf64>, tensor<4096xf64>)
      outs(%y_out: tensor<4096xf64>) -> tensor<4096xf64>

      return %y : tensor<4096xf64>
  }

  func.func @matmul_into_matvec(
    %a: tensor<4096x4096xf64>,
    %b: tensor<4096x4096xf64>,
    %x: tensor<4096xf64>
  ) -> tensor<4096xf64> {

    %c_init = tensor.empty() : tensor<4096x4096xf64>
    %c = func.call @matmul(%a, %b, %c_init) { fuse_with = "matvec" } 
      : (tensor<4096x4096xf64>, tensor<4096x4096xf64>, tensor<4096x4096xf64>) 
        -> tensor<4096x4096xf64>

    %y_init = tensor.empty() : tensor<4096xf64>
    %y_out = func.call @matvec(%c, %x, %y_init) { fuse_with = "matmul" } 
      : (tensor<4096x4096xf64>, tensor<4096xf64>, tensor<4096xf64>) 
        -> tensor<4096xf64>

    return %y_out : tensor<4096xf64>
  }

  func.func @matvec_into_matvec(
    %a: tensor<4096x4096xf64>,
    %b: tensor<4096x4096xf64>,
    %x: tensor<4096xf64>
  ) -> tensor<4096xf64> {
    %bx_init = tensor.empty() : tensor<4096xf64>
    %bx = func.call @matvec(%b, %x, %bx_init) { fuse_with = "matvec" } 
      : (tensor<4096x4096xf64>, tensor<4096xf64>, tensor<4096xf64>) 
        -> tensor<4096xf64>

    %y_init = tensor.empty() : tensor<4096xf64>
    %y_out = func.call @matvec(%a, %bx, %y_init) { fuse_with = "matvec" }
      : (tensor<4096x4096xf64>, tensor<4096xf64>, tensor<4096xf64>) 
        -> tensor<4096xf64>

    return %y_out : tensor<4096xf64>
  }
}

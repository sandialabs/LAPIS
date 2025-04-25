#sparse = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed), posWidth = 32, crdWidth = 32 }>
module {
  func.func @spmv(%A: tensor<?x?xf64, #sparse>, %x: tensor<?xf64>, %y: tensor<?xf64>) -> tensor<?xf64> {
    %0 = linalg.matvec ins(%A, %x: tensor<?x?xf64, #sparse>, tensor<?xf64>) outs(%y : tensor<?xf64>) -> tensor<?xf64>
    return %0 : tensor<?xf64>
  }
}


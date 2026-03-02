#sparse = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed), posWidth = 32, crdWidth = 32 }>

module {
  func.func private @spmv(%A: tensor<?x?xf64, #sparse>, %x: tensor<?xf64>, %ydst: tensor<?xf64>) -> tensor<?xf64> {
    %y = linalg.matvec ins(%A, %x: tensor<?x?xf64, #sparse>, tensor<?xf64>) outs(%ydst : tensor<?xf64>) -> tensor<?xf64>
    return %y : tensor<?xf64>
  }

  func.func @dot(%x : tensor<?xf64>, %y : tensor<?xf64>, %res : tensor<f64>) ->
  tensor<f64> attributes { noinline } {
    %dot = linalg.dot ins(%x, %y: tensor<?xf64>, tensor<?xf64>) 
      outs(%res: tensor<f64>) -> tensor<f64>
    return %dot: tensor<f64> 
  }

  func.func @main(%A : tensor<?x?xf64, #sparse>, %x : tensor<?xf64>, %y : tensor<?xf64>) 
  -> f64 {
    %0 = func.call @spmv(%A, %x, %y) { noinline, fuse_with = "dot" } : 
      (tensor<?x?xf64, #sparse>, tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>

    %dot_res = tensor.empty() : tensor<f64>
    %1 = func.call @dot(%0, %x, %dot_res) { noinline, fuse_with = "spmv" } : 
      (tensor<?xf64>, tensor<?xf64>, tensor<f64>) -> tensor<f64> 

    %ret = tensor.extract %1[] : tensor<f64> 
    return %ret : f64
  }
}

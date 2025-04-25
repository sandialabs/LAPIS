module attributes {torch.debug_module_name = "BatchedMatmul"} {
  func.func @forward(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 0.000000e+00 : f32
    %dim = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
    %dim_1 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
    %dim_2 = tensor.dim %arg1, %c0 : tensor<?x?x?xf32>
    %dim_3 = tensor.dim %arg1, %c1 : tensor<?x?x?xf32>
    %dim_4 = tensor.dim %arg1, %c2 : tensor<?x?x?xf32>
    %0 = arith.index_cast %dim : index to i64
    %1 = arith.index_cast %dim_2 : index to i64
    %2 = arith.cmpi eq, %0, %1 : i64
    cf.assert %2, "mismatching contracting dimension"
    %3 = arith.index_cast %dim_1 : index to i64
    %4 = arith.index_cast %dim_3 : index to i64
    %5 = arith.cmpi eq, %3, %4 : i64
    cf.assert %5, "mismatching contracting dimension"
    %8 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%arg2 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    return %8 : tensor<?x?x?xf32>
  }
}


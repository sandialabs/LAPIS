import torch
from lapis import KokkosBackend
from torch import nn
import numpy as np
import sys

module = """
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
"""

def main():
    a = torch.ones((5, 5)) + torch.eye(5)
    a = torch.broadcast_to(a, (64, 5, 5))
    b = torch.ones((5, 5)) - torch.eye(5)
    b = torch.broadcast_to(b, (64, 5, 5))
    ckokkos = torch.zeros((64, 5, 5))

    backend = KokkosBackend.KokkosBackend(dump_mlir=False)
    k_backend = backend.compile(module)

    print("a*b from kokkos (showing slice [0,:,:] only)")
    k_backend.forward(a, b, ckokkos)
    print(ckokkos[0, :, :])

    print("a*b from pytorch (showing slice [0,:,:] only)")
    ctorch = torch.zeros((64, 5, 5))
    torch.bmm(a, b, out=ctorch)
    print(ctorch[0, :, :])

    if np.allclose(ctorch, ckokkos):
        print("Success, results match")
        sys.exit(0)
    else:
        print("Failure, results do not match")
        sys.exit(1)

if __name__ == "__main__":
    main()


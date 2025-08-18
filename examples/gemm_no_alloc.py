import torch
from torch import Tensor
from lapis import KokkosBackend
from torch import nn
import numpy as np
import sys

module = """
module attributes {torch.debug_module_name = "Matmul"} {
  func.func @forward(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %dim_0 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
    %dim_1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %dim_2 = tensor.dim %arg1, %c0 : tensor<?x?xf32>
    %0 = arith.cmpi eq, %dim_1, %dim_2 : index
    cf.assert %0, "mismatching contracting dimension for torch.aten.mm"
    %3 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
    return %3: tensor<?x?xf32>
  }
}
"""

def main():
    m = 8
    k = 7
    n = 9
    a = torch.rand(m, k)
    b = torch.rand(k, n)
    ckokkos = torch.zeros((m, n))

    backend = KokkosBackend.KokkosBackend(dump_mlir=False)
    should_compile = True
    if should_compile:
        k_backend = backend.compile(module)
    else:
        import lapis_package.lapis_package as k_backend

    print("a*b from kokkos")
    print(f"{type(a)=} {type(b)=} {type(ckokkos)=}")
    k_backend.forward(a, b, ckokkos)
    print(ckokkos)

    print("a*b from pytorch")
    ctorch = torch.zeros((m, n))
    torch.matmul(a, b, out=ctorch)
    print(ctorch)

    if np.allclose(ctorch, ckokkos):
        print("Success, results match")
        sys.exit(0)
    else:
        print("Failure, results do not match")
        sys.exit(1)

if __name__ == "__main__":
    main()

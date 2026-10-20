import torch
from torch import Tensor
from torch_mlir import fx
from lapis import KokkosBackend
from torch import nn
import numpy as np
import sys

class BatchedMatmul(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        return torch.bmm(a, b)

def main():
    a = torch.ones((5, 5)) + torch.eye(5)
    a = torch.broadcast_to(a, (64, 5, 5))
    b = torch.ones((5, 5)) - torch.eye(5)
    b = torch.broadcast_to(b, (64, 5, 5))

    m = BatchedMatmul()
    m.train(False)

    dim_batch = torch.export.Dim("batch", min=1)
    dim_m = torch.export.Dim("m", min=1)
    dim_k = torch.export.Dim("k", min=1)
    dim_n = torch.export.Dim("n", min=1)
    dynamic_shapes = ({0: dim_batch, 1: dim_m, 2: dim_k}, {0: dim_batch, 1: dim_k, 2: dim_n})

    mlir_module = fx.export_and_import(m, a, b, output_type='linalg-on-tensors', func_name='forward', dynamic_shapes=dynamic_shapes)

    print(mlir_module)

    backend = KokkosBackend.KokkosBackend(dump_mlir=False)
    k_backend = backend.compile(mlir_module)

    print("a*b from kokkos (showing slice [0,:,:] only)")
    ckokkos = k_backend.forward(a, b)
    print(ckokkos[0, :, :])

    print("a*b from pytorch (showing slice [0,:,:] only)")
    ctorch = m.forward(a, b).numpy()
    print(ctorch[0, :, :])

    if np.allclose(ctorch, ckokkos):
        print("Success, results match")
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()


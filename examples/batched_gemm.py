import torch
from torch import Tensor
import torch_mlir
from torch_mlir import torchscript
from lapis import KokkosBackend
from torch import nn
from torch_mlir.compiler_utils import TensorPlaceholder
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

    aPH = TensorPlaceholder([-1, -1, -1], a.dtype)
    bPH = TensorPlaceholder([-1, -1, -1], b.dtype)

    mlir_module = torchscript.compile(m, (aPH, bPH), output_type='linalg-on-tensors')

    print(mlir_module)

    backend = KokkosBackend.KokkosBackend(dump_mlir=True)
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


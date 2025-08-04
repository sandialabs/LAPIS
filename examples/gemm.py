import torch
from torch import Tensor
import torch_mlir
from torch_mlir import torchscript
from lapis import KokkosBackend
from torch import nn
from torch_mlir.compiler_utils import TensorPlaceholder

class Matmul(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        return torch.matmul(a, b)

def main():
    a = torch.ones((5, 5)) + torch.eye(5)
    b = torch.ones((5, 5)) - torch.eye(5)

    m = Matmul()
    m.train(False)

    aPH = TensorPlaceholder([-1, -1], a.dtype)
    bPH = TensorPlaceholder([-1, -1], b.dtype)

    mlir_module = torchscript.compile(m, (aPH, bPH), output_type='linalg-on-tensors')

    backend = KokkosBackend.KokkosBackend(dump_mlir=False)
    k_backend = backend.compile(mlir_module)

    print("a*b from kokkos")
    print(k_backend.forward(a, b))

    print("a*b from pytorch")
    print(m.forward(a, b).numpy())

if __name__ == "__main__":
    main()


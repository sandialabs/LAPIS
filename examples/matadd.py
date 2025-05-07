import torch
from torch import Tensor
import torch_mlir
from torch_mlir import torchscript
from lapis import KokkosBackend
from torch import nn
import sys
from numpy import allclose

class Adder(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        return a + b


def main():
    a = torch.ones((5, 5))
    b = torch.eye(5)

    m = Adder()
    m.train(False)

    mlir_module = torchscript.compile(m, (a, b), output_type='linalg-on-tensors')

    print(mlir_module)

    backend = KokkosBackend.KokkosBackend()
    k_backend = backend.compile(mlir_module)

    print("a+b from pytorch")
    sumTorch = m.forward(a, b).numpy()
    print(sumTorch)

    print("a+b from kokkos")
    sumKokkos = k_backend.forward(a, b)
    print(sumKokkos)

    sys.exit(0 if allclose(sumTorch, sumKokkos) else 1)

if __name__ == "__main__":
    main()


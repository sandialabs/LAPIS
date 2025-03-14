import torch
from torch import Tensor
import torch_mlir
from torch_mlir import torchscript
from lapis import KokkosBackend
from torch import nn

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

    backend = KokkosBackend.KokkosBackend(dump_mlir=True)
    k_backend = backend.compile(mlir_module)

    print("a+b from kokkos")
    print(k_backend.forward(a, b))

    print("a+b from pytorch")
    print(m.forward(a, b).numpy())

if __name__ == "__main__":
    main()


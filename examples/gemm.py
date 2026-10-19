import torch
from torch import Tensor
from torch_mlir import fx
from lapis import KokkosBackend
from torch import nn

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

    mlir_module = fx.export_and_import(m, a, b, output_type='linalg-on-tensors', func_name='forward')

    backend = KokkosBackend.KokkosBackend(dump_mlir=False)
    k_backend = backend.compile(mlir_module)

    print("a*b from kokkos")
    print(k_backend.forward(a, b))

    print("a*b from pytorch")
    print(m.forward(a, b).numpy())

if __name__ == "__main__":
    main()


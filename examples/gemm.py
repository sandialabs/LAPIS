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

    dim_m = torch.export.Dim("m", min=1)
    dim_k = torch.export.Dim("k", min=1)
    dim_n = torch.export.Dim("n", min=1)
    dynamic_shapes = ({0: dim_m, 1: dim_k}, {0: dim_k, 1: dim_n})

    mlir_module = fx.export_and_import(m, a, b, output_type='linalg-on-tensors', func_name='forward', dynamic_shapes=dynamic_shapes)

    backend = KokkosBackend.KokkosBackend(dump_mlir=False)
    k_backend = backend.compile(mlir_module)

    print("a*b from kokkos")
    print(k_backend.forward(a, b).asnumpy())

    print("a*b from pytorch")
    print(m.forward(a, b).numpy())

if __name__ == "__main__":
    main()


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

class Matmul(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        return torch.matmul(a, b)

def main():
    a = torch.ones((5, 5))
    b = torch.eye(5)

    m1 = Adder()
    m1.train(False)
    m2 = Matmul()
    m2.train(False)

    ir1 = torchscript.compile(m1, (a, b), output_type='linalg-on-tensors')
    ir2 = torchscript.compile(m2, (a, b), output_type='linalg-on-tensors')

    backend1 = KokkosBackend.KokkosBackend(index_instance=0, num_instances=2)
    k1 = backend1.compile(ir1)
    backend2 = KokkosBackend.KokkosBackend(index_instance=1, num_instances=2)
    k2 = backend2.compile(ir2)

    print("Sum:")
    print(k1.forward(a,b))
    print("Prod:")
    print(k2.forward(a,b))

    del k1
    del k2

if __name__ == "__main__":
    main()


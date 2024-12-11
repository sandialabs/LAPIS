import torch
from torch import Tensor
#import torch_mlir
#from torch_mlir import torchscript
#from lapis import KokkosBackend
from torch import nn
from mpact.mpactbackend import mpact_linalg

class SpMV(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, x):
        return torch.mv(A, x)

def main():
    rowptrs = [0, 1, 5, 6, 8, 8]
    colinds = [1, 0, 2, 3, 4, 2, 0, 1]
    values = [1.1, 0.3, 2.2, 3.7, -4, -19, -2, 1]

    A = torch.sparse_csr_tensor( \
            torch.tensor(rowptrs, dtype=torch.int32), \
            torch.tensor(colinds, dtype=torch.int32), \
            torch.tensor(values, dtype=torch.float))

    x = torch.ones((5))

    # What A*x should be
    ygold = [1.1000, 2.2000, -19.0000, -1.0000, 0.0000]

    m = SpMV()
    m.train(False)

    module = mpact_linalg(m, A, x)

    print("MLIR at linalg level: ")
    print(module.operation.get_asm())

    #mlir_module = torchscript.compile(m, (a, b), output_type='linalg-on-tensors')

    #backend = KokkosBackend.KokkosBackend(dump_mlir=True)
    #k_backend = backend.compile(mlir_module)

    #c = k_backend.forward(a, b)
    #print("c from kokkos")
    #print(c)

    #print("c from pytorch")
    #print(m.forward(a, b))

if __name__ == "__main__":
    main()


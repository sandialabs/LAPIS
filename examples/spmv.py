import torch
from torch import Tensor
#import torch_mlir
#from torch_mlir import torchscript
#from torch_mlir import torchscript
from lapis import KokkosBackend
from torch import nn
from scipy.io import mmread

class SpMV(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, x):
        return torch.mv(A, x)

def main():
    n = 504855
    Asp = mmread('af_shell7.mtx').tocsr()

    A = torch.sparse_csr_tensor( \
            torch.tensor(Asp.indptr, dtype=torch.int32), \
            torch.tensor(Asp.indices, dtype=torch.int32), \
            torch.tensor(Asp.data, dtype=torch.float))

    x = torch.ones((n), dtype = torch.float)
    y = torch.ones((n), dtype = torch.float)

    #with torch.no_grad():
    #    m = SpMV()
    #    m.train(False)
    #    m.forward(A, x, y)

    with torch.no_grad():
        m = SpMV()
        m.train(False)
        backend = KokkosBackend.KokkosBackend(dump_mlir=True)
        k_backend = backend.compile_mpact(m, (A, x))

    #m = SpMV().eval()
    #module = torchscript.compile(m, (A, x, y), output_type="linalg-on-tensors")
    ##backend = refbackend.RefBackendLinalgOnTensorsBackend()
    ##compiled = backend.compile(module)
    ##jit_module = backend.load(compiled)
    #backend = KokkosBackend.KokkosBackend(dump_mlir=False)
    #kBackend.compile(module)


    #print("MLIR at linalg level: ")
    #print(module.operation.get_asm())

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


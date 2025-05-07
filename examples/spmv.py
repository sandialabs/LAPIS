import torch
from lapis import KokkosBackend
from mpact.mpactbackend import mpact_linalg
from torch import nn

from scipy.io import mmread
from utils.NewSparseTensorFactory import newSparseTensorFactory
from utils.NewSparseTensorFactory import LevelFormat
import numpy as np
import ctypes
import sys

class SpMV(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, x):
        return torch.mv(A, x)

def main():
    rowptrs = np.array([0, 1, 5, 6, 8, 8], dtype=np.int32)
    colinds = np.array([1, 0, 2, 3, 4, 2, 0, 1], dtype=np.int32)
    values = np.array([1.1, 0.3, 2.2, 3.7, -4, -19, -2, 1], dtype=np.double)
    A = torch.sparse_csr_tensor( \
            torch.tensor(rowptrs, dtype=torch.int32), \
            torch.tensor(colinds, dtype=torch.int32), \
            torch.tensor(values, dtype=torch.double))
    m = 5
    n = 5

    x = torch.ones((n), dtype = torch.double)
    y = torch.ones((m), dtype = torch.double)

    module_torch = SpMV()
    module_torch.train(False)

    # Use MPACT/TorchFX to export the torch module while maintaining sparsity
    # (torchscript, which we use for dense examples, can't do this)
    module_linalg = mpact_linalg(module_torch, A, x)
    backend = KokkosBackend.KokkosBackend(decompose_tensors=True)
    module_kokkos = backend.compile(module_linalg)

    print("y = Ax from torch:")
    ytorch = module_torch.forward(A, x).numpy()
    print(ytorch)
    print("y = Ax from kokkos:")
    ykokkos = module_kokkos.lapis_main(rowptrs, colinds, values, ((m, n), (len(rowptrs), len(colinds), len(values))), x.numpy())
    print(ykokkos)
    sys.exit(0 if np.allclose(ytorch, ykokkos) else 1)

if __name__ == "__main__":
    main()


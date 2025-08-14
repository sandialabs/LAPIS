import torch
from torch import Tensor
import torch_mlir
from torch_mlir import torchscript
from lapis import KokkosBackend
from torch import nn
import sys
from numpy import allclose
from torch_mlir.compiler_utils import TensorPlaceholder

def main():
    N = 2
    C = 3
    W = 30
    H = 20

    T = torch.rand(N,C,H,W)

    # Maxpool is used in the ResNet model architecture,
    # but since that code is very large and maxpool is tricky to lower,
    # break it out into its own test here.
    #
    # It is the only kernel we have encountered with
    # nested, multi-dimensional, non-sum reductions.
    m = nn.MaxPool2d(3, stride=None, padding=1, dilation=2)
    m.train(False)

    ph = TensorPlaceholder([-1, C, H, W], torch.float32)
    mlir_module = torchscript.compile(m, ph, output_type='linalg-on-tensors')

    backend = KokkosBackend.KokkosBackend(dump_mlir=False)
    k_backend = backend.compile(mlir_module)

    mpTorch = m(T).numpy()
    mpKokkos = k_backend.forward(T)

    if allclose(mpTorch, mpKokkos):
        print("Success, results match with torch")
        sys.exit(0)
    else:
        print("Torch result:")
        print(mpTorch)
        print("Kokkos result:")
        print(mpKokkos)
        print("Failure, results do not match torch")
        sys.exit(1)

if __name__ == "__main__":
    main()


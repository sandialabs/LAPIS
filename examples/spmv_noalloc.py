from lapis import KokkosBackend
import numpy as np
import ctypes
import sys

moduleText = """
#sparse = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed), posWidth = 32, crdWidth = 32 }>
module {
  func.func @spmv(%A: tensor<?x?xf64, #sparse>, %x: tensor<?xf64>, %y: tensor<?xf64>) -> tensor<?xf64> {
    %0 = linalg.matvec ins(%A, %x: tensor<?x?xf64, #sparse>, tensor<?xf64>) outs(%y : tensor<?xf64>) -> tensor<?xf64>
    return %0 : tensor<?xf64>
  }
}"""

def main():
    rowptrs = np.array([0, 1, 5, 6, 8, 8], dtype=np.int32)
    colinds = np.array([1, 0, 2, 3, 4, 2, 0, 1], dtype=np.int32)
    values = np.array([1.1, 0.3, 2.2, 3.7, -4, -19, -2, 1], dtype=np.double)
    m = 5
    n = 5

    x = np.ones((n), dtype=np.double)
    ykokkos = np.zeros((m), dtype=np.double)

    # Use MPACT/TorchFX to export the torch module while maintaining sparsity
    # (torchscript, which we use for dense examples, can't do this)
    backend = KokkosBackend.KokkosBackend(decompose_tensors=True)
    should_compile = True
    if should_compile:
        module_kokkos = backend.compile(moduleText)
    else:
        import lapis_package.lapis_package as module_kokkos

    print("y = Ax from kokkos:")
    module_kokkos.spmv(rowptrs, colinds, values, ((m, n), (len(rowptrs), len(colinds), len(values))), x, ykokkos)
    print(ykokkos)
    ycorrect = [1.1, 2.2, -19, -1, 0]
    if np.allclose(ykokkos, ycorrect):
        print("Success")
        sys.exit(0)
    else:
        print("Failure: incorrect result, should be", ycorrect)
        sys.exit(1)

if __name__ == "__main__":
    main()


from lapis import KokkosBackend
import numpy as np
import ctypes
import sys

moduleText = """
module {
  func.func @sum_prod(%arg0: memref<?xf64>) -> (f64, f64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = memref.dim %arg0, %c0 : memref<?xf64>
    %cst0 = arith.constant 0.000000e+00 : f64
    %cst1 = arith.constant 1.000000e+00 : f64
    %0:2 = scf.parallel (%arg2) = (%c0) to (%dim) step (%c1) init (%cst0, %cst1) -> (f64, f64) {
      %2 = memref.load %arg0[%arg2] : memref<?xf64>
      scf.reduce(%2, %2 : f64, f64) {
      ^bb0(%arg3: f64, %arg4: f64):
        %6 = arith.addf %arg3, %arg4 : f64
        scf.reduce.return %6 : f64
      }, {
      ^bb0(%arg3: f64, %arg4: f64):
        %6 = arith.mulf %arg3, %arg4 : f64
        scf.reduce.return %6 : f64
      }
    }
    return %0#0, %0#1 : f64, f64
  }
}
"""

def main():
    n = 100
    x = np.zeros((n), dtype=np.double)
    for i in range(n):
        x[i] = 1.05

    backend = KokkosBackend.KokkosBackend()
    module_kokkos = backend.compile(moduleText)

    [xsum, xprod] = module_kokkos.sum_prod(x)
    print("Sum:", xsum)
    print("Product:", xprod)
    correct = [n * 1.05, 1.05 ** n]
    if np.allclose(correct, [xsum, xprod]):
        print("Success")
        sys.exit(0)
    else:
        print("Failure: incorrect result", [xsum, xprod], "should be", correct)
        sys.exit(1)

if __name__ == "__main__":
    main()


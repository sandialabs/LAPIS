from lapis import KokkosBackend
import numpy as np
import ctypes
import sys

# Simple function that returns x * cos(y) + z

moduleText = """
module {
  func.func @nrm2(%arg0: memref<?xf64>) -> f64 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f64
    %dim = memref.dim %arg0, %c0 : memref<?xf64>
    %1 = scf.parallel (%arg1) = (%c0) to (%dim) step (%c1) init (%cst) -> (f64) {
      %2 = memref.load %arg0[%arg1] : memref<?xf64>
      %3 = arith.mulf %2, %2 : f64
      scf.reduce(%3 : f64) {
        ^bb0(%arg9: f64, %arg10: f64):
          %9 = arith.addf %arg9, %arg10 : f64
          scf.reduce.return %9 : f64
      }
    }
    return %1 : f64
  }
}"""

def main():
    backend = KokkosBackend.KokkosBackend(decompose_tensors=True, dump_mlir=False)
    module_kokkos = backend.forward_diff_compile(moduleText, 'nrm2', 'd_nrm2', ['dupnoneed'], ['dup'])
    eps = 1e-4
    vec = [1.0, -2.5, 1.5, 0.6]
    dvec = np.ones([4])
    print("nrm2(vec) =", module_kokkos.nrm2(vec))
    d = module_kokkos.d_nrm2(vec, dvec)
    print("derivative of nrm2(vec):", d)
    expected_d = 2 * sum(vec)
    print("expected derivative of nrm2(vec):", expected_d)

    if np.allclose([d], [expected_d]):
        print("Success")
        sys.exit(0)
    else:
        print("Failed")
        sys.exit(1)

if __name__ == "__main__":
    main()


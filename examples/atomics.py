from lapis import KokkosBackend
import numpy as np
import ctypes
import sys

# Program computes floor(log(k) / log(base))
# by repeatedly multiplying by base.

moduleText = """
module {
  func.func @atomics_test(%vec: memref<?xf64>) -> f64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %f0 = arith.constant 0.0 : f64 
    %n = memref.dim %vec, %c0 : memref<?xf64>
    %sum = memref.alloc() : memref<1xf64>
    memref.store %f0, %sum[%c0] : memref<1xf64>
    scf.parallel (%arg5) = (%c0) to (%n) step (%c1) {
      %0 = memref.load %vec[%arg5] : memref<?xf64>
      %1 = memref.atomic_rmw addf %0, %sum[%c0] : (f64, memref<1xf64>) -> f64
      scf.reduce
     }
    %result = memref.load %sum[%c0] : memref<1xf64>
    return %result : f64
  }
}"""

def main():
    backend = KokkosBackend.KokkosBackend()
    module_kokkos = backend.compile(moduleText)
    vec = np.array([1.1, 0.3, 2.2, 3.7, -4, -19, -2, 1], dtype=np.double)
    correctSum = np.sum(vec)
    resultSum = module_kokkos.atomics_test(vec)
    print("Atomic based sum: result was", resultSum, "and expected result was", correctSum)
    if np.allclose([resultSum], [correctSum]):
        print("Success")
        sys.exit(0)
    else:
        print("Failed")
        sys.exit(1)

if __name__ == "__main__":
    main()


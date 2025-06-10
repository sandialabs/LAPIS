from lapis import KokkosBackend
import numpy as np
import ctypes
import sys

# Program computes floor(log(k) / log(base))
# by repeatedly multiplying by base.

moduleText = """
module {
  func.func @while_test(%base: f64, %k: f64) -> index {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %f0 = arith.constant 0.0 : f64
    %f1 = arith.constant 1.0 : f64
    %pow, %result = scf.while (%prod = %f1, %i = %c0) : (f64, index) -> (f64, index)
    {
        %prodnext = arith.mulf %prod, %base : f64
        %continue = arith.cmpf ole, %prodnext, %k : f64
        scf.condition(%continue) %prodnext, %i : f64, index
    } do {
    ^bb0(%prodnext : f64, %i : index):
        %inext = arith.addi %i, %c1 : index
        scf.yield %prodnext, %inext : f64, index
    }
    return %result : index
  }
}"""

def main():
    backend = KokkosBackend.KokkosBackend()
    module_kokkos = backend.compile(moduleText)

    result = module_kokkos.while_test(1.5, 130.0)
    print("Result (should be 12): ", result)
    sys.exit(0 if result == 12 else 1)

if __name__ == "__main__":
    main()


from lapis import KokkosBackend
import numpy as np
import ctypes
import sys

# f returns column k of A with alpha (negate = false) or -alpha (negate = true) added to each element.
# Tests scalar arguments of different types.
moduleText = """
module {
  func.func @f(%A: tensor<8x8xf64>, %alpha: f64, %k: i32, %negate: i1) -> tensor<8xf64> {
    %kind = arith.index_cast %k : i32 to index
    %val = scf.if %negate -> f64 {
        %malpha = arith.negf %alpha : f64
        scf.yield %malpha : f64
    }
    else {
        scf.yield %alpha : f64
    }
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %col = tensor.extract_slice %A[0, %kind][8, 1][1, 1] : tensor<8x8xf64> to tensor<8xf64>
    %update = tensor.splat %val : tensor<8xf64>
    %o = tensor.empty() : tensor<8xf64>
    %sum = linalg.add ins(%col, %update : tensor<8xf64>, tensor<8xf64>) outs(%o : tensor<8xf64>) -> tensor<8xf64>
    return %sum : tensor<8xf64>
  }
}"""

def main():
    A = np.identity(8, dtype=np.double)
    alpha = 3.14
    k = 2

    xcorrect = np.copy(A[:, 2])
    for i in range(8):
        xcorrect[i] += alpha

    backend = KokkosBackend.KokkosBackend(decompose_tensors=True)
    should_compile = True
    if should_compile:
        module_kokkos = backend.compile(moduleText)
    else:
        import lapis_package.lapis_package as module_kokkos
    x = module_kokkos.f(A, alpha, k, False).asnumpy()
    print("Result 1:", x)

    if not np.allclose(x, xcorrect):
        print("Failure: incorrect result (negate = False), should be", xcorrect)
        sys.exit(1)

    xcorrect = np.copy(A[:, 2])
    for i in range(8):
        xcorrect[i] -= alpha

    x = module_kokkos.f(A, alpha, k, True).asnumpy()
    print("Result 2:", x)

    if not np.allclose(x, xcorrect):
        print("Failure: incorrect result (negate = True), should be", xcorrect)
        sys.exit(1)

    print("Success")
    sys.exit(0)

if __name__ == "__main__":
    main()


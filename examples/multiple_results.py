from lapis import KokkosBackend
import numpy as np
import ctypes
import sys

moduleText = """
module {
  func.func @plus_norm(%x: tensor<?xf64>, %y: tensor<?xf64>) -> (tensor<?xf64>, f64) {
    %c0 = arith.constant 0 : index
    %x_plus_y = linalg.add ins(%x, %y : tensor<?xf64>, tensor<?xf64>) outs(%y: tensor<?xf64>) -> tensor<?xf64>
    %temp = bufferization.alloc_tensor() : tensor<f64>
    %sumTensor = linalg.dot ins(%x_plus_y, %x_plus_y : tensor<?xf64>, tensor<?xf64>) outs(%temp: tensor<f64>) -> tensor<f64>
    %sum = tensor.extract %sumTensor[] : tensor<f64>
    %norm = math.sqrt %sum : f64
    return %x_plus_y, %norm : tensor<?xf64>, f64
  }
}
"""

def main():
    x = np.array([1.1, 0.3, 2.2, 3.7, -4, -19, -2, 1], dtype=np.double)
    y = np.array([4.1, -3.3, 2.7, -3.7, 4, 9, -2.4, 10], dtype=np.double)

    # Use MPACT/TorchFX to export the torch module while maintaining sparsity
    # (torchscript, which we use for dense examples, can't do this)
    backend = KokkosBackend.KokkosBackend(decompose_tensors=True)
    module_kokkos = backend.compile(moduleText)

    (x_plus_y, norm) = module_kokkos.plus_norm(x, y)
    print("x + y:", x_plus_y)
    print("norm(x+y):", norm)
    correct = x + y
    if np.allclose(x_plus_y, correct) and np.allclose(norm, np.linalg.norm(correct)):
        print("Success")
        sys.exit(0)
    else:
        print("Failure: incorrect result")
        sys.exit(1)

if __name__ == "__main__":
    main()


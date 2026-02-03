from lapis import KokkosBackend
import numpy as np
import ctypes
import sys

# Simple function that returns x * cos(y) + z

moduleText = """
module {
  func.func @myfunc(%x: f64, %y: f64, %z: f64) -> f64 {
    %0 = math.cos %y : f64
    %1 = arith.mulf %0, %x : f64
    %2 = arith.addf %1, %z : f64
    return %2 : f64
  }
}"""

def main():
    backend = KokkosBackend.KokkosBackend(decompose_tensors=True, dump_mlir=False)
    module_kokkos = backend.forward_diff_compile(moduleText, 'myfunc', 'grad_myfunc', ['dupnoneed'], ['dup', 'dup', 'dup'])
    eps = 1e-4
    # Pick some values for x,y,z and take secant approximation of gradient there
    x = 2.4
    y = -1.56
    z = 4.1234
    print("myfunc(x,y,z) =", module_kokkos.myfunc(x, y, z))
    dfdx_approx = (module_kokkos.myfunc(x + eps, y, z) - module_kokkos.myfunc(x - eps, y, z)) / (2.0 * eps)
    dfdy_approx = (module_kokkos.myfunc(x, y + eps, z) - module_kokkos.myfunc(x, y - eps, z)) / (2.0 * eps)
    dfdz_approx = (module_kokkos.myfunc(x, y, z + eps) - module_kokkos.myfunc(x, y, z - eps)) / (2.0 * eps)
    dfdx = module_kokkos.grad_myfunc(x, 1.0, y, 0.0, z, 0.0)
    dfdy = module_kokkos.grad_myfunc(x, 0.0, y, 1.0, z, 0.0)
    dfdz = module_kokkos.grad_myfunc(x, 0.0, y, 0.0, z, 1.0)
    print(f"              grad myfunc(x,y,z) = ({dfdx}, {dfdy}, {dfdz})")
    print(f"secant approx grad myfunc(x,y,z) = ({dfdx_approx}, {dfdy_approx}, {dfdz_approx})")
    if np.allclose([dfdx, dfdy, dfdz], [dfdx_approx, dfdy_approx, dfdz_approx]):
        print("Success")
        sys.exit(0)
    else:
        print("Failed")
        sys.exit(1)

if __name__ == "__main__":
    main()


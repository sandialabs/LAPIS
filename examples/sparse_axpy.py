from lapis import KokkosBackend
import numpy as np
import sys
from utils.NewSparseTensorFactory import newSparseTensorFactory
from utils.NewSparseTensorFactory import LevelFormat

moduleText = """
#map = affine_map<(d0) -> (d0)>
#sparse = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>
module {
  func.func @sparse_axpy(%arg0: tensor<16xi64, #sparse>, %arg1: tensor<16xi64, #sparse>) -> (tensor<16xi64, #sparse>, index) attributes {llvm.emit_c_interface} {
    %0 = tensor.empty() : tensor<16xi64, #sparse>
    %c2 = arith.constant 2 : i64
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<16xi64, #sparse>, tensor<16xi64, #sparse>) outs(%0 : tensor<16xi64, #sparse>) {
    ^bb0(%in: i64, %in_0: i64, %out: i64):
      %3 = sparse_tensor.binary %in, %in_0 : i64, i64 to i64
       overlap = {
      ^bb0(%arg2: i64, %arg3: i64):
        %4 = arith.muli %arg2, %c2 : i64
        %5 = arith.addi %4, %arg3 : i64
        sparse_tensor.yield %5 : i64
      }
       left = {
      ^bb0(%arg2: i64):
        %6 = arith.muli %arg2, %c2 : i64
        sparse_tensor.yield %6 : i64
      }
       right = {
      ^bb0(%arg2: i64):
        sparse_tensor.yield %arg2 : i64
      }
      linalg.yield %3 : i64
    } -> tensor<16xi64, #sparse>
    %2 = sparse_tensor.number_of_entries %1 : tensor<16xi64, #sparse>
    return %1, %2 : tensor<16xi64, #sparse>, index
  }

  func.func @print_sparse_vec(%arg0: tensor<16xi64, #sparse>) {
    sparse_tensor.print %arg0 : tensor<16xi64, #sparse>
    return
  }

  func.func @sparse_to_dense(%arg0: tensor<16xi64, #sparse>) -> tensor<16xi64> {
    %0 = sparse_tensor.convert %arg0 : tensor<16xi64, #sparse> to tensor<16xi64>
    return %0 : tensor<16xi64>
  }
}
"""

# Run sparse_axpy and check output. Return True if success.
def check_axpy(module_kokkos, v1_pos, v1_inds, v1_vals, v2_pos, v2_inds, v2_vals):
    gold = np.zeros(16)
    for i in range(len(v1_inds)):
        gold[v1_inds[i]] += 2 * v1_vals[i]
    for i in range(len(v2_inds)):
        gold[v2_inds[i]] += v2_vals[i]
    v1 = newSparseTensorFactory()([16], np.int64, postype=np.int64, crdtype=np.int64, buffers=[v1_pos, v1_inds, v1_vals])
    v2 = newSparseTensorFactory()([16], np.int64, postype=np.int64, crdtype=np.int64, buffers=[v2_pos, v2_inds, v2_vals])
    correct_nnz = sum([1 for val in gold if val != 0])
    [result, actual_nnz] = module_kokkos.sparse_axpy(v1, v2)
    print("Test case result:")
    module_kokkos.print_sparse_vec(result)
    result = module_kokkos.sparse_to_dense(result)
    if correct_nnz != actual_nnz:
        print("Failed: result nonzero count incorrect")
        return False
    if not np.allclose(gold, result):
        print("Failed: result value(s) incorrect")
        return False
    print("Success")
    return True

def main():
    # Create two sparse vectors of i64, which have length 16
    v1_pos = np.array([0, 6], dtype=np.int64)
    v1_inds = np.array([0, 1, 5, 6, 7, 15], dtype=np.int64)
    v1_vals = np.array([-2, 1.2, 3, 3.14159, -10, -20], dtype=np.int64)
    v2_pos = np.array([0, 5], dtype=np.int64)
    v2_inds = np.array([7, 9, 10, 11, 15], dtype=np.int64)
    v2_vals = np.array([1, -2, 3, -4, 5], dtype=np.int64)
    empty_pos = np.array([0, 0], dtype=np.int64)
    empty_inds = np.array([], dtype=np.int64)
    empty_vals = np.array([], dtype=np.int64)
    # Use MPACT/TorchFX to export the torch module while maintaining sparsity
    # (torchscript, which we use for dense examples, can't do this)
    backend = KokkosBackend.KokkosBackend()
    module_kokkos = backend.compile(moduleText)

    failures = 0
    if not check_axpy(module_kokkos, v1_pos, v1_inds, v1_vals, v2_pos, v2_inds, v2_vals):
        failures += 1
    if not check_axpy(module_kokkos, v2_pos, v2_inds, v2_vals, v1_pos, v1_inds, v1_vals):
        failures += 1
    if not check_axpy(module_kokkos, v1_pos, v1_inds, v1_vals, empty_pos, empty_inds, empty_vals):
        failures += 1
    if not check_axpy(module_kokkos, empty_pos, empty_inds, empty_vals, v1_pos, v1_inds, v1_vals):
        failures += 1
    if not check_axpy(module_kokkos, v1_pos, v1_inds, v1_vals, v1_pos, v1_inds, v1_vals):
        failures += 1
    if not check_axpy(module_kokkos, empty_pos, empty_inds, empty_vals, empty_pos, empty_inds, empty_vals):
        failures += 1
    if failures == 0:
        print("All cases succeeded")
        sys.exit(0)
    else:
        print("At least one case failed")
        sys.exit(1)

if __name__ == "__main__":
    main()


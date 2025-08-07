from lapis import KokkosBackend
import numpy as np
import sys
from utils.NewSparseTensorFactory import newSparseTensorFactory
from utils.NewSparseTensorFactory import LevelFormat

moduleText = """
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#dcsr = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed, d1 : compressed) }>
#sparse_vec = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>
module {
  func.func @column_sums(%arg0: tensor<16x16xi64, #dcsr>) -> (tensor<16xi64, #sparse_vec>, index, index) attributes {llvm.emit_c_interface} {
    %0 = tensor.empty() : tensor<16xi64, #sparse_vec>
    %c0_i64 = arith.constant 0 : i64
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction", "parallel"]} ins(%arg0 : tensor<16x16xi64, #dcsr>) outs(%0 : tensor<16xi64, #sparse_vec>) {
    ^bb0(%in: i64, %out: i64):
      %3 = sparse_tensor.reduce %out, %in, %c0_i64 : i64 {
      ^bb0(%arg1: i64, %arg2: i64):
        %4 = arith.addi %arg1, %arg2 : i64
        sparse_tensor.yield %4 : i64
      }
      linalg.yield %3 : i64
    } -> tensor<16xi64, #sparse_vec>
    %rank = tensor.rank %1 : tensor<16xi64, #sparse_vec>
    %2 = sparse_tensor.number_of_entries %1 : tensor<16xi64, #sparse_vec>
    return %1, %rank, %2 : tensor<16xi64, #sparse_vec>, index, index
  }

  func.func @dense_to_dcsr(%arg0: tensor<16x16xi64>) -> tensor<16x16xi64, #dcsr> {
    %0 = sparse_tensor.convert %arg0 : tensor<16x16xi64> to tensor<16x16xi64, #dcsr>
    return %0 : tensor<16x16xi64, #dcsr>
  }

  func.func @print_dcsr(%arg0: tensor<16x16xi64, #dcsr>) {
    sparse_tensor.print %arg0 : tensor<16x16xi64, #dcsr>
    return
  }

  func.func @sparse_vec_to_dense(%arg0: tensor<16xi64, #sparse_vec>) -> tensor<16xi64> {
    %0 = sparse_tensor.convert %arg0 : tensor<16xi64, #sparse_vec> to tensor<16xi64>
    return %0 : tensor<16xi64>
  }
}
"""

def main():
    backend = KokkosBackend.KokkosBackend()
    module_kokkos = backend.compile(moduleText)
    # Construct a very sparse 16x16 matrix, then convert it to DCSR format
    A = np.zeros((16, 16), dtype=np.int64)
    A[1, 1] = 5
    A[2, 1] = -4
    A[2, 2] = 2
    A[7, 9] = 14
    A[1, 15] = 7
    A[14, 3] = 8
    # Compute the correct result for column sums
    gold = np.zeros((16), dtype=np.int64)
    for i in range(16):
        for j in range(16):
            gold[j] += A[i, j]
    gold_nnz = 0
    for i in range(16):
        if gold[i] != 0:
            gold_nnz += 1

    A_dcsr = module_kokkos.dense_to_dcsr(A)
    module_kokkos.print_dcsr(A_dcsr)
    [result, rank, nnz] = module_kokkos.column_sums(A_dcsr)
    # Convert the sparse vector result to dense to check the output
    result_dense = module_kokkos.sparse_vec_to_dense(result)
    print("Results:        ", rank, nnz, result_dense)
    print("Correct result: ", 1, gold_nnz, gold)

    if rank == 1 and nnz == gold_nnz and np.allclose(result_dense, gold):
        print("Success: all results correct")
        sys.exit(0)
    else:
        print("Failure: at least one result incorrect")
        sys.exit(1)

if __name__ == "__main__":
    main()


import numpy as np
import ctypes
import sys
from lapis import KokkosBackend
from utils.NewSparseTensorFactory import newSparseTensorFactory
from utils.NewSparseTensorFactory import LevelFormat
from scipy.sparse import csr_matrix

moduleText = """
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#csrv = #sparse_tensor.encoding<{ map = (d0, d1, d2) ->
    (d0 : dense, d1 : compressed, d2 : dense)}>

module {
  func.func @pte_softmax(%attn2: tensor<?x?x?xf64, #csrv>) -> tensor<?x?x?xf64, #csrv> {
    %sc0 = arith.constant 0 : index
    %sc1 = arith.constant 1 : index
    %sc1_i8 = arith.constant 1 : i8
    %sc2 = arith.constant 2 : index
    %scst = arith.constant 0.000000e+00 : f64
    %sdim = tensor.dim %attn2, %sc0 : tensor<?x?x?xf64, #csrv>
    %sdim_0 = tensor.dim %attn2, %sc1 : tensor<?x?x?xf64, #csrv>
    %sdim_1 = tensor.dim %attn2, %sc2 : tensor<?x?x?xf64, #csrv>
    %sc0_2 = arith.constant 0 : index
    %sdim_3 = tensor.dim %attn2, %sc0_2 : tensor<?x?x?xf64, #csrv>
    %sc1_4 = arith.constant 1 : index
    %sdim_5 = tensor.dim %attn2, %sc1_4 : tensor<?x?x?xf64, #csrv>
    %sc2_6 = arith.constant 2 : index
    %sdim_7 = tensor.dim %attn2, %sc2_6 : tensor<?x?x?xf64, #csrv>
    %s11 = tensor.splat %scst[%sdim_3, %sdim_7] : tensor<?x?xf64>
    %sminus_inf = arith.constant -3.40282347E+38 : f64

    %s21 = linalg.fill ins(%sminus_inf : f64) outs(%s11 : tensor<?x?xf64>)
      -> tensor<?x?xf64>
    %s31 = linalg.generic {indexing_maps = [#map, #map1],
      iterator_types = ["parallel", "reduction", "parallel"]}
      ins(%attn2 : tensor<?x?x?xf64, #csrv>) outs(%s21 : tensor<?x?xf64>) {
        ^bb0(%sin: f64, %sout: f64):
          %sres = sparse_tensor.reduce %sin, %sout, %sminus_inf : f64 {
            ^bb0(%sx0: f64, %sx1: f64):
              %s00 = arith.maxnumf %sx0, %sx1 : f64
              sparse_tensor.yield %s00: f64
          }
          linalg.yield %sres : f64
    } -> tensor<?x?xf64>
    %s3 = linalg.generic {indexing_maps = [#map, #map],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%attn2 : tensor<?x?x?xf64, #csrv>)
      outs(%attn2 : tensor<?x?x?xf64, #csrv>) {
        ^bb0(%sin: f64, %sout: f64):
          %sx = linalg.index 0: index
          %sz = linalg.index 2: index
          %sresult = sparse_tensor.unary %sin : f64 to f64
          present={
          ^bb0(%sin1: f64):
            %smaxel = tensor.extract %s31[%sx, %sz]: tensor<?x?xf64>
            %s8 = arith.subf %sin1, %smaxel : f64
            %sret = math.exp %s8 : f64
            sparse_tensor.yield %sret : f64
          }
          absent={}
          linalg.yield %sresult : f64
    } -> tensor<?x?x?xf64, #csrv>
    %s1 = tensor.splat %scst[%sdim_3, %sdim_7] : tensor<?x?xf64>
    %scst_8 = arith.constant 0. : f64
    %s2 = linalg.fill ins(%scst_8 : f64) outs(%s1 : tensor<?x?xf64>)
      -> tensor<?x?xf64>
    %s4 = linalg.generic {indexing_maps = [#map, #map1],
      iterator_types = ["parallel", "reduction", "parallel"]}
      ins(%s3 : tensor<?x?x?xf64, #csrv>) outs(%s2 : tensor<?x?xf64>) {
        ^bb0(%sin: f64, %sout: f64):
          %sres = sparse_tensor.reduce %sin, %sout, %scst_8 : f64 {
            ^bb0(%sx0: f64, %sx1: f64):
              %s00 = arith.addf %sx0, %sx1 : f64
              sparse_tensor.yield %s00: f64
          }
          linalg.yield %sres : f64
    } -> tensor<?x?xf64>
    %attn31  = linalg.generic {indexing_maps = [#map],
      iterator_types = ["parallel", "parallel", "parallel"]}
      outs(%s3: tensor<?x?x?xf64, #csrv>) {
        ^bb0(%sin: f64):
          %sx = linalg.index 0: index
          %sz = linalg.index 2: index
          %sresult = sparse_tensor.unary %sin : f64 to f64
          present={
          ^bb0(%sin1: f64):
            %sdenom = tensor.extract %s4[%sx, %sz]: tensor<?x?xf64>
            %sret = arith.divf %sin1, %sdenom : f64
            sparse_tensor.yield %sret : f64
          }
          absent={}
          linalg.yield %sresult : f64
    } -> tensor<?x?x?xf64, #csrv>
    bufferization.dealloc_tensor %s1: tensor<?x?xf64>
    bufferization.dealloc_tensor %s11: tensor<?x?xf64>
    return %attn31 : tensor<?x?x?xf64, #csrv>
  }

  func.func @print_csrv(%A: tensor<?x?x?xf64, #csrv>)
  {
    sparse_tensor.print %A : tensor<?x?x?xf64, #csrv>
    return
  }

  func.func @dense_to_csrv(%arg0: tensor<?x?x?xf64>) -> tensor<?x?x?xf64, #csrv> {
    %0 = sparse_tensor.convert %arg0 : tensor<?x?x?xf64> to tensor<?x?x?xf64, #csrv>
    return %0 : tensor<?x?x?xf64, #csrv>
  }

  func.func @csrv_to_dense(%arg0: tensor<?x?x?xf64, #csrv>) -> tensor<?x?x?xf64> {
    %0 = sparse_tensor.convert %arg0 : tensor<?x?x?xf64, #csrv> to tensor<?x?x?xf64>
    return %0 : tensor<?x?x?xf64>
  }
}
"""

def main():
    # Define a small CSR graph, where each entry is a fiber of b values
    # Randomly populate so that each batch is a different matrix (but same sparsity pattern)
    m = 5   # Rows of A
    n = 6   # Cols of A
    b = 4   # Batch dim (fiber length)
    nnz = 8
    rowptrs = np.array([0, 1, 5, 6, 8, 8], dtype=np.int64)
    colinds = np.array([1, 0, 2, 3, 4, 5, 0, 1], dtype=np.int64)
    rng = np.random.default_rng()
    values = np.require(rng.uniform(1.0, 2.0, (nnz * b)), dtype=np.double, requirements="C")
    Adense = np.zeros((m, n, b), dtype=np.double)
    for i in range(m):
        rowBegin = rowptrs[i]
        rowEnd = rowptrs[i+1]
        for j in range(rowBegin, rowEnd):
            col = colinds[j]
            for k in range(b):
                Adense[i, col, k] = values[j*b + k]
        for k in range(b):
            Adense[i, i, k] = 1
    successes = []
    skipped = []
    failures = []
    parStrats = ['none', 'dense-outer-loop', 'dense-any-loop', 'any-storage-any-loop']
    instance = 0
    checkResult = None
    for par in parStrats:
        if par == 'any-storage-any-loop':
            skipped.append(par)
            continue
        # num_instances is only 3 since we skipped any-storage-any-loop
        backend = KokkosBackend.KokkosBackend(decompose_tensors=False, parallel_strategy=par, index_instance=instance, num_instances=len(parStrats) - 1, dump_mlir=False)
        instance += 1
        module_kokkos = backend.compile(moduleText)
        A = module_kokkos.dense_to_csrv(Adense)
        # For debugging: print the CSRV formatted matrix
        #module_kokkos.print_csrv(A)
        result = module_kokkos.pte_softmax(A)
        resultDense = module_kokkos.csrv_to_dense(result).asnumpy()
        print("Result (converted to dense): ")
        print(resultDense)
        if checkResult is None:
            checkResult = np.copy(resultDense)
            successes.append(par)
        else:
            err = np.linalg.norm((checkResult - resultDense).flatten())
            print("RMS difference with sequential answer:", err)
            if err > 1e-8:
                print("par strat", par, "result disagreed with sequential")
                failures.append(par)
            else:
                successes.append(par)
    print("Succeeded for parallelization strategies:", successes)
    print("Failed for parallelization strategies:", failures)
    print("Skipped parallelization strategies:", skipped)
    if len(failures):
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()


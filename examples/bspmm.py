import numpy as np
import ctypes
import sys
from lapis import KokkosBackend
from utils.NewSparseTensorFactory import newSparseTensorFactory
from utils.NewSparseTensorFactory import LevelFormat
from scipy.sparse import csr_matrix

# input IR is from Miheer's 27-sparse-tensor-bspmm test
# (but where dense tensors use no explicit sparse_tensor encoding, for simplicity)
moduleText = """
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#csrv = #sparse_tensor.encoding<{ map = (d0, d1, d2) -> (d0 : dense, d1 : compressed, d2 : dense), posWidth = 32, crdWidth = 32}>
#bspmm_map = {
  indexing_maps = [
    affine_map<(n1, n2, dh, nh) -> (n1, n2, nh)>,  // attn (in)
    affine_map<(n1, n2, dh, nh) -> (n2, dh, nh)>,  // v (in)
    affine_map<(n1, n2, dh, nh) -> (n1, dh, nh)>   // out (out)
  ],
  iterator_types = ["parallel", "parallel", "reduction", "parallel"],
  doc = "out(n1, dh, nh) = attn(n1, n2, nh) * v(n2, dh, nh)"
}

module {
  func.func @pte_local_bspmm(%A: tensor<?x?x?xf32, #csrv>,
    %B: tensor<?x?x?xf32>) ->  tensor<?x?x?xf32>
  {
    %c0_index = arith.constant 0 : index
    %c1_index = arith.constant 1 : index
    %c2_index = arith.constant 2 : index

    %c0_f32 = arith.constant 0.0 : f32

    %N1 = tensor.dim %A, %c0_index : tensor<?x?x?xf32, #csrv>
    %N2 = tensor.dim %A, %c1_index : tensor<?x?x?xf32, #csrv>
    %nh = tensor.dim %A, %c2_index : tensor<?x?x?xf32, #csrv>
    %dh = tensor.dim %B, %c1_index : tensor<?x?x?xf32>
    %spmm_in0 = tensor.empty (%N1, %dh, %nh) : tensor<?x?x?xf32>
    %spmm_in1 = linalg.fill ins(%c0_f32: f32)
      outs(%spmm_in0 : tensor<?x?x?xf32>)
      -> tensor<?x?x?xf32>
    %attn4 = linalg.generic #bspmm_map
      ins(%A, %B: tensor<?x?x?xf32, #csrv>, tensor<?x?x?xf32>)
      outs(%spmm_in1: tensor<?x?x?xf32>) {
      ^bb0(%q: f32, %k: f32, %attn: f32):  // no predecessors
        %0 = arith.mulf %q, %k : f32
        %1 = arith.addf %0, %attn: f32
        linalg.yield %1 : f32
    } -> tensor<?x?x?xf32>
    return %attn4 : tensor<?x?x?xf32>
  }

  func.func @print_csrv(%A: tensor<?x?x?xf32, #csrv>)
  {
    sparse_tensor.print %A : tensor<?x?x?xf32, #csrv>
    return
  }
}
"""

def main():
    # Define a small CSR graph, where each entry is a fiber of b values
    # Randomly populate so that each batch is a different matrix (but same sparsity pattern)
    m = 5   # Rows of A and C
    n = 6
    b = 8
    k = 3
    # A shape: m x n x batch
    # B shape: n x k x batch
    # C shape: m x k x batch
    nnz = 8
    rowptrs = np.array([0, 1, 5, 6, 8, 8], dtype=np.int32)
    colinds = np.array([1, 0, 2, 3, 4, 5, 0, 1], dtype=np.int32)
    rng = np.random.default_rng()
    values = np.require(rng.uniform(1.0, 2.0, (nnz * b)), dtype=np.float32, requirements="C")

    B = np.require(rng.uniform(1.0, 2.0, (n, k, b)), dtype=np.float32, requirements="C")
    C_gold = np.zeros((m, k, b), dtype=np.float32)
    # Compute the correct result once, one instance at a time
    for i in range(b):
        valSlice = [values[j*b + i] for j in range(nnz)]
        Aslice = csr_matrix((valSlice, colinds, rowptrs), (m, n))
        Bslice = B[:, :, i]
        Cslice = C_gold[:, :, i]
        result = Aslice @ Bslice
        for j in range(m):
            for l in range(k):
                Cslice[j, l] = result[j, l]

    successes = []
    skipped = []
    failures = []
    parStrats = ['none', 'dense-outer-loop', 'dense-any-loop', 'any-storage-any-loop']
    instance = 0
    for par in parStrats:
        if par == 'any-storage-any-loop':
            skipped.append(par)
            continue
        backend = KokkosBackend.KokkosBackend(decompose_tensors=True, parallel_strategy=par, index_instance=instance, num_instances=len(parStrats))
        instance += 1
        module_kokkos = backend.compile(moduleText)
        C_kokkos = module_kokkos.pte_local_bspmm(rowptrs, colinds, values, ((m, n, b), (m+1, nnz, nnz*b)), B).asnumpy()
        # For debugging: print the CSRV formatted matrix
        # module_kokkos.print_csrv(rowptrs, colinds, values, ((m, n, b), (m+1, nnz, nnz*b)))
        if np.allclose(C_gold, C_kokkos):
            successes.append(par)
        else:
            failures.append(par)
    print("Succeeded for parallelization strategies:", successes)
    print("Failed for parallelization strategies:", failures)
    print("Skipped parallelization strategies:", skipped)
    if len(failures):
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()


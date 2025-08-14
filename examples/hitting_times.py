from lapis import KokkosBackend
import numpy as np
import ctypes
import sys
from scipy.sparse import csr_matrix
from scipy.io import mmread

moduleText="""
#sparse = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed), posWidth = 32, crdWidth = 32 }>
#idmap = affine_map<(d0) -> (d0)>
module {
  func.func private @spmv(%A: tensor<?x?xf64, #sparse>, %x: tensor<?xf64>, %ydst: tensor<?xf64>) -> tensor<?xf64> {
    %y = linalg.matvec ins(%A, %x: tensor<?x?xf64, #sparse>, tensor<?xf64>) outs(%ydst : tensor<?xf64>) -> tensor<?xf64>
    return %y : tensor<?xf64>
  }

  func.func private @dot(%x: tensor<?xf64>, %y: tensor<?xf64>) -> f64 {
    %f0 = arith.constant 0.0 : f64
    %0 = tensor.splat %f0[] : tensor<f64>
    %dot = linalg.dot ins(%x, %y : tensor<?xf64>,tensor<?xf64>) outs(%0: tensor<f64>) -> tensor<f64>
    %6 = tensor.extract %dot[] : tensor<f64>
    return %6: f64
  }

  func.func private @axpby(%a: f64, %x: tensor<?xf64>, %b: f64, %y: tensor<?xf64>, %dst: tensor<?xf64>) -> tensor<?xf64> {
    %1 = linalg.generic {indexing_maps = [#idmap, #idmap, #idmap], iterator_types = ["parallel"]} ins(%x, %y: tensor<?xf64>, tensor<?xf64>) outs(%dst : tensor<?xf64>) {
    ^bb0(%inx: f64, %iny: f64, %out: f64):
      %4 = arith.mulf %inx, %a: f64
      %5 = arith.mulf %iny, %b: f64
      %6 = arith.addf %4, %5: f64
      linalg.yield %6 : f64
    } -> tensor<?xf64>
    return %1 : tensor<?xf64>
  }

  func.func private @mult(%x: tensor<?xf64>, %y: tensor<?xf64>, %dst: tensor<?xf64>) -> tensor<?xf64> {
    %1 = linalg.generic {indexing_maps = [#idmap, #idmap, #idmap], iterator_types = ["parallel"]} ins(%x, %y: tensor<?xf64>, tensor<?xf64>) outs(%dst : tensor<?xf64>) {
    ^bb0(%inx: f64, %iny: f64, %out: f64):
      %2 = arith.mulf %inx, %iny: f64
      linalg.yield %2 : f64
    } -> tensor<?xf64>
    return %1 : tensor<?xf64>
  }

  // Scale the given CSR matrix by alpha.
  // This operation is in-place (it modifies A's memory when bufferized)
  func.func private @scale_csr(%A: tensor<?x?xf64, #sparse>, %alpha: f64) -> tensor<?x?xf64, #sparse> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = linalg.generic
      {indexing_maps = [affine_map<(i,j) -> (i,j)>],
       iterator_types = ["parallel", "parallel"]}
    outs(%A: tensor<?x?xf64, #sparse>) {
      ^bb0(%a: f64) :
        %result = sparse_tensor.unary %a : f64 to f64
         present={
           ^bb1(%arg0: f64):
             %ret = arith.mulf %arg0, %alpha : f64
             sparse_tensor.yield %ret : f64
         }
      absent={}
      linalg.yield %result : f64
    } -> tensor<?x?xf64, #sparse>
    return %0 : tensor<?x?xf64, #sparse>
  }

  // Apply the normalized graph Laplacian, with the seed-set mask also applied to both the input and output vectors
  // y := (PiPi^T)(D-alpha*A)(PiPi^T)x
  func.func private @plpt(%Aalpha: tensor<?x?xf64, #sparse>, %x: tensor<?xf64>, %d: tensor<?xf64>, %mask: tensor<?xf64>, %tmp: tensor<?xf64>, %dst: tensor<?xf64>) -> tensor<?xf64> {
    // first, apply mask to x (result goes in tmp, so that x is preserved)
    %mask_x = func.call @mult(%x, %mask, %tmp) : (tensor<?xf64>, tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
    // then perform the spmv using Aalpha
    %Aalpha_mask_x = func.call @spmv(%Aalpha, %mask_x, %dst) : (tensor<?x?xf64, #sparse>, tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
    // finish the calculation in a single elementwise operation:
    // take d*mask_x, subtract Aalpha_mask_x, then apply the mask to that.
    %0 = linalg.generic
      {indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>, affine_map<(i) -> (i)>, affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
       iterator_types = ["parallel"]}
    ins(%Aalpha_mask_x, %mask_x, %mask, %d : tensor<?xf64>, tensor<?xf64>, tensor<?xf64>, tensor<?xf64>)
    outs(%dst: tensor<?xf64>) {
      ^bb0(%aamx: f64, %mx : f64, %m: f64, %dval : f64, %unused : f64):
        %1 = arith.mulf %dval, %mx : f64
        %2 = arith.subf %1, %aamx : f64
        %3 = arith.mulf %2, %m : f64
        linalg.yield %3: f64
    } -> tensor<?xf64>
    return %0 : tensor<?xf64>
  }

  // CG solve with diagonal preconditioner
  // !! Note: d is not inverted here, so applying it as preconditioner is an elementwise division.
  // Returns: x, numiter, resnorm
  func.func private @pcg(%A: tensor<?x?xf64, #sparse>, %b: tensor<?xf64>, %mask: tensor<?xf64>, %d: tensor<?xf64>, %tol: f64, %maxiter: index) -> (tensor<?xf64>, index, f64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %n = tensor.dim %b, %c0 : tensor<?xf64>
    %f0 = arith.constant 0.0 : f64
    %f1 = arith.constant 1.0 : f64
    %fm1 = arith.constant -1.0 : f64
    // Preallocate some intermediate tensors for dst-passing style
    %tmp = tensor.empty(%n) : tensor<?xf64>
    %buf0 = tensor.empty(%n) : tensor<?xf64>
    %buf1 = tensor.empty(%n) : tensor<?xf64>
    %buf2 = tensor.empty(%n) : tensor<?xf64>
    // Assume initial guess x0 = 0
    // Then r0 = b - A*x0 = b
    %r0 = linalg.copy ins(%b : tensor<?xf64>) outs(%buf0 : tensor<?xf64>) -> tensor<?xf64>
    %z0 = linalg.div ins(%r0, %d : tensor<?xf64>, tensor<?xf64>) outs(%buf1 : tensor<?xf64>) -> tensor<?xf64>
    %p0 = linalg.copy ins(%z0 : tensor<?xf64>) outs(%buf2 : tensor<?xf64>) -> tensor<?xf64>
    %x0 = tensor.splat %f0[%n] : tensor<?xf64>
    %Apbuf = tensor.empty(%n) : tensor<?xf64>
    %rr0 = func.call @dot(%r0, %r0) : (tensor<?xf64>, tensor<?xf64>) -> f64
    %initres = math.sqrt %rr0 : f64
    %x, %p, %z, %r, %final_relres, %rz, %iters = scf.while (%xiter = %x0, %piter = %p0, %ziter = %z0, %riter = %r0, %rziter = %f0, %i = %c1) : (tensor<?xf64>, tensor<?xf64>, tensor<?xf64>, tensor<?xf64>, f64, index) -> (tensor<?xf64>, tensor<?xf64>, tensor<?xf64>, tensor<?xf64>, f64, f64, index)
    {
      //%Ap = func.call @spmv(%A, %piter, %Apbuf) : (tensor<?x?xf64, #sparse>, tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
      // note: %A here is premultiplied by discount (alpha)
      %Ap = func.call @plpt(%A, %piter, %d, %mask, %tmp, %Apbuf) : (tensor<?x?xf64, #sparse>, tensor<?xf64>, tensor<?xf64>, tensor<?xf64>, tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
      %pAp = func.call @dot(%piter, %Ap) : (tensor<?xf64>, tensor<?xf64>) -> f64
      %rz = func.call @dot(%riter, %ziter) : (tensor<?xf64>, tensor<?xf64>) -> f64
      %alpha = arith.divf %rz, %pAp : f64
      %malpha = arith.negf %alpha : f64
      // Update x and r
      %xnext = func.call @axpby(%f1, %xiter, %alpha, %piter, %xiter) : (f64, tensor<?xf64>, f64, tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
      %rnext = func.call @axpby(%f1, %riter, %malpha, %Ap, %riter) : (f64, tensor<?xf64>, f64, tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
      // Test against tolerance and
      %rr = func.call @dot(%rnext, %rnext) : (tensor<?xf64>, tensor<?xf64>) -> f64
      %rnorm = math.sqrt %rr : f64
      %relres = arith.divf %rnorm, %initres : f64
      %not_converged = arith.cmpf ogt, %relres, %tol : f64
      // we have already completed an iteration, which is why i is intially 1
      %below_maxiter = arith.cmpi ne, %i, %maxiter : index
      %continue = arith.andi %not_converged, %below_maxiter : i1
      scf.condition(%continue) %xnext, %piter, %ziter, %rnext, %relres, %rz, %i: tensor<?xf64>, tensor<?xf64>, tensor<?xf64>, tensor<?xf64>, f64, f64, index
    } do {
    ^bb0(%xiter : tensor<?xf64>, %piter : tensor<?xf64>, %ziter : tensor<?xf64>, %riter : tensor<?xf64>, %unused : f64, %rziter : f64, %i : index):
      %znext = linalg.div ins(%riter, %d : tensor<?xf64>, tensor<?xf64>) outs(%ziter : tensor<?xf64>) -> tensor<?xf64>
      %rznext = func.call @dot(%riter, %znext) : (tensor<?xf64>, tensor<?xf64>) -> f64
      %beta = arith.divf %rznext, %rziter : f64
      %pnext = func.call @axpby(%f1, %znext, %beta, %piter, %piter) : (f64, tensor<?xf64>, f64, tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
      %inext = arith.addi %i, %c1 : index
      scf.yield %xiter, %pnext, %znext, %riter, %rznext, %inext : tensor<?xf64>, tensor<?xf64>, tensor<?xf64>, tensor<?xf64>, f64, index
    }
    return %x, %iters, %final_relres : tensor<?xf64>, index, f64
  }

  // Compute the first two normalized hitting-time moments
  // Returns vectors m1, m2
  func.func @mht(%A: tensor<?x?xf64, #sparse>, %mask: tensor<?xf64>, %d: tensor<?xf64>, %alpha: f64, %tol: f64, %maxiter: index) -> (tensor<?xf64>, tensor<?xf64>) {
    %c0 = arith.constant 0 : index
    // Scale A by alpha (in-place)
    %Aalpha = func.call @scale_csr(%A, %alpha) : (tensor<?x?xf64, #sparse>, f64) -> tensor<?x?xf64, #sparse>
    %n = tensor.dim %d, %c0 : tensor<?xf64>
    // Form right-hand side for 1st moment: mask*d
    %bbuf = tensor.empty(%n) : tensor<?xf64>
    %b1 = func.call @mult(%d, %mask, %bbuf) : (tensor<?xf64>, tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
    // Call the solver
    %mom1, %its1, %res1 = func.call @pcg(%Aalpha, %b1, %mask, %d, %tol, %maxiter) : (tensor<?x?xf64, #sparse>, tensor<?xf64>, tensor<?xf64>, tensor<?xf64>, f64, index) -> (tensor<?xf64>, index, f64)
    // Form right-hand side for 2nd moment: mask * (D .* mom1 + Aalpha * mom1)
    // (where .* is ewise mult)
    %Aalpha_mom1 = func.call @spmv(%Aalpha, %mom1, %bbuf) : (tensor<?x?xf64, #sparse>, tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
    %b2 = linalg.generic
      {indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>, affine_map<(i) -> (i)>, affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
       iterator_types = ["parallel"]}
    ins(%Aalpha_mom1, %mask, %mom1, %d : tensor<?xf64>, tensor<?xf64>, tensor<?xf64>, tensor<?xf64>)
    outs(%bbuf: tensor<?xf64>) {
      ^bb0(%aam1: f64, %m : f64, %m1: f64, %dval : f64, %unused : f64):
        %1 = arith.mulf %dval, %m1 : f64
        %2 = arith.addf %1, %aam1 : f64
        %3 = arith.mulf %2, %m : f64
        linalg.yield %3: f64
    } -> tensor<?xf64>
    %mom2, %its2, %res2 = func.call @pcg(%Aalpha, %b2, %mask, %d, %tol, %maxiter) : (tensor<?x?xf64, #sparse>, tensor<?xf64>, tensor<?xf64>, tensor<?xf64>, f64, index) -> (tensor<?xf64>, index, f64)
    return %mom1, %mom2 : tensor<?xf64>, tensor<?xf64>
  }

  func.func @normalize_mom2(%mom1: tensor<?xf64>, %mom2: tensor<?xf64>) -> tensor<?xf64> {
    %c0 = arith.constant 0 : index
    %f0 = arith.constant 0.0 : f64
    %n = tensor.dim %mom1, %c0 : tensor<?xf64>
    %buf = tensor.empty(%n) : tensor<?xf64>
    %nm2 = linalg.generic
      {indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
       iterator_types = ["parallel"]}
    ins(%mom1, %mom2 : tensor<?xf64>, tensor<?xf64>)
    outs(%buf: tensor<?xf64>) {
      ^bb0(%m1: f64, %m2 : f64, %unused : f64):
        %m1_squared = arith.mulf %m1, %m1 : f64
        %stdev_squared = arith.subf %m2, %m1_squared : f64
        %stdev = math.sqrt %stdev_squared : f64
        linalg.yield %stdev : f64
    } -> tensor<?xf64>
    return %nm2 : tensor<?xf64>
  }
}
"""

def main():
    # Read in the tiny test graph and diagonal preconditioner
    A = csr_matrix(mmread('data/mht_A.mtx'))
    n = A.shape[0] # Assume A is square
    nnz = A.nnz
    # D is stored in MatrixMarket file, but really it is just 1D (diagonals only)
    D = csr_matrix(mmread('data/mht_D.mtx')).data
    mask = np.ones(n)
    seedset = [3]
    for s in seedset:
        mask[s] = 0
    backend = KokkosBackend.KokkosBackend(decompose_tensors=True)
    module_kokkos = backend.compile(moduleText)
    # Solve for first two hitting time moments 
    (mom1, mom2) = module_kokkos.mht(A.indptr, A.indices, A.data, ((n, n), (n + 1, nnz, nnz)), mask, D, 0.99, 1e-10, 20)
    # Normalize 2nd moment
    mom2_norm = module_kokkos.normalize_mom2(mom1, mom2)
    print("1st moment:", mom1)
    print("2nd moment:", mom2_norm)
    mom1_gold = [9.9999999999999957e+01, 9.9999999999999957e+01, 1.9947875961498600e+01, 0, 1.8542105566249749e+01, 1.9909372734041774e+01, 9.9999999999999957e+01, 1.9970947033662114e+01, 2.0196661010469668e+01, 1.4624345260746775e+01, 1.8252877036565582e+01, 1.9213516388393590e+01]
    mom2_gold = [9.9498743710661458e+01, 9.9498743710661458e+01, 1.8293875644665363e+01, 0, 1.8171461287706194e+01, 1.8290280409536777e+01, 9.9498743710661458e+01, 1.8296158084407839e+01, 1.8302225113051392e+01, 1.7625689778519135e+01, 1.8232610902948014e+01, 1.8203665470231048e+01]
    if np.allclose(mom1, mom1_gold) and np.allclose(mom2_norm, mom2_gold):
        print("Success, moments matched expected vectors")
        sys.exit(0)
    else:
        print("Failure, moments did not match")
        sys.exit(1)

if __name__ == "__main__":
    main()


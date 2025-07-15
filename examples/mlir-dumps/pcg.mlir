#sparse = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed), posWidth = 32, crdWidth = 32 }>
#idmap = affine_map<(d0) -> (d0)>
module {
  func.func private @spmv(%A: tensor<?x?xf64, #sparse>, %x: tensor<?xf64>, %ydst: tensor<?xf64>) -> tensor<?xf64> {
    %y = linalg.matvec ins(%A, %x: tensor<?x?xf64, #sparse>, tensor<?xf64>) outs(%ydst : tensor<?xf64>) -> tensor<?xf64>
    return %y : tensor<?xf64>
  }

  func.func private @dot(%x: tensor<?xf64>, %y: tensor<?xf64>) -> f64 {
    %0 = tensor.empty() : tensor<f64>
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

  // CG solve with diagonal preconditioner
  // Returns: x, numiter, resnorm
  func.func @main(%A: tensor<?x?xf64, #sparse>, %b: tensor<?xf64>, %dinv: tensor<?xf64>, %tol: f64, %maxiter: index) -> (tensor<?xf64>, index, f64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %n = tensor.dim %b, %c0 : tensor<?xf64>
    %f0 = arith.constant 0.0 : f64
    %f1 = arith.constant 1.0 : f64
    %fm1 = arith.constant -1.0 : f64

    // Preallocate some intermediate tensors for dst-passing style
    %buf0 = tensor.empty(%n) : tensor<?xf64>
    %buf1 = tensor.empty(%n) : tensor<?xf64>
    %buf2 = tensor.empty(%n) : tensor<?xf64>

    // Assume initial guess x0 = 0
    // Then r0 = b - A*x0 = b
    %r0 = linalg.copy ins(%b : tensor<?xf64>) outs(%buf0 : tensor<?xf64>) -> tensor<?xf64>
    %z0 = func.call @mult(%r0, %dinv, %buf1) : (tensor<?xf64>, tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
    %p0 = linalg.copy ins(%z0 : tensor<?xf64>) outs(%buf2 : tensor<?xf64>) -> tensor<?xf64>
    %x0 = tensor.splat %f0[%n] : tensor<?xf64>
    %Apbuf = tensor.empty(%n) : tensor<?xf64>
    %rr0 = func.call @dot(%r0, %r0) : (tensor<?xf64>, tensor<?xf64>) -> f64
    %initres = math.sqrt %rr0 : f64

    %x, %p, %z, %r, %final_relres, %rz, %iters = scf.while (%xiter = %x0, %piter = %p0, %ziter = %z0, %riter = %r0, %rziter = %f0, %i = %c1) : (tensor<?xf64>, tensor<?xf64>, tensor<?xf64>, tensor<?xf64>, f64, index) -> (tensor<?xf64>, tensor<?xf64>, tensor<?xf64>, tensor<?xf64>, f64, f64, index)
    {
      %Ap = func.call @spmv(%A, %piter, %Apbuf) { fuse_with = "dot" } : (tensor<?x?xf64, #sparse>, tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
      %pAp = func.call @dot(%Ap, %piter) { fuse_with = "spmv" } : (tensor<?xf64>, tensor<?xf64>) -> f64
      %rz = func.call @dot(%riter, %ziter) : (tensor<?xf64>, tensor<?xf64>) -> f64
      %alpha = arith.divf %rz, %pAp : f64
      %malpha = arith.negf %alpha : f64

      // Update x and r
      %xnext = func.call @axpby(%f1, %xiter, %alpha, %piter, %xiter) { fuse_with = "axpby" } : (f64, tensor<?xf64>, f64, tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
      %rnext = func.call @axpby(%f1, %riter, %malpha, %Ap, %riter) { fuse_with = "axpby" } : (f64, tensor<?xf64>, f64, tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>

      // Test against tolerance and 
      %rr = func.call @dot(%rnext, %rnext) : (tensor<?xf64>, tensor<?xf64>) -> f64
      %rnorm = math.sqrt %rr : f64
      %relres = arith.divf %rnorm, %initres : f64
      %not_converged = arith.cmpf ogt, %relres, %tol : f64

      // we have already completed an iteration, which is why i is intially 1
      %below_maxiter = arith.cmpi ne, %i, %maxiter : index
      %continue = arith.andi %not_converged, %below_maxiter : i1

      scf.condition(%continue) %xnext, %piter, %ziter, %rnext, %relres, %rz, %i: tensor<?xf64>, tensor<?xf64>, tensor<?xf64>, tensor<?xf64>, f64, f64, index
    } 
    do {
    ^bb0(%xiter : tensor<?xf64>, %piter : tensor<?xf64>, %ziter : tensor<?xf64>, %riter : tensor<?xf64>, %unused : f64, %rziter : f64, %i : index):
      %znext = func.call @mult(%riter, %dinv, %ziter) : (tensor<?xf64>, tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
      %rznext = func.call @dot(%riter, %znext) : (tensor<?xf64>, tensor<?xf64>) -> f64
      %beta = arith.divf %rznext, %rziter : f64
      %pnext = func.call @axpby(%f1, %znext, %beta, %piter, %piter) : (f64, tensor<?xf64>, f64, tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
      %inext = arith.addi %i, %c1 : index
      scf.yield %xiter, %pnext, %znext, %riter, %rznext, %inext : tensor<?xf64>, tensor<?xf64>, tensor<?xf64>, tensor<?xf64>, f64, index
    }
    return %x, %iters, %final_relres : tensor<?xf64>, index, f64
  }
}


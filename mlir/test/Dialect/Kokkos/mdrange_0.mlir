// RUN: %lapis-opt %s --kokkos-mdrange-iteration | diff %s.gold -
module {
  func.func @myfunc(%arg0: memref<?x?x?xf64>, %arg1: memref<?xindex>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: index) {
    return
  }
}
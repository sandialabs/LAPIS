// RUN: %lapis-opt %s --kokkos-loop-mapping | diff %s.gold -

// Simple 1D (RangePolicy) offloadable parallel reduce,
module {
  func.func @sum(%arg0: index) -> index {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = scf.parallel (%arg1) = (%c0) to (%arg0) step (%c1) init (%c0) -> index {
      scf.reduce(%c1 : index) {
      ^bb0(%arg2: index, %arg3: index):
        %1 = arith.addi %arg2, %arg3 : index
        scf.reduce.return %1 : index
      }
    }
    return %0 : index
  }
}


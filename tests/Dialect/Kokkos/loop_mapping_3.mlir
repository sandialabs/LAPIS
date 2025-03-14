// RUN: %lapis-opt %s --kokkos-loop-mapping | diff %s.gold -

// Simple 4D  nested parallel for, for
// testing hierarchical parallelism.
// Loops should be mapped to Team, Thread, sequential, Vector.

// Take a 4D input memref, as well as 3D, 2D and 1D output memrefs.
// Each loop contributes the result of loop nested inside it (or just the memref).
// Function then returns the top loop's result.

module {
  func.func @loop3d(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?xf32>, %arg3: memref<?xf32>) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %cst = arith.constant 0.000000e+00 : f32
    %dim = memref.dim %arg0, %c0 : memref<?x?x?x?xf32>
    %dim_0 = memref.dim %arg0, %c1 : memref<?x?x?x?xf32>
    %dim_1 = memref.dim %arg0, %c2 : memref<?x?x?x?xf32>
    %dim_2 = memref.dim %arg0, %c3 : memref<?x?x?x?xf32>
    %0 = scf.parallel (%arg4) = (%c0) to (%dim) step (%c1) init (%cst) -> f32 {
      %1 = scf.parallel (%arg5) = (%c0) to (%dim_0) step (%c1) init (%cst) -> f32 {
        %2 = scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) init (%cst) -> f32 {
          %3 = scf.parallel (%arg7) = (%c0) to (%dim_2) step (%c1) init (%cst) -> f32 {
            %4 = memref.load %arg0[%arg4, %arg5, %arg6, %arg7] : memref<?x?x?x?xf32>
            scf.reduce(%4 : f32) {
            ^bb0(%arg8: f32, %arg9: f32):
              %5 = arith.addf %arg8, %arg9 : f32
              scf.reduce.return %5 : f32
            }
          }
          memref.store %3, %arg1[%arg4, %arg5, %arg6] : memref<?x?x?xf32>
          scf.reduce(%3 : f32) {
          ^bb0(%arg7: f32, %arg8: f32):
            %4 = arith.addf %arg7, %arg8 : f32
            scf.reduce.return %4 : f32
          }
        }
        memref.store %2, %arg2[%arg4, %arg5] : memref<?x?xf32>
        scf.reduce(%2 : f32) {
        ^bb0(%arg6: f32, %arg7: f32):
          %3 = arith.addf %arg6, %arg7 : f32
          scf.reduce.return %3 : f32
        }
      }
      memref.store %1, %arg3[%arg4] : memref<?xf32>
      scf.reduce(%1 : f32) {
      ^bb0(%arg5: f32, %arg6: f32):
        %2 = arith.addf %arg5, %arg6 : f32
        scf.reduce.return %2 : f32
      }
    }
    return %0 : f32
  }
}


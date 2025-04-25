lapis-opt --sparse-compiler-kokkos=decompose-sparse-tensors batched_gemm.mlir | lapis-translate -o batched_gemm.hpp

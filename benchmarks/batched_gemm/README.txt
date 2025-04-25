batched_gemm.mlir contains a high-level MLIR program that computes a batched matmul,
using the linalg.batch_matmul operation.

The script runLAPIS.sh will apply the LAPIS pipeline to this, generating the
file batched_gemm.hpp.

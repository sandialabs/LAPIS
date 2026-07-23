# Algebraic kernel-fusion benchmarks

This harness builds two versions of each input MLIR module:

- `baseline`: lowered with `--sparse-compiler-kokkos`;
- `optimized`: lowered through the same pipeline with its
  `algebraic-kernel-fusion` option enabled.

Both variants use the same native C++ driver, deterministic inputs, correctness
check, warmup count, timed iteration count, and a `Kokkos::fence` around every
sample. The benchmark reports minimum, median, and mean wall-clock time. The
summary speedup is the baseline median divided by the optimized median.

The harness requires `lapis-opt`, `lapis-translate`, and CMake on `PATH`, plus
the same `KOKKOS_ROOT` and `SUPPORT_LIB` variables used by LAPIS's numerical
tests. `KOKKOS_ROOT` may name either an installed prefix containing
`lib/cmake/Kokkos/KokkosConfig.cmake` or a Kokkos build directory containing
`KokkosConfig.cmake` directly. Set `CXX` to the compiler required by that
Kokkos configuration, such as `nvcc_wrapper` for CUDA.

```sh
python benchmarks/algebraic_kernel_fusion/run.py \
  --case abx --warmup 3 --iterations 20

# Run every discovered case and retain the raw measurements.
python benchmarks/algebraic_kernel_fusion/run.py \
  --case all --warmup 3 --iterations 20 --output results.csv
```

Build products are written under `benchmarks/algebraic_kernel_fusion/build`,
which is ignored by the repository's existing `build*/` rule.

Each case consists of `cases/<name>.mlir` and `cases/<name>.cpp`. Adding a case
does not require changing the harness: the MLIR file is discovered
automatically, and the C++ driver supplies that function's typed inputs and
validation logic.

The current cases are:

| Case | Source expression | Principal optimization |
| --- | --- | --- |
| ABx | `(A B) x` | Reorder to `A (B x)` and avoid the `A B` matrix. |
| Large ABx | `(A B) x` | Larger profile for distinguishing computation from fixed overhead. |
| Linear attention | `(Q K^T) V` | Reorder to `Q (K^T V)` and avoid the score matrix. |
| Large linear attention | `(Q K^T) V` | Larger sequence-length profile. |
| Batched linear attention | Batched `(Q K^T) V` | Exercise batch-index liveness and backend mapping. |
| PCG | One dense preconditioned-CG iteration | Exercise equal-work fusion selection and externally visible intermediates. |
| Burgers | 20 SSP-RK2 steps for a 128x128 2D scalar viscous Burgers solve | Fuse three derivative primitives, RHS formation, and Euler update without materializing derivative fields. |

Each timed sample is one complete generated function call. It therefore
includes output and private-intermediate allocation, zero initialization,
kernel dispatch, and synchronization. This measures end-to-end kernel-pipeline
latency rather than an isolated inner loop. Results record the configured
Kokkos execution space and should only be compared within the same build and
machine.

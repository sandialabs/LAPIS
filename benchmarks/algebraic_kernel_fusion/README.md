# Algebraic kernel-fusion benchmarks

This harness builds two versions of each input MLIR module:

- `baseline`: lowered directly with `--sparse-compiler-kokkos`;
- `optimized`: first transformed with `--algebraic-kernel-fusion`, then
  lowered through the same Kokkos pipeline.

Both variants use the same native C++ driver, deterministic inputs, correctness
check, warmup count, timed iteration count, and a `Kokkos::fence` around every
sample. The benchmark reports minimum, median, and mean wall-clock time. The
summary speedup is the baseline median divided by the optimized median.

The harness requires `lapis-opt`, `lapis-translate`, and CMake on `PATH`, plus
the same `KOKKOS_ROOT` and `SUPPORT_LIB` variables used by LAPIS's numerical
tests.

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
| Linear attention | `(Q K^T) V` | Reorder to `Q (K^T V)` and avoid the score matrix. |
| PCG | One dense preconditioned-CG iteration | Exercise equal-work fusion selection and externally visible intermediates. |

Each timed sample is one complete generated function call. It therefore
includes output and private-intermediate allocation, zero initialization,
kernel dispatch, and synchronization. This measures end-to-end kernel-pipeline
latency rather than an isolated inner loop. Results record the configured
Kokkos execution space and should only be compared within the same build and
machine.

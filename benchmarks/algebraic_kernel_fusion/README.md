# Algebraic kernel-fusion benchmarks

This harness builds two versions of each input MLIR module:

- `baseline`: lowered with `--sparse-compiler-kokkos`;
- `optimized`: lowered through the same pipeline with its
  `algebraic-kernel-fusion` option enabled.

Both variants use the same native C++ driver, deterministic inputs, correctness
check, warmup count, timed iteration count, and a `Kokkos::fence` around every
sample. The benchmark reports minimum, median, and mean wall-clock time. The
summary speedup is the baseline median divided by the optimized median.

The harness lowers every selected case, creates one CMake project, and builds
all baseline/optimized executables in a single configuration. The selected
Kokkos package determines the backend; there is no backend-specific compilation
logic in the harness.

## Quick start

The harness requires `lapis-opt`, `lapis-translate`, and CMake on `PATH`, plus
the same `KOKKOS_ROOT` and `SUPPORT_LIB` variables used by LAPIS's numerical
tests. From the repository root, source the workspace environment before
returning to the repository:

```sh
cd ..
source ./lapis-env.sh
cd LAPIS

python benchmarks/algebraic_kernel_fusion/run.py \
  --case abx --warmup 3 --iterations 20 \
  --label workstation-serial --output serial.csv

# Run every discovered case and retain the raw measurements.
python benchmarks/algebraic_kernel_fusion/run.py \
  --case all --warmup 3 --iterations 20 \
  --label workstation --output results.csv
```

Command-line arguments override the environment:

```sh
python benchmarks/algebraic_kernel_fusion/run.py \
  --case all \
  --lapis-opt /path/to/lapis-opt \
  --lapis-translate /path/to/lapis-translate \
  --kokkos-root /path/to/kokkos/install-or-build \
  --support-lib /path/to/libmlir_c_runner_utils.so \
  --cxx /path/to/compiler-or-wrapper \
  --output results.csv
```

`KOKKOS_ROOT`/`--kokkos-root` may name an installed prefix containing
`lib/cmake/Kokkos/KokkosConfig.cmake`, a Kokkos build directory containing
`KokkosConfig.cmake`, or the package-config directory itself. The C++ compiler
must be compatible with that Kokkos installation. Use `--cxx` to avoid relying
on CMake's ambient compiler selection. Extra configuration options may be
passed with repeated `--cmake-arg=...` arguments.

Build products are written under `benchmarks/algebraic_kernel_fusion/build`,
which is ignored by the repository's existing `build*/` rule.

Each case consists of `cases/<name>.mlir` and `cases/<name>.cpp`. Adding a case
does not require changing the harness: the MLIR file is discovered
automatically, and the C++ driver supplies that function's typed inputs and
validation logic.

Use `--list-cases` to inspect the suite without requiring a LAPIS or Kokkos
installation. `--expect-backend` provides a useful guard against accidentally
benchmarking a Serial build when a GPU build was intended.

## Backend examples

These commands assume that LAPIS tools and `SUPPORT_LIB` are already available.
The exact compiler path depends on how the corresponding Kokkos package was
built.

OpenMP:

```sh
OMP_NUM_THREADS=16 OMP_PROC_BIND=spread OMP_PLACES=cores \
python benchmarks/algebraic_kernel_fusion/run.py \
  --case all --kokkos-root /opt/kokkos-openmp \
  --cxx /usr/bin/g++ --expect-backend OpenMP \
  --label cpu-openmp-16t --output openmp.csv
```

CUDA:

```sh
CUDA_VISIBLE_DEVICES=0 \
python benchmarks/algebraic_kernel_fusion/run.py \
  --case all --kokkos-root /opt/kokkos-cuda \
  --cxx /opt/kokkos/bin/nvcc_wrapper --expect-backend Cuda \
  --label nvidia-gpu0 --output cuda.csv
```

HIP:

```sh
ROCR_VISIBLE_DEVICES=0 \
python benchmarks/algebraic_kernel_fusion/run.py \
  --case all --kokkos-root /opt/kokkos-hip \
  --cxx /opt/rocm/bin/hipcc --expect-backend HIP \
  --label amd-gpu0 --output hip.csv
```

Intel GPU through Kokkos SYCL:

```sh
source /opt/intel/oneapi/setvars.sh
ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
python benchmarks/algebraic_kernel_fusion/run.py \
  --case all --kokkos-root /opt/kokkos-sycl \
  --cxx "$(command -v icpx)" --expect-backend SYCL \
  --label intel-gpu --output sycl.csv
```

CUDA, HIP, SYCL, OpenMP, and device-selection environment variables are passed
through unchanged. The result CSV records the runtime execution-space name,
Kokkos version/devices/architecture, CMake and Kokkos compilers, LAPIS Git
revision, host platform, run label, and relevant runtime environment. Use
`--notes` for allocation-specific details and `--append-output` to add runs to
an existing CSV with the same schema.

## Plotting

`plot.py` reads one or more result CSVs directly. It pairs baseline and
optimized measurements from the same run, then produces a speedup comparison
and a baseline-versus-optimized execution-time comparison:

```sh
/home/aj/miniforge3/envs/dev/bin/python \
  benchmarks/algebraic_kernel_fusion/plot.py \
  cuda.csv openmp.csv hip.csv sycl.csv \
  --output-prefix plots/algebraic-fusion \
  --format png --format pdf
```

This writes `plots/algebraic-fusion-speedup.{png,pdf}` and
`plots/algebraic-fusion-time.{png,pdf}`. Execution time is plotted in
milliseconds on a log scale. Speedup is baseline median divided by optimized
median and includes a 1x reference line.

Rows sharing a case and configuration but carrying different timestamps are
treated as repeated runs. Their plotted center is the median and their whiskers
span the observed minimum and maximum. Give each machine/backend configuration
a distinct `run.py --label`; the remaining system metadata prevents unlike
Kokkos builds from being combined accidentally.

Use repeated `--case` options to select and order cases, `--metric speedup` or
`--metric time` for one figure, `--speedup-scale log` when regressions and gains
span a wide range, and `--format svg` for editable vector output. Run
`plot.py --help` for the full interface.

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

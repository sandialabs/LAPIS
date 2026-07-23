#!/usr/bin/env python3

import argparse
import csv
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
CASES = ROOT / "cases"
REPOSITORY = ROOT.parents[1]
VARIANTS = ("baseline", "optimized")
CAPTURED_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OMP_PROC_BIND",
    "OMP_PLACES",
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "ONEAPI_DEVICE_SELECTOR",
    "SYCL_DEVICE_FILTER",
    "ZE_AFFINITY_MASK",
)


class MissingPrerequisite(RuntimeError):
    """A missing optional runtime dependency, as opposed to a test failure."""


@dataclass(frozen=True)
class Result:
    case: str
    variant: str
    backend: str
    warmup: int
    iterations: int
    minimum_seconds: float
    median_seconds: float
    mean_seconds: float
    checksum: float


@dataclass(frozen=True)
class RunMetadata:
    timestamp_utc: str
    label: str
    hostname: str
    operating_system: str
    architecture: str
    lapis_revision: str
    cmake_cxx_compiler: str
    cmake_cxx_compiler_id: str
    cmake_cxx_compiler_version: str
    kokkos_version: str
    kokkos_devices: str
    kokkos_arch: str
    kokkos_cxx_compiler: str
    kokkos_cxx_compiler_id: str
    kokkos_cxx_compiler_version: str
    kokkos_config: str
    runtime_environment: str
    notes: str


def discover_cases():
    mlir_cases = {path.stem for path in CASES.glob("*.mlir")}
    driver_cases = {path.stem for path in CASES.glob("*.cpp")}
    incomplete = sorted(mlir_cases ^ driver_cases)
    if incomplete:
        raise RuntimeError(
            "benchmark cases require matching .mlir and .cpp files: "
            + ", ".join(incomplete)
        )
    return sorted(mlir_cases)


def parse_args():
    try:
        available = discover_cases()
    except RuntimeError as error:
        raise SystemExit(str(error)) from error

    parser = argparse.ArgumentParser(
        description="Build and run paired LAPIS algebraic-fusion benchmarks"
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=available + ["all"],
        help="benchmark to run; repeat the option or use 'all'",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="list available benchmark cases and exit",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--build-jobs", type=int, default=2)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument(
        "--output", type=Path, help="optional CSV file for measurements"
    )
    parser.add_argument(
        "--append-output",
        action="store_true",
        help="append to an existing CSV after verifying its schema",
    )
    parser.add_argument(
        "--label",
        default=platform.node() or "unknown",
        help="human-readable machine or allocation label stored in the CSV",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="free-form run notes stored in the CSV",
    )
    parser.add_argument("--lapis-opt", default=shutil.which("lapis-opt"))
    parser.add_argument("--lapis-translate", default=shutil.which("lapis-translate"))
    parser.add_argument("--cmake", default=shutil.which("cmake"))
    parser.add_argument(
        "--cxx",
        default=os.environ.get("CXX"),
        help="C++ compiler or Kokkos compiler wrapper passed explicitly to CMake",
    )
    parser.add_argument(
        "--kokkos-root",
        type=Path,
        default=Path(os.environ["KOKKOS_ROOT"])
        if os.environ.get("KOKKOS_ROOT")
        else None,
        help="Kokkos installation, build, or package-config directory",
    )
    parser.add_argument(
        "--support-lib",
        type=Path,
        default=Path(os.environ["SUPPORT_LIB"])
        if os.environ.get("SUPPORT_LIB")
        else None,
        help="MLIR C runner support library used by generated modules",
    )
    parser.add_argument(
        "--cmake-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="extra CMake configure argument; repeat as needed",
    )
    parser.add_argument(
        "--expect-backend",
        help="fail unless the runtime execution-space name matches this value",
    )
    parser.add_argument(
        "--skip-if-unavailable",
        action="store_true",
        help="return CTest's skip code when optional prerequisites are missing",
    )
    args = parser.parse_args()
    if args.list_cases:
        args.cases = available
        return args
    if args.warmup <= 0 or args.iterations <= 0 or args.build_jobs <= 0:
        parser.error("--warmup, --iterations, and --build-jobs must be positive")
    if args.append_output and not args.output:
        parser.error("--append-output requires --output")
    selected = args.case or [available[0]]
    args.cases = available if "all" in selected else list(dict.fromkeys(selected))
    return args


def run_checked(command, *, input_text=None, cwd=None, env=None, stream_output=False):
    rendered = shlex.join(str(argument) for argument in command)
    print(f"+ {rendered}", flush=True)
    completed = subprocess.run(
        [str(argument) for argument in command],
        input=input_text,
        text=True,
        stdout=None if stream_output else subprocess.PIPE,
        stderr=None if stream_output else subprocess.PIPE,
        cwd=cwd,
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        output = ""
        if not stream_output:
            output = ":\n" + completed.stdout + completed.stderr
        raise RuntimeError(f"command failed ({rendered}){output}")
    return completed.stdout or ""


def resolve_executable(value, description):
    if not value:
        raise MissingPrerequisite(f"{description} was not provided or found")
    resolved = shutil.which(str(value))
    if not resolved:
        candidate = Path(value).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            resolved = str(candidate.resolve())
    if not resolved:
        raise MissingPrerequisite(f"{description} is not executable: {value}")
    return resolved


def find_kokkos_config(root):
    candidates = [
        root,
        root / "lib" / "cmake" / "Kokkos",
        root / "lib64" / "cmake" / "Kokkos",
        root / "share" / "cmake" / "Kokkos",
    ]
    for candidate in candidates:
        if (candidate / "KokkosConfig.cmake").is_file():
            return candidate.resolve()
    raise MissingPrerequisite(f"KokkosConfig.cmake not found under {root}")


def require_runtime(args):
    missing = []
    for attribute, description in (
        ("lapis_opt", "lapis-opt"),
        ("lapis_translate", "lapis-translate"),
        ("cmake", "CMake"),
    ):
        try:
            setattr(
                args,
                attribute,
                resolve_executable(getattr(args, attribute), description),
            )
        except MissingPrerequisite as error:
            missing.append(str(error))

    if args.cxx:
        try:
            args.cxx = resolve_executable(args.cxx, "C++ compiler")
        except MissingPrerequisite as error:
            missing.append(str(error))

    if not args.kokkos_root:
        missing.append("Kokkos root was not provided with --kokkos-root or KOKKOS_ROOT")
    else:
        try:
            args.kokkos_config = find_kokkos_config(
                args.kokkos_root.expanduser().resolve()
            )
        except MissingPrerequisite as error:
            missing.append(str(error))

    if not args.support_lib:
        missing.append(
            "support library was not provided with --support-lib or SUPPORT_LIB"
        )
    else:
        args.support_lib = args.support_lib.expanduser().resolve()
        if not args.support_lib.is_file():
            missing.append(f"support library does not exist: {args.support_lib}")

    if missing:
        raise MissingPrerequisite("; ".join(missing))


def lower_module(args, case, variant, source, generated_directory):
    pipeline_options = [
        "parallelization-strategy=any-storage-any-loop",
        "decompose-sparse-tensors",
    ]
    if variant == "optimized":
        pipeline_options.append("algebraic-kernel-fusion")
    pipeline = "--sparse-compiler-kokkos=" + " ".join(pipeline_options)
    lowered = run_checked([args.lapis_opt, pipeline], input_text=source)
    module = generated_directory / f"{case}_{variant}_module.cpp"
    wrapper = generated_directory / f"{case}_{variant}_wrapper.py"
    run_checked(
        [
            args.lapis_translate,
            "-o",
            module,
            f"--py={wrapper}",
            "--finalize",
        ],
        input_text=lowered,
    )
    return module


def write_build_project(args, project_directory, modules_by_case):
    targets = []
    for case, modules in modules_by_case.items():
        driver = (CASES / f"{case}.cpp").resolve()
        for variant, module in modules.items():
            target = f"{case}_{variant}"
            translation_unit = project_directory / f"{target}.cpp"
            translation_unit.write_text(
                f'#define LAPIS_BENCHMARK_MODULE "{module.resolve()}"\n'
                f'#define LAPIS_BENCHMARK_VARIANT "{variant}"\n'
                f'#include "{driver}"\n'
            )
            targets.append(
                f"add_executable({target} {translation_unit.name})\n"
                f"target_link_libraries({target} PRIVATE Kokkos::kokkos "
                f'"{args.support_lib}")\n'
            )

    cmake = f"""\
cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(LAPISAlgebraicFusionBenchmark LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(Kokkos REQUIRED CONFIG PATHS "{args.kokkos_config}" NO_DEFAULT_PATH)
file(WRITE "${{CMAKE_BINARY_DIR}}/lapis-benchmark-build-metadata.txt"
  "cmake_cxx_compiler=${{CMAKE_CXX_COMPILER}}\\n"
  "cmake_cxx_compiler_id=${{CMAKE_CXX_COMPILER_ID}}\\n"
  "cmake_cxx_compiler_version=${{CMAKE_CXX_COMPILER_VERSION}}\\n"
  "kokkos_version=${{Kokkos_VERSION}}\\n"
  "kokkos_devices=${{Kokkos_DEVICES}}\\n"
  "kokkos_arch=${{Kokkos_ARCH}}\\n"
  "kokkos_cxx_compiler=${{Kokkos_CXX_COMPILER}}\\n"
  "kokkos_cxx_compiler_id=${{Kokkos_CXX_COMPILER_ID}}\\n"
  "kokkos_cxx_compiler_version=${{Kokkos_CXX_COMPILER_VERSION}}\\n")
{"".join(targets)}
"""
    (project_directory / "CMakeLists.txt").write_text(cmake)


def parse_result(output):
    lines = [line for line in output.splitlines() if line.startswith("RESULT,")]
    if len(lines) != 1:
        raise RuntimeError("expected one RESULT line, got:\n" + output)
    values = next(csv.reader(lines))
    if len(values) != 10:
        raise RuntimeError(f"expected 10 fields in RESULT line, found {len(values)}")
    try:
        return Result(
            case=values[1],
            variant=values[2],
            backend=values[3],
            warmup=int(values[4]),
            iterations=int(values[5]),
            minimum_seconds=float(values[6]),
            median_seconds=float(values[7]),
            mean_seconds=float(values[8]),
            checksum=float(values[9]),
        )
    except ValueError as error:
        raise RuntimeError(f"invalid RESULT line: {lines[0]}") from error


def reset_managed_directory(path):
    marker_name = ".lapis-algebraic-benchmark-directory"
    marker = path / marker_name
    if path.exists():
        if not marker.is_file():
            raise RuntimeError(
                f"refusing to replace unmanaged benchmark directory: {path}"
            )
        shutil.rmtree(path)
    path.mkdir(parents=True)
    (path / marker_name).write_text(
        "Managed by benchmarks/algebraic_kernel_fusion/run.py.\n"
    )


def read_build_metadata(binary_directory):
    metadata_file = binary_directory / "lapis-benchmark-build-metadata.txt"
    if not metadata_file.is_file():
        raise RuntimeError(f"CMake did not produce build metadata: {metadata_file}")
    metadata = {}
    for line in metadata_file.read_text().splitlines():
        key, separator, value = line.partition("=")
        if not separator:
            raise RuntimeError(f"malformed build metadata line: {line}")
        metadata[key] = value
    return metadata


def get_lapis_revision():
    completed = subprocess.run(
        ["git", "-C", str(REPOSITORY), "describe", "--always", "--dirty"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def make_run_metadata(args, build_metadata):
    def value(key):
        return build_metadata.get(key) or "unknown"

    captured_environment = {
        name: os.environ[name] for name in CAPTURED_ENVIRONMENT if os.environ.get(name)
    }
    return RunMetadata(
        timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        label=args.label,
        hostname=platform.node() or "unknown",
        operating_system=platform.platform(),
        architecture=platform.machine() or "unknown",
        lapis_revision=get_lapis_revision(),
        cmake_cxx_compiler=value("cmake_cxx_compiler"),
        cmake_cxx_compiler_id=value("cmake_cxx_compiler_id"),
        cmake_cxx_compiler_version=value("cmake_cxx_compiler_version"),
        kokkos_version=value("kokkos_version"),
        kokkos_devices=value("kokkos_devices"),
        kokkos_arch=value("kokkos_arch"),
        kokkos_cxx_compiler=value("kokkos_cxx_compiler"),
        kokkos_cxx_compiler_id=value("kokkos_cxx_compiler_id"),
        kokkos_cxx_compiler_version=value("kokkos_cxx_compiler_version"),
        kokkos_config=str(args.kokkos_config),
        runtime_environment=json.dumps(captured_environment, sort_keys=True),
        notes=args.notes,
    )


def print_run_metadata(metadata):
    print("Benchmark configuration:")
    print(f"  label: {metadata.label}")
    print(f"  LAPIS revision: {metadata.lapis_revision}")
    print(
        "  C++ compiler: "
        f"{metadata.cmake_cxx_compiler_id} "
        f"{metadata.cmake_cxx_compiler_version} "
        f"({metadata.cmake_cxx_compiler})"
    )
    print(
        f"  Kokkos: {metadata.kokkos_version}; "
        f"devices={metadata.kokkos_devices}; arch={metadata.kokkos_arch}"
    )
    print(f"  Kokkos package: {metadata.kokkos_config}")
    print(f"  runtime environment: {metadata.runtime_environment}")


def prepare_benchmarks(args):
    build_directory = args.build_dir.expanduser().resolve()
    build_directory.mkdir(parents=True, exist_ok=True)
    project_directory = build_directory / "generated-project"
    binary_directory = build_directory / "cmake-build"
    reset_managed_directory(project_directory)
    reset_managed_directory(binary_directory)

    generated_directory = project_directory / "generated"
    generated_directory.mkdir()
    modules_by_case = {}
    for case in args.cases:
        source = (CASES / f"{case}.mlir").read_text()
        modules_by_case[case] = {
            variant: lower_module(args, case, variant, source, generated_directory)
            for variant in VARIANTS
        }
    write_build_project(args, project_directory, modules_by_case)

    configure_command = [
        args.cmake,
        "-S",
        project_directory,
        "-B",
        binary_directory,
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    if args.cxx:
        configure_command.append(f"-DCMAKE_CXX_COMPILER={args.cxx}")
    configure_command.extend(args.cmake_arg)
    configure_environment = os.environ.copy()
    # The explicitly resolved package directory is the source of truth. Avoid
    # CMake policy warnings and stale package-root variables when a command-line
    # --kokkos-root overrides the ambient environment.
    configure_environment.pop("KOKKOS_ROOT", None)
    configure_environment.pop("Kokkos_ROOT", None)
    run_checked(
        configure_command,
        env=configure_environment,
        stream_output=True,
    )
    run_checked(
        [
            args.cmake,
            "--build",
            binary_directory,
            "--parallel",
            args.build_jobs,
        ],
        stream_output=True,
    )
    return binary_directory, read_build_metadata(binary_directory)


def make_runtime_environment(args):
    runtime_environment = os.environ.copy()
    runtime_library_paths = [str(args.support_lib.parent)]
    existing_library_path = runtime_environment.get("LD_LIBRARY_PATH")
    if existing_library_path:
        runtime_library_paths.append(existing_library_path)
    runtime_environment["LD_LIBRARY_PATH"] = os.pathsep.join(runtime_library_paths)
    return runtime_environment


def run_case(args, case, binary_directory, runtime_environment):
    results = []
    for variant in VARIANTS:
        output = run_checked(
            [
                binary_directory / f"{case}_{variant}",
                "--warmup",
                args.warmup,
                "--iterations",
                args.iterations,
            ],
            env=runtime_environment,
        )
        result = parse_result(output)
        if result.case != case or result.variant != variant:
            raise RuntimeError(
                f"{case}_{variant} reported itself as {result.case}_{result.variant}"
            )
        if result.warmup != args.warmup or result.iterations != args.iterations:
            raise RuntimeError(f"{case}_{variant} reported inconsistent run counts")
        if (
            args.expect_backend
            and result.backend.casefold() != args.expect_backend.casefold()
        ):
            raise RuntimeError(
                f"expected backend {args.expect_backend}, "
                f"but {case}_{variant} used {result.backend}"
            )
        print(output, end="")
        results.append(result)

    baseline, optimized = results
    checksum_scale = max(1.0, abs(baseline.checksum), abs(optimized.checksum))
    if abs(baseline.checksum - optimized.checksum) > 1.0e-4 * checksum_scale:
        raise RuntimeError(
            f"{case} checksum mismatch between baseline and optimized variants"
        )
    speedup = baseline.median_seconds / optimized.median_seconds
    print(f"SUMMARY,{case},median_speedup,{speedup:.6f}")
    return results


def write_results(path, append, metadata, results):
    metadata_fields = [field.name for field in fields(RunMetadata)]
    result_fields = [field.name for field in fields(Result)]
    field_names = metadata_fields + result_fields
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    has_existing_data = path.is_file() and path.stat().st_size != 0
    if append and has_existing_data:
        with path.open(newline="") as existing:
            existing_header = next(csv.reader(existing), [])
        if existing_header != field_names:
            raise RuntimeError(f"cannot append to {path}: existing CSV schema differs")

    mode = "a" if append and has_existing_data else "w"
    with path.open(mode, newline="") as output:
        writer = csv.DictWriter(output, fieldnames=field_names)
        if mode == "w":
            writer.writeheader()
        metadata_values = asdict(metadata)
        for result in results:
            writer.writerow(metadata_values | asdict(result))
    print(f"WROTE,{path}")


def main():
    args = parse_args()
    if args.list_cases:
        for case in args.cases:
            print(case)
        return 0

    try:
        require_runtime(args)
        binary_directory, build_metadata = prepare_benchmarks(args)
        metadata = make_run_metadata(args, build_metadata)
        print_run_metadata(metadata)

        runtime_environment = make_runtime_environment(args)
        wrote_results = False
        for case in args.cases:
            results = run_case(args, case, binary_directory, runtime_environment)
            if args.output:
                write_results(
                    args.output,
                    args.append_output or wrote_results,
                    metadata,
                    results,
                )
                wrote_results = True
    except MissingPrerequisite as error:
        print(f"SKIP: {error}", file=sys.stderr)
        return 77 if args.skip_if_unavailable else 1
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

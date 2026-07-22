#!/usr/bin/env python3

import argparse
import csv
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
CASES = ROOT / "cases"


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build and run paired LAPIS algebraic-fusion benchmarks"
    )
    available = sorted(path.stem for path in CASES.glob("*.mlir"))
    parser.add_argument(
        "--case",
        action="append",
        choices=available + ["all"],
        help="benchmark to run; repeat the option or use 'all'",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument(
        "--output", type=Path, help="optional CSV file for raw measurements"
    )
    parser.add_argument("--lapis-opt", default=shutil.which("lapis-opt"))
    parser.add_argument(
        "--lapis-translate", default=shutil.which("lapis-translate")
    )
    parser.add_argument(
        "--skip-if-unavailable",
        action="store_true",
        help="return CTest's skip code when optional prerequisites are missing",
    )
    args = parser.parse_args()
    if args.warmup <= 0 or args.iterations <= 0:
        parser.error("--warmup and --iterations must be positive")
    selected = args.case or [available[0]]
    args.cases = available if "all" in selected else selected
    return args


def run_checked(command, *, input_text=None, cwd=None, env=None):
    completed = subprocess.run(
        [str(argument) for argument in command],
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        rendered = " ".join(str(argument) for argument in command)
        raise RuntimeError(
            f"command failed ({rendered}):\n"
            + completed.stdout
            + completed.stderr
        )
    return completed.stdout


def require_runtime(args):
    missing = []
    if not args.lapis_opt:
        missing.append("lapis-opt")
    if not args.lapis_translate:
        missing.append("lapis-translate")
    for variable in ("KOKKOS_ROOT", "SUPPORT_LIB"):
        if not os.environ.get(variable):
            missing.append(variable)
    if missing:
        raise MissingPrerequisite(
            "missing benchmark prerequisites: " + ", ".join(missing)
        )


def lower_module(args, case, variant, source, case_build):
    pipeline_options = [
        "parallelization-strategy=any-storage-any-loop",
        "decompose-sparse-tensors",
    ]
    if variant == "optimized":
        pipeline_options.append("algebraic-kernel-fusion")
    pipeline = "--sparse-compiler-kokkos=" + " ".join(pipeline_options)
    lowered = run_checked([args.lapis_opt, pipeline], input_text=source)
    module = case_build / f"{case}_{variant}_module.cpp"
    wrapper = case_build / f"{case}_{variant}_wrapper.py"
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


def find_kokkos_config():
    root = Path(os.environ["KOKKOS_ROOT"])
    candidates = [root]
    for library_directory in ("lib", "lib64"):
        candidates.append(root / library_directory / "cmake" / "Kokkos")
    for candidate in candidates:
        if (candidate / "KokkosConfig.cmake").is_file():
            return candidate
    raise RuntimeError(f"KokkosConfig.cmake not found under {root}")


def write_build_project(case, case_build, modules):
    support = Path(os.environ["SUPPORT_LIB"]).resolve()
    if not support.is_file():
        raise RuntimeError(f"SUPPORT_LIB does not exist: {support}")

    targets = []
    driver = (CASES / f"{case}.cpp").resolve()
    for variant, module in modules.items():
        target = f"{case}_{variant}"
        translation_unit = case_build / f"{target}.cpp"
        translation_unit.write_text(
            f'#define LAPIS_BENCHMARK_MODULE "{module.resolve()}"\n'
            f'#define LAPIS_BENCHMARK_VARIANT "{variant}"\n'
            f'#include "{driver}"\n'
        )
        targets.append(
            f"add_executable({target} {translation_unit.name})\n"
            f"target_link_libraries({target} PRIVATE Kokkos::kokkos "
            f'"{support}")\n'
        )

    cmake = f"""\
cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(LAPISAlgebraicFusionBenchmark LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(Kokkos REQUIRED PATHS "{find_kokkos_config()}" NO_DEFAULT_PATH)
{"".join(targets)}
"""
    (case_build / "CMakeLists.txt").write_text(cmake)


def parse_result(output):
    lines = [line for line in output.splitlines() if line.startswith("RESULT,")]
    if len(lines) != 1:
        raise RuntimeError("expected one RESULT line, got:\n" + output)
    fields = next(csv.reader(lines))
    return Result(
        case=fields[1],
        variant=fields[2],
        backend=fields[3],
        warmup=int(fields[4]),
        iterations=int(fields[5]),
        minimum_seconds=float(fields[6]),
        median_seconds=float(fields[7]),
        mean_seconds=float(fields[8]),
        checksum=float(fields[9]),
    )


def build_and_run_case(args, case):
    case_build = args.build_dir.resolve() / case
    if case_build.exists():
        shutil.rmtree(case_build)
    case_build.mkdir(parents=True)

    source = (CASES / f"{case}.mlir").read_text()
    modules = {
        variant: lower_module(args, case, variant, source, case_build)
        for variant in ("baseline", "optimized")
    }
    write_build_project(case, case_build, modules)

    binary_directory = case_build / "cmake-build"
    run_checked(
        [
            shutil.which("cmake"),
            "-S",
            case_build,
            "-B",
            binary_directory,
            "-DCMAKE_BUILD_TYPE=Release",
        ]
    )
    run_checked(
        [
            shutil.which("cmake"),
            "--build",
            binary_directory,
            "--parallel",
            "2",
        ]
    )

    results = []
    runtime_environment = os.environ.copy()
    runtime_library_paths = [str(Path(os.environ["SUPPORT_LIB"]).resolve().parent)]
    existing_library_path = runtime_environment.get("LD_LIBRARY_PATH")
    if existing_library_path:
        runtime_library_paths.append(existing_library_path)
    runtime_environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        runtime_library_paths
    )
    for variant in ("baseline", "optimized"):
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


def main():
    args = parse_args()
    try:
        require_runtime(args)
        results = []
        for case in args.cases:
            results.extend(build_and_run_case(args, case))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", newline="") as output:
                writer = csv.writer(output)
                writer.writerow(Result.__dataclass_fields__)
                for result in results:
                    writer.writerow(vars(result).values())
    except MissingPrerequisite as error:
        print(f"SKIP: {error}", file=sys.stderr)
        return 77 if args.skip_if_unavailable else 1
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

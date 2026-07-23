#!/usr/bin/env python3

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
from statistics import median
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


VARIANTS = ("baseline", "optimized")
REQUIRED_COLUMNS = {"case", "variant", "backend", "median_seconds"}
CASE_ORDER = (
    "abx",
    "abx_large",
    "linear_attention",
    "linear_attention_large",
    "batched_linear_attention",
    "pcg",
    "burgers",
)
CASE_LABELS = {
    "abx": "ABx",
    "abx_large": "ABx\nlarge",
    "linear_attention": "Linear\nattention",
    "linear_attention_large": "Linear attention\nlarge",
    "batched_linear_attention": "Batched linear\nattention",
    "pcg": "PCG",
    "burgers": "Burgers",
}
COLORS = (
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
)


@dataclass(frozen=True)
class Configuration:
    label: str
    backend: str
    kokkos_version: str
    kokkos_arch: str
    lapis_revision: str
    runtime_environment: str
    notes: str


@dataclass(frozen=True)
class RunKey:
    run_id: str
    configuration: Configuration
    case: str
    warmup: str
    iterations: str


@dataclass(frozen=True)
class PairedResult:
    configuration: Configuration
    case: str
    baseline_seconds: float
    optimized_seconds: float

    @property
    def speedup(self):
        return self.baseline_seconds / self.optimized_seconds


@dataclass(frozen=True)
class Distribution:
    center: float
    lower_error: float
    upper_error: float
    count: int


@dataclass(frozen=True)
class Aggregate:
    baseline: Distribution
    optimized: Distribution
    speedup: Distribution


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot LAPIS algebraic-fusion benchmark CSV files"
    )
    parser.add_argument("csv", nargs="+", type=Path, help="result CSV file")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("algebraic-kernel-fusion"),
        help="path prefix for generated figures",
    )
    parser.add_argument(
        "--format",
        action="append",
        choices=("png", "pdf", "svg"),
        help="output format; repeat to generate multiple formats",
    )
    parser.add_argument(
        "--metric",
        choices=("both", "speedup", "time"),
        default="both",
    )
    parser.add_argument(
        "--case",
        action="append",
        help="case to include; repeat to set filtering and display order",
    )
    parser.add_argument(
        "--speedup-scale",
        choices=("linear", "log"),
        default="linear",
    )
    parser.add_argument(
        "--title",
        default="Algebraic kernel fusion",
        help="base figure title",
    )
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    args.formats = list(dict.fromkeys(args.format or ["png"]))
    return args


def get_field(row, name, default):
    value = row.get(name, "").strip()
    return value or default


def parse_positive_float(value, source, line_number):
    try:
        parsed = float(value)
    except ValueError as error:
        raise RuntimeError(
            f"{source}:{line_number}: invalid median_seconds value {value!r}"
        ) from error
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise RuntimeError(
            f"{source}:{line_number}: median_seconds must be finite and positive"
        )
    return parsed


def load_paired_results(paths):
    unpaired = {}
    for source in paths:
        source = source.expanduser().resolve()
        if not source.is_file():
            raise RuntimeError(f"result CSV does not exist: {source}")
        with source.open(newline="") as input_file:
            reader = csv.DictReader(input_file)
            if reader.fieldnames is None:
                raise RuntimeError(f"result CSV has no header: {source}")
            missing = REQUIRED_COLUMNS - set(reader.fieldnames)
            if missing:
                raise RuntimeError(
                    f"{source}: missing required columns: " + ", ".join(sorted(missing))
                )

            for line_number, row in enumerate(reader, start=2):
                variant = get_field(row, "variant", "")
                if variant not in VARIANTS:
                    raise RuntimeError(
                        f"{source}:{line_number}: unsupported variant {variant!r}"
                    )
                timestamp = get_field(row, "timestamp_utc", "")
                label = get_field(row, "label", source.stem)
                configuration = Configuration(
                    label=label,
                    backend=get_field(row, "backend", "unknown"),
                    kokkos_version=get_field(row, "kokkos_version", "unknown"),
                    kokkos_arch=get_field(row, "kokkos_arch", "unknown"),
                    lapis_revision=get_field(row, "lapis_revision", "unknown"),
                    runtime_environment=get_field(row, "runtime_environment", "{}"),
                    notes=get_field(row, "notes", ""),
                )
                key = RunKey(
                    run_id=timestamp or str(source),
                    configuration=configuration,
                    case=get_field(row, "case", ""),
                    warmup=get_field(row, "warmup", "unknown"),
                    iterations=get_field(row, "iterations", "unknown"),
                )
                timing = parse_positive_float(
                    row["median_seconds"], source, line_number
                )
                variants = unpaired.setdefault(key, {})
                if variant in variants:
                    if variants[variant] == timing:
                        continue
                    raise RuntimeError(
                        f"{source}:{line_number}: conflicting duplicate "
                        f"{variant} result for {key.case}"
                    )
                variants[variant] = timing

    paired = []
    for key, timings in unpaired.items():
        missing = set(VARIANTS) - set(timings)
        if missing:
            raise RuntimeError(
                f"run {key.run_id!r}, case {key.case!r} is missing "
                + ", ".join(sorted(missing))
            )
        paired.append(
            PairedResult(
                configuration=key.configuration,
                case=key.case,
                baseline_seconds=timings["baseline"],
                optimized_seconds=timings["optimized"],
            )
        )
    if not paired:
        raise RuntimeError("input CSV files contain no paired benchmark results")
    return paired


def summarize(values):
    center = median(values)
    return Distribution(
        center=center,
        lower_error=center - min(values),
        upper_error=max(values) - center,
        count=len(values),
    )


def aggregate_results(results):
    grouped = {}
    for result in results:
        values = grouped.setdefault(
            (result.configuration, result.case),
            {"baseline": [], "optimized": [], "speedup": []},
        )
        values["baseline"].append(result.baseline_seconds)
        values["optimized"].append(result.optimized_seconds)
        values["speedup"].append(result.speedup)
    return {
        key: Aggregate(
            baseline=summarize(values["baseline"]),
            optimized=summarize(values["optimized"]),
            speedup=summarize(values["speedup"]),
        )
        for key, values in grouped.items()
    }


def select_cases(results, requested):
    available = {result.case for result in results}
    if requested:
        missing = [case for case in requested if case not in available]
        if missing:
            raise RuntimeError(
                "requested cases are absent from the input: " + ", ".join(missing)
            )
        return list(dict.fromkeys(requested))
    known = [case for case in CASE_ORDER if case in available]
    unknown = sorted(available - set(CASE_ORDER))
    return known + unknown


def select_configurations(results, selected_cases):
    selected = set(selected_cases)
    return list(
        dict.fromkeys(
            result.configuration for result in results if result.case in selected
        )
    )


def configuration_labels(configurations):
    base_labels = [
        f"{configuration.label} ({configuration.backend})"
        for configuration in configurations
    ]
    counts = {label: base_labels.count(label) for label in base_labels}
    labels = {}
    for configuration, base in zip(configurations, base_labels):
        label = base
        if counts[base] > 1:
            label += (
                f"\nKokkos {configuration.kokkos_version}, "
                f"{configuration.lapis_revision}"
            )
        labels[configuration] = label
    return labels


def case_label(case):
    return CASE_LABELS.get(case, case.replace("_", " ").title())


def distribution_for(aggregates, configuration, case, metric):
    aggregate = aggregates.get((configuration, case))
    return getattr(aggregate, metric) if aggregate else None


def distribution_vectors(aggregates, configuration, cases, metric, factor=1.0):
    distributions = [
        distribution_for(aggregates, configuration, case, metric) for case in cases
    ]
    values = [
        distribution.center * factor if distribution else math.nan
        for distribution in distributions
    ]
    lower = [
        distribution.lower_error * factor if distribution else 0.0
        for distribution in distributions
    ]
    upper = [
        distribution.upper_error * factor if distribution else 0.0
        for distribution in distributions
    ]
    return values, [lower, upper]


def nonzero_errors(errors):
    return errors if any(value > 0.0 for row in errors for value in row) else None


def style_axes(axis):
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_speedup(aggregates, configurations, cases, labels, title, scale):
    figure_width = max(7.0, 1.4 * len(cases))
    figure, axis = plt.subplots(figsize=(figure_width, 4.8))
    x_positions = list(range(len(cases)))
    group_width = 0.82
    bar_width = group_width / len(configurations)

    for configuration_number, configuration in enumerate(configurations):
        offset = (configuration_number - (len(configurations) - 1) / 2.0) * bar_width
        positions = [position + offset for position in x_positions]
        values, errors = distribution_vectors(
            aggregates, configuration, cases, "speedup"
        )
        axis.bar(
            positions,
            values,
            width=bar_width * 0.92,
            yerr=nonzero_errors(errors),
            capsize=3,
            color=COLORS[configuration_number % len(COLORS)],
            label=labels[configuration],
        )

    axis.axhline(1.0, color="#333333", linewidth=1.2, linestyle="--")
    axis.set_xticks(x_positions, [case_label(case) for case in cases])
    axis.set_ylabel("Speedup (baseline / optimized)")
    axis.set_yscale(scale)
    if scale == "linear":
        axis.set_ylim(bottom=0.0)
    axis.set_title(f"{title}: speedup")
    axis.legend(frameon=False)
    style_axes(axis)
    figure.tight_layout()
    return figure


def plot_time(aggregates, configurations, cases, labels, title):
    column_count = min(2, len(configurations))
    row_count = math.ceil(len(configurations) / column_count)
    panel_width = max(6.0, 1.1 * len(cases))
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(panel_width * column_count, 4.2 * row_count),
        sharey=True,
        squeeze=False,
    )
    flattened_axes = list(axes.flat)
    x_positions = list(range(len(cases)))
    bar_width = 0.38

    for axis, configuration in zip(flattened_axes, configurations):
        baseline, baseline_errors = distribution_vectors(
            aggregates,
            configuration,
            cases,
            "baseline",
            factor=1000.0,
        )
        optimized, optimized_errors = distribution_vectors(
            aggregates,
            configuration,
            cases,
            "optimized",
            factor=1000.0,
        )
        axis.bar(
            [position - bar_width / 2.0 for position in x_positions],
            baseline,
            width=bar_width,
            yerr=nonzero_errors(baseline_errors),
            capsize=3,
            color="#9D9D9D",
            label="Baseline",
        )
        axis.bar(
            [position + bar_width / 2.0 for position in x_positions],
            optimized,
            width=bar_width,
            yerr=nonzero_errors(optimized_errors),
            capsize=3,
            color=COLORS[0],
            label="Optimized",
        )
        axis.set_xticks(x_positions, [case_label(case) for case in cases])
        axis.set_yscale("log")
        axis.set_title(labels[configuration])
        axis.tick_params(axis="x", labelsize=9)
        style_axes(axis)

    for axis in flattened_axes[len(configurations) :]:
        axis.set_visible(False)
    for axis in axes[:, 0]:
        axis.set_ylabel("Median time (ms, log scale)")
    handles, legend_labels = flattened_axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=2,
        frameon=False,
    )
    figure.suptitle(f"{title}: execution time", y=0.995)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    return figure


def save_figure(figure, prefix, name, formats, dpi):
    prefix = prefix.expanduser()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for output_format in formats:
        output = prefix.parent / f"{prefix.name}-{name}.{output_format}"
        figure.savefig(output, dpi=dpi, bbox_inches="tight")
        print(f"WROTE,{output.resolve()}")
    plt.close(figure)


def print_summary(aggregates, configurations, cases, labels):
    for configuration in configurations:
        for case in cases:
            aggregate = aggregates.get((configuration, case))
            if not aggregate:
                continue
            print(
                "SUMMARY,"
                f"{labels[configuration]},{case},{aggregate.speedup.count},"
                f"{aggregate.baseline.center * 1000.0:.9g},"
                f"{aggregate.optimized.center * 1000.0:.9g},"
                f"{aggregate.speedup.center:.9g}"
            )


def main():
    args = parse_args()
    try:
        results = load_paired_results(args.csv)
        cases = select_cases(results, args.case)
        configurations = select_configurations(results, cases)
        labels = configuration_labels(configurations)
        aggregates = aggregate_results(results)
        print_summary(aggregates, configurations, cases, labels)

        if args.metric in ("both", "speedup"):
            figure = plot_speedup(
                aggregates,
                configurations,
                cases,
                labels,
                args.title,
                args.speedup_scale,
            )
            save_figure(
                figure,
                args.output_prefix,
                "speedup",
                args.formats,
                args.dpi,
            )
        if args.metric in ("both", "time"):
            figure = plot_time(
                aggregates,
                configurations,
                cases,
                labels,
                args.title,
            )
            save_figure(
                figure,
                args.output_prefix,
                "time",
                args.formats,
                args.dpi,
            )
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

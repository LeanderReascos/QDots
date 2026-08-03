"""Analyze timing and memory data stored by ``Profiler``.

Example:
    python profiler_analysis.py data/single_dot_error.pickle
    python profiler_analysis.py data/single_dot_error.pickle --out Figures/single_dot_error_profiler --format png

The input pickle is expected to be one of the result dictionaries produced by
the calculation scripts in this directory, or a single point dictionary with a
``profiler`` entry.
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


METRIC_KEYS = (
    "time_s",
    "memory_before_MB",
    "memory_after_MB",
    "memory_change_MB",
)

POINT_EXCLUDE_KEYS = {"profiler", "story", "potential"}
RUN_EXCLUDE_KEYS = {"points"}
STAGE_ROW_KEYS = {
    "stage",
    "stage_order",
    "category",
    "iteration",
    *METRIC_KEYS,
}

CATEGORY_ORDER = [
    "MAD init",
    "MRA potential",
    "Single-particle orbitals",
    "Integral setup",
    "Integrals",
    "Two-body setup",
    "Two-body transform",
    "FCI",
    "RDM2 transform",
    "Orbital refinement",
    "Other",
]

ITERATION_PATTERNS = (
    re.compile(r"(?:for iteration|iteration)\s+(\d+)$", re.IGNORECASE),
    re.compile(r"\s+(\d+)$"),
)


def load_results(path: str | Path) -> Any:
    """Load a trusted result pickle.

    Pickle can execute embedded code while loading. Use this only for result
    files you produced yourself or otherwise trust.
    """

    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def flatten_profiler(data: Any) -> list[dict[str, Any]]:
    """Return one row per profiler stage."""

    rows: list[dict[str, Any]] = []

    for base, point in iter_points(data):
        profiler = point.get("profiler")
        if not isinstance(profiler, dict):
            continue

        for stage_order, (stage_name, metrics) in enumerate(profiler.items()):
            if not isinstance(metrics, dict):
                continue

            category, iteration = normalize_stage(stage_name)
            row = dict(base)
            row.update(
                {
                    "stage": stage_name,
                    "stage_order": stage_order,
                    "category": category,
                    "iteration": iteration,
                }
            )
            for key in METRIC_KEYS:
                row[key] = to_float(metrics.get(key))
            rows.append(row)

    return rows


def summarize_points(stage_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one row per calculation point."""

    by_point: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in stage_rows:
        by_point[str(row["point_uid"])].append(row)

    summaries: list[dict[str, Any]] = []
    for point_uid, rows in by_point.items():
        rows = sorted(rows, key=lambda row: row["stage_order"])
        base = {
            key: value
            for key, value in rows[0].items()
            if key not in STAGE_ROW_KEYS
        }

        times = [value for value in (to_float(row.get("time_s")) for row in rows) if value is not None]
        before = [to_float(row.get("memory_before_MB")) for row in rows]
        after = [to_float(row.get("memory_after_MB")) for row in rows]
        changes = [to_float(row.get("memory_change_MB")) for row in rows]

        start_memory = next((value for value in before if value is not None), None)
        final_memory = next((value for value in reversed(after) if value is not None), None)
        peak_memory = max((value for value in after if value is not None), default=None)
        positive_changes = [value for value in changes if value is not None and value > 0.0]

        summary = dict(base)
        summary.update(
            {
                "profiled_time_s": sum(times),
                "setup_time_s": sum(
                    row["time_s"]
                    for row in rows
                    if row.get("iteration") is None and row.get("time_s") is not None
                ),
                "iteration_time_s": sum(
                    row["time_s"]
                    for row in rows
                    if row.get("iteration") is not None and row.get("time_s") is not None
                ),
                "start_memory_MB": start_memory,
                "final_memory_MB": final_memory,
                "peak_memory_MB": peak_memory,
                "peak_memory_growth_MB": (
                    peak_memory - start_memory
                    if peak_memory is not None and start_memory is not None
                    else None
                ),
                "net_memory_change_MB": (
                    final_memory - start_memory
                    if final_memory is not None and start_memory is not None
                    else None
                ),
                "positive_stage_memory_change_MB": sum(positive_changes),
                "n_stages_profiled": len(rows),
            }
        )

        log_time = to_float(summary.pop("log_time", None))
        if log_time is not None:
            summary["scf_log_time_s"] = log_time

        if "n_iterations" not in summary and "iterations" in summary:
            summary["n_iterations"] = summary["iterations"]

        summaries.append(summary)

    return sorted(
        summaries,
        key=lambda row: (
            number_or_inf(row.get("run_index")),
            number_or_inf(row.get("point_index")),
        ),
    )


def iter_points(data: Any):
    """Yield ``(base_fields, point_dict)`` from supported result shapes."""

    if isinstance(data, dict) and "runs" in data:
        for run_index, run in enumerate(data.get("runs", [])):
            if not isinstance(run, dict):
                continue
            run_fields = scalar_fields(run, RUN_EXCLUDE_KEYS)
            for point_index, point in enumerate(run.get("points", [])):
                if not isinstance(point, dict):
                    continue
                base = dict(run_fields)
                base.update(scalar_fields(point, POINT_EXCLUDE_KEYS))
                base["run_index"] = run_index
                base["point_index"] = point_index
                base["point_uid"] = f"run{run_index:03d}_point{point_index:05d}"
                base["series"] = series_label(base)
                yield base, point
        return

    if isinstance(data, list):
        for point_index, point in enumerate(data):
            if not isinstance(point, dict):
                continue
            base = scalar_fields(point, POINT_EXCLUDE_KEYS)
            base["run_index"] = 0
            base["point_index"] = point_index
            base["point_uid"] = f"run000_point{point_index:05d}"
            base["series"] = series_label(base)
            yield base, point
        return

    if isinstance(data, dict) and "profiler" in data:
        base = scalar_fields(data, POINT_EXCLUDE_KEYS)
        base["run_index"] = 0
        base["point_index"] = 0
        base["point_uid"] = "run000_point00000"
        base["series"] = series_label(base)
        yield base, data
        return

    raise ValueError("No profiler points found. Expected a result dict, a point list, or one point dict.")


def normalize_stage(stage_name: str) -> tuple[str, int | None]:
    lower = stage_name.lower()
    iteration = None
    for pattern in ITERATION_PATTERNS:
        match = pattern.search(stage_name)
        if match:
            iteration = int(match.group(1))
            break

    if "mad world" in lower:
        category = "MAD init"
    elif "mra" in lower and ("potential" in lower or "function factory" in lower):
        category = "MRA potential"
    elif "single-particle" in lower:
        category = "Single-particle orbitals"
    elif "two-body integrals transformation" in lower:
        category = "Two-body transform"
    elif "two-body integrals computation" in lower:
        category = "Two-body setup"
    elif "rdm2" in lower:
        category = "RDM2 transform"
    elif "fci" in lower:
        category = "FCI"
    elif "orbital refinement" in lower:
        category = "Orbital refinement"
    elif lower == "integral computation" or lower.startswith("integrals") or "integrals computation" in lower:
        category = "Integrals"
    else:
        category = "Other"

    return category, iteration


def analyze(
    data: Any,
    out_dir: str | Path,
    *,
    formats: tuple[str, ...] = ("pdf",),
    aggregate: str = "median",
    yscale: str = "linear",
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stage_rows = flatten_profiler(data)
    if not stage_rows:
        raise ValueError("No profiler data found in the input.")

    point_rows = summarize_points(stage_rows)
    outputs = [
        write_csv(stage_rows, out_dir / "profiler_stages.csv"),
        write_csv(point_rows, out_dir / "profiler_points.csv"),
    ]

    outputs.extend(plot_time_scaling(point_rows, out_dir, formats=formats, yscale=yscale))
    outputs.extend(plot_time_breakdown(stage_rows, out_dir, formats=formats, aggregate=aggregate, yscale=yscale))
    outputs.extend(plot_memory_scaling(point_rows, out_dir, formats=formats))
    outputs.extend(plot_iteration_heatmap(stage_rows, out_dir, formats=formats, aggregate=aggregate))
    outputs.extend(plot_largest_orbital_iteration_breakdown(stage_rows, out_dir, formats=formats, yscale=yscale))

    return outputs


def plot_time_scaling(
    point_rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    formats: tuple[str, ...],
    yscale: str,
) -> list[Path]:
    plt = require_matplotlib()
    rows = [row for row in point_rows if to_float(row.get("n_orbitals")) is not None]
    if not rows:
        return []

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.0), sharex=True, constrained_layout=True)
    for series, series_rows in grouped(rows, "series").items():
        series_rows = sorted(series_rows, key=lambda row: (number_or_inf(row.get("n_orbitals")), row["point_index"]))
        x = [row["n_orbitals"] for row in series_rows]
        axes[0].plot(x, [row["profiled_time_s"] for row in series_rows], marker="o", label=series)
        axes[1].plot(x, [to_float(row.get("n_iterations")) for row in series_rows], marker="o", label=series)

    axes[0].set_ylabel("profiled time (s)")
    axes[0].set_yscale(yscale)
    axes[0].set_title("Total profiled time")
    axes[1].set_xlabel("number of orbitals")
    axes[1].set_ylabel("SCF iterations")
    axes[1].set_title("Convergence work")
    finish_axes(axes)
    add_legend(axes[0], rows)
    return save_figure(fig, out_dir, "time_scaling_by_orbitals", formats)


def plot_time_breakdown(
    stage_rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    formats: tuple[str, ...],
    aggregate: str,
    yscale: str,
) -> list[Path]:
    plt = require_matplotlib()

    point_category = defaultdict(list)
    for row in stage_rows:
        n_orbitals = to_float(row.get("n_orbitals"))
        time_s = to_float(row.get("time_s"))
        if n_orbitals is None or time_s is None:
            continue
        key = (row["point_uid"], int(n_orbitals), row["category"])
        point_category[key].append(time_s)

    grouped_values: dict[tuple[int, str], list[float]] = defaultdict(list)
    for (_point_uid, n_orbitals, category), values in point_category.items():
        grouped_values[(n_orbitals, category)].append(sum(values))

    if not grouped_values:
        return []

    n_orbitals_values = sorted({key[0] for key in grouped_values})
    categories = ordered_categories({key[1] for key in grouped_values})
    x_positions = list(range(len(n_orbitals_values)))
    bottoms = [0.0] * len(n_orbitals_values)

    fig, ax = plt.subplots(figsize=(max(8.0, 0.42 * len(n_orbitals_values)), 5.2), constrained_layout=True)
    colors = category_colors(plt, categories)

    for category in categories:
        heights = [
            aggregate_numbers(grouped_values.get((n_orbitals, category), []), aggregate)
            for n_orbitals in n_orbitals_values
        ]
        ax.bar(x_positions, heights, bottom=bottoms, label=category, color=colors[category])
        bottoms = [bottom + (height or 0.0) for bottom, height in zip(bottoms, heights)]

    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(value) for value in n_orbitals_values], rotation=45 if len(n_orbitals_values) > 12 else 0)
    ax.set_xlabel("number of orbitals")
    ax.set_ylabel(f"{aggregate} time per point (s)")
    ax.set_title("Profiler stage time breakdown")
    ax.set_yscale(yscale)
    finish_axes([ax])
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    return save_figure(fig, out_dir, "stage_time_breakdown_by_orbitals", formats)


def plot_memory_scaling(
    point_rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    formats: tuple[str, ...],
) -> list[Path]:
    plt = require_matplotlib()
    rows = [
        row
        for row in point_rows
        if to_float(row.get("n_orbitals")) is not None and to_float(row.get("peak_memory_MB")) is not None
    ]
    if not rows:
        return []

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.0), sharex=True, constrained_layout=True)
    for series, series_rows in grouped(rows, "series").items():
        series_rows = sorted(series_rows, key=lambda row: (number_or_inf(row.get("n_orbitals")), row["point_index"]))
        x = [row["n_orbitals"] for row in series_rows]
        axes[0].plot(x, [row["peak_memory_MB"] for row in series_rows], marker="o", label=series)
        axes[1].plot(x, [row["peak_memory_growth_MB"] for row in series_rows], marker="o", label=series)

    axes[0].set_ylabel("peak RSS (MB)")
    axes[0].set_title("Peak memory")
    axes[1].set_xlabel("number of orbitals")
    axes[1].set_ylabel("peak growth (MB)")
    axes[1].set_title("Memory growth during point")
    finish_axes(axes)
    add_legend(axes[0], rows)
    return save_figure(fig, out_dir, "memory_scaling_by_orbitals", formats)


def plot_iteration_heatmap(
    stage_rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    formats: tuple[str, ...],
    aggregate: str,
) -> list[Path]:
    plt = require_matplotlib()

    point_iteration = defaultdict(list)
    for row in stage_rows:
        iteration = row.get("iteration")
        n_orbitals = to_float(row.get("n_orbitals"))
        time_s = to_float(row.get("time_s"))
        if iteration is None or n_orbitals is None or time_s is None:
            continue
        point_iteration[(row["point_uid"], int(n_orbitals), int(iteration))].append(time_s)

    grouped_values: dict[tuple[int, int], list[float]] = defaultdict(list)
    for (_point_uid, n_orbitals, iteration), values in point_iteration.items():
        grouped_values[(n_orbitals, iteration)].append(sum(values))

    if not grouped_values:
        return []

    n_orbitals_values = sorted({key[0] for key in grouped_values})
    iterations = list(range(max(key[1] for key in grouped_values) + 1))
    matrix = []
    for n_orbitals in n_orbitals_values:
        matrix.append(
            [
                aggregate_numbers(grouped_values.get((n_orbitals, iteration), []), aggregate, default=math.nan)
                for iteration in iterations
            ]
        )

    fig, ax = plt.subplots(
        figsize=(max(7.0, 0.35 * len(iterations)), max(4.0, 0.32 * len(n_orbitals_values))),
        constrained_layout=True,
    )
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_xlabel("SCF iteration")
    ax.set_ylabel("number of orbitals")
    ax.set_yticks(range(len(n_orbitals_values)))
    ax.set_yticklabels([str(value) for value in n_orbitals_values])
    ax.set_xticks(iteration_ticks(iterations))
    ax.set_title(f"{aggregate.title()} time per SCF iteration")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("time (s)")
    return save_figure(fig, out_dir, "iteration_time_heatmap", formats)


def plot_largest_orbital_iteration_breakdown(
    stage_rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    formats: tuple[str, ...],
    yscale: str,
) -> list[Path]:
    plt = require_matplotlib()

    loop_rows = [
        row
        for row in stage_rows
        if row.get("iteration") is not None
        and to_float(row.get("n_orbitals")) is not None
        and to_float(row.get("time_s")) is not None
    ]
    if not loop_rows:
        return []

    candidate_uid = max(
        {row["point_uid"] for row in loop_rows},
        key=lambda uid: (
            max(to_float(row.get("n_orbitals")) or -math.inf for row in loop_rows if row["point_uid"] == uid),
            sum(to_float(row.get("time_s")) or 0.0 for row in loop_rows if row["point_uid"] == uid),
        ),
    )
    rows = [row for row in loop_rows if row["point_uid"] == candidate_uid]
    n_orbitals = int(to_float(rows[0].get("n_orbitals")) or 0)
    iterations = sorted({int(row["iteration"]) for row in rows})
    categories = ordered_categories({row["category"] for row in rows})
    x_positions = list(range(len(iterations)))
    bottoms = [0.0] * len(iterations)

    fig, ax = plt.subplots(figsize=(max(8.0, 0.45 * len(iterations)), 5.2), constrained_layout=True)
    colors = category_colors(plt, categories)

    for category in categories:
        heights = []
        for iteration in iterations:
            heights.append(
                sum(
                    row["time_s"]
                    for row in rows
                    if row["category"] == category and int(row["iteration"]) == iteration
                )
            )
        ax.bar(x_positions, heights, bottom=bottoms, label=category, color=colors[category])
        bottoms = [bottom + height for bottom, height in zip(bottoms, heights)]

    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(value) for value in iterations])
    ax.set_xlabel("SCF iteration")
    ax.set_ylabel("time (s)")
    ax.set_title(f"Iteration stage breakdown, n_orbitals={n_orbitals}")
    ax.set_yscale(yscale)
    finish_axes([ax])
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    return save_figure(fig, out_dir, "iteration_stage_breakdown_largest_orbitals", formats)



def plot_task_time_violins_by_electron(
    stage_rows: list[dict[str, Any]],
    *,
    electron_numbers: tuple[int, ...] = (2, 3),
    categories: list[str] | None = None,
    yscale: str = "log",
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
) -> tuple[Any, Any]:
    """Plot per-iteration task-time violins for each electron number.

    Each violin contains the profiler ``time_s`` values for one task category,
    collected over all SCF iterations at a fixed ``n_orbitals``.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rows = [
        row
        for row in stage_rows
        if row.get("iteration") is not None
        and to_float(row.get("time_s")) is not None
        and (to_float(row.get("time_s")) or 0.0) > 0.0
        and to_float(row.get("n_electrons")) in electron_numbers
        and to_float(row.get("n_orbitals")) is not None
    ]
    if not rows:
        raise ValueError("No per-iteration profiler times found for the requested electron numbers.")

    if categories is None:
        categories = ordered_categories({row["category"] for row in rows})
    else:
        categories = [category for category in categories if any(row["category"] == category for row in rows)]
    if not categories:
        raise ValueError("No task categories available for violin plotting.")

    if figsize is None:
        figsize = (7.2 * len(electron_numbers), 5.6)
    fig, axes = plt.subplots(1, len(electron_numbers), figsize=figsize, sharey=True, squeeze=False)
    axes = axes[0]
    colors = category_colors(plt, categories)
    group_width = 0.78
    offsets = np.linspace(-group_width / 2, group_width / 2, len(categories)) if len(categories) > 1 else np.array([0.0])
    violin_width = min(0.16, group_width / max(1, len(categories)) * 0.8)

    for ax, n_electrons in zip(axes, electron_numbers):
        electron_rows = [row for row in rows if int(to_float(row.get("n_electrons")) or -1) == n_electrons]
        n_orbitals_values = sorted({int(to_float(row["n_orbitals"]) or 0) for row in electron_rows})
        x_centers = np.arange(len(n_orbitals_values), dtype=float)

        for category_index, category in enumerate(categories):
            datasets = []
            positions = []
            for orbital_index, n_orbitals in enumerate(n_orbitals_values):
                values = [
                    to_float(row.get("time_s"))
                    for row in electron_rows
                    if int(to_float(row.get("n_orbitals")) or -1) == n_orbitals
                    and row["category"] == category
                ]
                values = [value for value in values if value is not None and value > 0.0]
                if not values:
                    continue
                datasets.append(_violin_values(values))
                positions.append(x_centers[orbital_index] + offsets[category_index])

            if datasets:
                violin = ax.violinplot(
                    datasets,
                    positions=positions,
                    widths=violin_width,
                    showmeans=False,
                    showmedians=True,
                    showextrema=False,
                )
                for body in violin["bodies"]:
                    body.set_facecolor(colors[category])
                    body.set_edgecolor("black")
                    body.set_alpha(0.72)
                    body.set_linewidth(0.6)
                if "cmedians" in violin:
                    violin["cmedians"].set_color("black")
                    violin["cmedians"].set_linewidth(0.8)

        ax.set_title(rf"$N={n_electrons}$")
        ax.set_xlabel("Number of orbitals")
        ax.set_xticks(x_centers)
        ax.set_xticklabels([str(value) for value in n_orbitals_values])
        ax.set_yscale(yscale)
        ax.grid(True, which="major", alpha=0.28)
        ax.grid(True, which="minor", axis="y", alpha=0.16)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Task time per iteration (s)")
    handles = [Patch(facecolor=colors[category], edgecolor="black", alpha=0.72, label=category) for category in categories]
    fig.legend(handles=handles, loc="upper center", ncol=min(len(categories), 4), frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    if save_path is not None:
        fig.savefig(save_path)
    return fig, axes


def plot_iteration_heatmaps_by_electron(
    stage_rows: list[dict[str, Any]],
    *,
    electron_numbers: tuple[int, ...] = (2, 3),
    aggregate: str = "median",
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
) -> tuple[Any, Any]:
    """Plot heatmaps of total profiled time per SCF iteration."""

    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    if figsize is None:
        figsize = (7.0 * len(electron_numbers), 5.0)
    fig, axes = plt.subplots(1, len(electron_numbers), figsize=figsize, squeeze=False, constrained_layout=True)
    axes = axes[0]

    matrices = []
    metadata = []
    positive_values = []
    for n_electrons in electron_numbers:
        matrix, n_orbitals_values, iterations = iteration_time_matrix(stage_rows, n_electrons, aggregate=aggregate)
        matrices.append(matrix)
        metadata.append((n_orbitals_values, iterations))
        positive_values.extend(matrix[np.isfinite(matrix) & (matrix > 0.0)].ravel().tolist())

    if not positive_values:
        raise ValueError("No positive iteration times found for heatmap plotting.")

    norm = mpl.colors.LogNorm(vmin=min(positive_values), vmax=max(positive_values))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")

    for ax, n_electrons, matrix, (n_orbitals_values, iterations) in zip(axes, electron_numbers, matrices, metadata):
        masked = np.ma.masked_invalid(matrix)
        image = ax.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
        ax.set_title(rf"$N={n_electrons}$")
        ax.set_xlabel("SCF iteration")
        ax.set_ylabel("Number of orbitals")
        ax.set_yticks(np.arange(len(n_orbitals_values)))
        ax.set_yticklabels([str(value) for value in n_orbitals_values])
        ax.set_xticks(iteration_ticks(list(iterations)))
        ax.set_xticklabels([str(value) for value in iteration_ticks(list(iterations))])

    colorbar = fig.colorbar(image, ax=axes, shrink=0.92, pad=0.02)
    colorbar.set_label("Iteration time (s)")
    if save_path is not None:
        fig.savefig(save_path)
    return fig, axes


def plot_stage_time_breakdowns_by_electron(
    stage_rows: list[dict[str, Any]],
    *,
    electron_numbers: tuple[int, ...] = (2, 3),
    aggregate: str = "median",
    yscale: str = "log",
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
) -> tuple[Any, Any]:
    """Plot stacked stage-time bars separately for each electron number."""

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rows = [
        row
        for row in stage_rows
        if to_float(row.get("n_electrons")) in electron_numbers
        and to_float(row.get("n_orbitals")) is not None
        and to_float(row.get("time_s")) is not None
    ]
    if not rows:
        raise ValueError("No profiler stage times found for the requested electron numbers.")

    categories = ordered_categories({row["category"] for row in rows})
    colors = category_colors(plt, categories)
    if figsize is None:
        figsize = (7.0 * len(electron_numbers), 5.4)
    fig, axes = plt.subplots(1, len(electron_numbers), figsize=figsize, sharey=True, squeeze=False)
    axes = axes[0]

    positive_heights = []
    breakdowns = []
    for n_electrons in electron_numbers:
        n_orbitals_values, category_values = stage_breakdown_by_orbitals(
            stage_rows, n_electrons, categories=categories, aggregate=aggregate
        )
        breakdowns.append((n_orbitals_values, category_values))
        for values in category_values.values():
            positive_heights.extend([value for value in values if value > 0.0])

    y_floor = min(positive_heights) * 0.5 if positive_heights else 1e-12

    for ax, n_electrons, (n_orbitals_values, category_values) in zip(axes, electron_numbers, breakdowns):
        x = np.arange(len(n_orbitals_values))
        bottoms = np.zeros(len(n_orbitals_values), dtype=float)
        for category in categories:
            heights = np.array(category_values.get(category, [0.0] * len(n_orbitals_values)), dtype=float)
            plot_bottoms = np.where(bottoms > 0.0, bottoms, y_floor)
            ax.bar(x, heights, bottom=plot_bottoms, color=colors[category], edgecolor="black", linewidth=0.25)
            bottoms += heights

        ax.set_title(rf"$N={n_electrons}$")
        ax.set_xlabel("Number of orbitals")
        ax.set_xticks(x)
        ax.set_xticklabels([str(value) for value in n_orbitals_values])
        ax.set_yscale(yscale)
        ax.set_ylim(bottom=y_floor)
        ax.grid(True, which="major", axis="y", alpha=0.28)
        ax.grid(True, which="minor", axis="y", alpha=0.16)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(f"{aggregate.title()} stage time per point (s)")
    handles = [Patch(facecolor=colors[category], edgecolor="black", label=category) for category in categories]
    fig.legend(handles=handles, loc="upper center", ncol=min(len(categories), 4), frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    if save_path is not None:
        fig.savefig(save_path)
    return fig, axes


def plot_time_scaling_summary(
    point_rows: list[dict[str, Any]],
    *,
    electron_numbers: tuple[int, ...] = (2, 3),
    yscale: str = "log",
    figsize: tuple[float, float] = (8.0, 5.6),
    save_path: str | Path | None = None,
) -> tuple[Any, Any]:
    """Plot total, setup, and iterative profiled time versus ``n_orbitals``."""

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    line_styles = {
        "profiled_time_s": "-",
        "setup_time_s": "--",
        "iteration_time_s": ":",
    }
    labels = {
        "profiled_time_s": "total",
        "setup_time_s": "setup",
        "iteration_time_s": "SCF loop",
    }
    colors = plt.get_cmap("tab10")

    for index, n_electrons in enumerate(electron_numbers):
        rows = sorted(
            [row for row in point_rows if int(to_float(row.get("n_electrons")) or -1) == n_electrons],
            key=lambda row: number_or_inf(row.get("n_orbitals")),
        )
        if not rows:
            continue
        x = [row["n_orbitals"] for row in rows]
        for key, linestyle in line_styles.items():
            ax.plot(
                x,
                [to_float(row.get(key)) for row in rows],
                marker="o",
                linestyle=linestyle,
                color=colors(index),
                label=rf"$N={n_electrons}$ {labels[key]}",
            )

    ax.set_xlabel("Number of orbitals")
    ax.set_ylabel("Time (s)")
    ax.set_yscale(yscale)
    ax.grid(True, which="major", alpha=0.28)
    ax.grid(True, which="minor", axis="y", alpha=0.16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
    return fig, ax


def plot_memory_scaling_summary(
    point_rows: list[dict[str, Any]],
    *,
    electron_numbers: tuple[int, ...] = (2, 3),
    figsize: tuple[float, float] = (8.0, 6.0),
    save_path: str | Path | None = None,
) -> tuple[Any, Any]:
    """Plot peak RSS and peak RSS growth versus ``n_orbitals``."""

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    colors = plt.get_cmap("tab10")

    for index, n_electrons in enumerate(electron_numbers):
        rows = sorted(
            [row for row in point_rows if int(to_float(row.get("n_electrons")) or -1) == n_electrons],
            key=lambda row: number_or_inf(row.get("n_orbitals")),
        )
        if not rows:
            continue
        x = [row["n_orbitals"] for row in rows]
        axes[0].plot(x, [to_float(row.get("peak_memory_MB")) for row in rows], marker="o", color=colors(index), label=rf"$N={n_electrons}$")
        axes[1].plot(x, [to_float(row.get("peak_memory_growth_MB")) for row in rows], marker="o", color=colors(index), label=rf"$N={n_electrons}$")

    axes[0].set_ylabel("Peak RSS (MB)")
    axes[1].set_ylabel("Peak growth (MB)")
    axes[1].set_xlabel("Number of orbitals")
    for ax in axes:
        ax.grid(True, which="major", alpha=0.28)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
    return fig, axes


def iteration_time_matrix(
    stage_rows: list[dict[str, Any]],
    n_electrons: int,
    *,
    aggregate: str = "median",
) -> tuple[Any, list[int], Any]:
    """Return matrix[n_orbitals, iteration] with total time per SCF iteration."""

    import numpy as np

    point_iteration: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for row in stage_rows:
        if int(to_float(row.get("n_electrons")) or -1) != n_electrons:
            continue
        iteration = row.get("iteration")
        n_orbitals = to_float(row.get("n_orbitals"))
        time_s = to_float(row.get("time_s"))
        if iteration is None or n_orbitals is None or time_s is None:
            continue
        point_iteration[(row["point_uid"], int(n_orbitals), int(iteration))].append(time_s)

    grouped_values: dict[tuple[int, int], list[float]] = defaultdict(list)
    for (_point_uid, n_orbitals, iteration), values in point_iteration.items():
        grouped_values[(n_orbitals, iteration)].append(sum(values))

    if not grouped_values:
        raise ValueError(f"No iteration timings found for N={n_electrons}.")

    n_orbitals_values = sorted({key[0] for key in grouped_values})
    iterations = np.arange(max(key[1] for key in grouped_values) + 1)
    matrix = np.full((len(n_orbitals_values), len(iterations)), np.nan, dtype=float)
    for row_index, n_orbitals in enumerate(n_orbitals_values):
        for iteration in iterations:
            value = aggregate_numbers(grouped_values.get((n_orbitals, int(iteration)), []), aggregate, default=math.nan)
            matrix[row_index, int(iteration)] = value if value > 0.0 else math.nan
    return matrix, n_orbitals_values, iterations


def stage_breakdown_by_orbitals(
    stage_rows: list[dict[str, Any]],
    n_electrons: int,
    *,
    categories: list[str],
    aggregate: str = "median",
) -> tuple[list[int], dict[str, list[float]]]:
    """Aggregate profiler stage times by category and ``n_orbitals``."""

    point_category: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for row in stage_rows:
        if int(to_float(row.get("n_electrons")) or -1) != n_electrons:
            continue
        n_orbitals = to_float(row.get("n_orbitals"))
        time_s = to_float(row.get("time_s"))
        if n_orbitals is None or time_s is None:
            continue
        point_category[(row["point_uid"], int(n_orbitals), row["category"])].append(time_s)

    grouped_values: dict[tuple[int, str], list[float]] = defaultdict(list)
    for (_point_uid, n_orbitals, category), values in point_category.items():
        grouped_values[(n_orbitals, category)].append(sum(values))

    n_orbitals_values = sorted({key[0] for key in grouped_values})
    category_values = {
        category: [aggregate_numbers(grouped_values.get((n_orbitals, category), []), aggregate) for n_orbitals in n_orbitals_values]
        for category in categories
    }
    return n_orbitals_values, category_values


def _violin_values(values: list[float]) -> list[float]:
    """Make one-sample violins drawable while keeping the displayed scale unchanged."""

    if len(values) != 1:
        return values
    value = values[0]
    epsilon = max(abs(value) * 1e-3, 1e-12)
    return [max(value - epsilon, value * 0.999), value + epsilon]

def write_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ordered_columns(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in columns})
    return path


def ordered_columns(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "run_index",
        "point_index",
        "point_uid",
        "series",
        "n_electrons",
        "V",
        "v",
        "vx",
        "vy",
        "n_orbitals",
        "energy",
        "converged",
        "n_iterations",
        "profiled_time_s",
        "setup_time_s",
        "iteration_time_s",
        "scf_log_time_s",
        "start_memory_MB",
        "final_memory_MB",
        "peak_memory_MB",
        "peak_memory_growth_MB",
        "net_memory_change_MB",
        "positive_stage_memory_change_MB",
        "n_stages_profiled",
        "stage_order",
        "stage",
        "category",
        "iteration",
        *METRIC_KEYS,
    ]
    present = set().union(*(row.keys() for row in rows)) if rows else set()
    columns = [key for key in preferred if key in present]
    columns.extend(sorted(present.difference(columns)))
    return columns


def scalar_fields(mapping: dict[str, Any], exclude: set[str]) -> dict[str, Any]:
    fields = {}
    for key, value in mapping.items():
        if key in exclude:
            continue
        value = to_builtin_scalar(value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            fields[key] = value
    return fields


def to_builtin_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return value
    return value


def to_float(value: Any) -> float | None:
    value = to_builtin_scalar(value)
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def number_or_inf(value: Any) -> float:
    number = to_float(value)
    return number if number is not None else math.inf


def aggregate_numbers(values: list[float], aggregate: str, default: float = 0.0) -> float:
    if not values:
        return default
    if aggregate == "mean":
        return mean(values)
    if aggregate == "median":
        return median(values)
    raise ValueError(f"Unsupported aggregate: {aggregate}")


def grouped(rows: list[dict[str, Any]], key: str) -> dict[Any, list[dict[str, Any]]]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row.get(key)].append(row)
    return dict(groups)


def series_label(fields: dict[str, Any]) -> str:
    parts = [f"run {fields.get('run_index', 0)}"]
    if "n_electrons" in fields:
        parts.append(f"N={format_value(fields['n_electrons'])}")
    if "V" in fields:
        parts.append(f"V={format_value(fields['V'])}")
    return ", ".join(parts)


def format_value(value: Any) -> str:
    number = to_float(value)
    if number is None:
        return str(value)
    return f"{number:g}"


def ordered_categories(categories: set[str]) -> list[str]:
    known = [category for category in CATEGORY_ORDER if category in categories]
    unknown = sorted(categories.difference(known))
    return known + unknown


def category_colors(plt: Any, categories: list[str]) -> dict[str, Any]:
    cmap = plt.get_cmap("tab20")
    return {category: cmap(index % cmap.N) for index, category in enumerate(categories)}


def finish_axes(axes: Any) -> None:
    for ax in axes:
        ax.grid(True, which="major", color="#dddddd", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def add_legend(ax: Any, rows: list[dict[str, Any]]) -> None:
    if len({row.get("series") for row in rows}) <= 8:
        ax.legend(loc="best", frameon=False)


def iteration_ticks(iterations: list[int]) -> list[int]:
    if len(iterations) <= 12:
        return iterations
    step = max(1, math.ceil(len(iterations) / 12))
    return iterations[::step]


def save_figure(fig: Any, out_dir: Path, stem: str, formats: tuple[str, ...]) -> list[Path]:
    outputs = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, bbox_inches="tight")
        outputs.append(path)
    fig.clear()
    return outputs


def require_matplotlib() -> Any:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.grid": True,
            "axes.axisbelow": True,
            "font.size": 10,
        }
    )
    return plt


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    return value


def default_out_dir(input_path: Path) -> Path:
    if input_path.parent.name.startswith("data"):
        return input_path.parent.parent / "Figures" / f"{input_path.stem}_profiler"
    return input_path.with_suffix("").parent / f"{input_path.stem}_profiler"


def demo_data() -> dict[str, Any]:
    def point(n_orbitals: int, scale: float) -> dict[str, Any]:
        return {
            "v": 0.4,
            "n_orbitals": n_orbitals,
            "energy": -44.9 - scale,
            "n_iterations": 4,
            "converged": True,
            "log_time": 1.4 * scale,
            "profiler": {
                "MAD world initialization": stage(0.3, 255.0, 280.0),
                "MRA potential creation": stage(0.8 * scale, 280.0, 285.0),
                "Single-particle orbitals calculation": stage(1.2 * scale, 285.0, 330.0 + 12 * scale),
                "Integrals computation for iteration 0": stage(0.08 * scale, 330.0, 331.0),
                "FCI calculation for iteration 0": stage(0.06 * scale, 331.0, 332.0),
                "Orbital Refinement for iteration 0": stage(0.3 * scale, 332.0, 333.0),
                "Integrals computation for iteration 1": stage(0.08 * scale, 333.0, 333.0),
                "FCI calculation for iteration 1": stage(0.06 * scale, 333.0, 334.0),
                "Orbital Refinement for iteration 1": stage(0.2 * scale, 334.0, 334.0),
            },
        }

    return {
        "metadata": {"description": "demo profiler data"},
        "runs": [
            {"n_electrons": 2, "V": 0.4, "points": [point(2, 1.0), point(4, 2.0), point(6, 3.5)]},
        ],
    }


def stage(time_s: float, memory_before: float, memory_after: float) -> dict[str, float]:
    return {
        "time_s": time_s,
        "memory_before_MB": memory_before,
        "memory_after_MB": memory_after,
        "memory_change_MB": memory_after - memory_before,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", help="Result pickle to analyze.")
    parser.add_argument("--out", type=Path, help="Output directory for CSV files and figures.")
    parser.add_argument("--format", action="append", choices=("pdf", "png", "svg"), help="Figure format. May be repeated.")
    parser.add_argument("--aggregate", choices=("median", "mean"), default="median", help="Aggregation for repeated points.")
    parser.add_argument("--yscale", choices=("linear", "log"), default="linear", help="Y-axis scaling for timing plots.")
    parser.add_argument("--demo", action="store_true", help="Use built-in demo data instead of loading a pickle.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    formats = tuple(args.format or ("pdf",))

    if args.demo:
        data = demo_data()
        out_dir = args.out or Path("/tmp/qdots_profiler_demo")
    else:
        if not args.input:
            raise SystemExit("Provide an input pickle or use --demo.")
        input_path = Path(args.input)
        data = load_results(input_path)
        out_dir = args.out or default_out_dir(input_path)

    outputs = analyze(data, out_dir, formats=formats, aggregate=args.aggregate, yscale=args.yscale)
    print(f"Wrote {len(outputs)} files to {out_dir}:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()

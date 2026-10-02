"""Render the paper's fixed compute extrapolations as a Newsreader web figure.

Requires Matplotlib and NumPy. Supply the paper's scaling_results_v1 data
directory; source files are read only. The original fitted coefficients,
frontier assignments, and sampled curves are preserved without refitting.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist/static"
FONT = ROOT / "scripts/Newsreader-plot.ttf"
SEGMENTS = ("Middle third", "Last third")
COLORS = {"Middle third": "#198175", "Last third": "#c75b38"}
PANELS = (
    ("ar", "mean_frame", "AR · Average frame accuracy", (-3.5, 7.0)),
    ("ar", "full_trajectory", "AR · Full-trajectory accuracy", (-3.5, 16.0)),
    ("bidir", "mean_frame", "Bidir · Average frame accuracy", (-3.5, 16.0)),
)


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def scientific_label(log_value):
    value = 10 ** log_value
    return f"{value:.3g}" if value < 1000 else f"{value:.2e}".replace("e+0", "e").replace("e+", "e")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, required=True,
                        help="Directory containing the paper's state_compute_segmented data files")
    args = parser.parse_args()
    for folder in ("figures", "data"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    paths = {
        "curves": args.source_data / "state_compute_segmented_curves.csv",
        "frontier": args.source_data / "state_compute_segmented_frontier.csv",
        "checkpoints": args.source_data / "state_compute_segmented_checkpoints.csv",
        "analysis": args.source_data / "state_compute_segmented_analysis.json",
    }
    curves, frontier, checkpoints = (read_rows(paths[key]) for key in ("curves", "frontier", "checkpoints"))
    analysis = json.loads(paths["analysis"].read_text())
    fits = analysis["fits"]
    assert len(fits) == 6
    assert all(float(row["full_trajectory"]) == 0 for row in checkpoints if row["family"] == "bidir")

    # Public snapshots and metadata contain no local filesystem locations.
    for key in ("curves", "frontier", "checkpoints"):
        shutil.copyfile(paths[key], OUT / "data" / f"compute-extrapolation-{key}.csv")
    font_manager.fontManager.addfont(str(FONT))
    assert font_manager.FontProperties(fname=str(FONT)).get_name() == "Newsreader"
    plt.rcParams.update({
        "font.family": "Newsreader", "font.size": 14.5,
        "axes.titlesize": 17, "axes.labelsize": 14,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "text.color": "#172539", "axes.labelcolor": "#172539",
        "xtick.color": "#506075", "ytick.color": "#506075",
        "axes.edgecolor": "#c9d3df", "axes.spines.top": False,
        "axes.spines.right": False, "axes.unicode_minus": False,
        "svg.fonttype": "path", "svg.hashsalt": "compute-extrapolation",
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 10.5))
    fig.subplots_adjust(left=.13, right=.955, bottom=.06, top=.865, hspace=.61)
    handles = [
        Line2D([], [], color=COLORS[segment], marker="o", markersize=5,
               linewidth=1.7, label="Final third" if segment == "Last third" else segment)
        for segment in SEGMENTS
    ] + [
        Line2D([], [], color="#506075", linewidth=1.6, label="Fit within window"),
        Line2D([], [], color="#506075", linewidth=1.6, linestyle=(0, (4, 3)), label="Extrapolation"),
        Patch(facecolor="#edf1f4", edgecolor="#c9d3df", label="Observed compute range"),
        Line2D([], [], color="#506075", marker="D", markerfacecolor="white",
               markersize=5, linewidth=0, label="98% intersection"),
    ]
    legend = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, .996),
                        ncol=2, frameon=False, fontsize=12.2, columnspacing=2.0,
                        handlelength=2.0, handletextpad=.6, labelspacing=.7)
    annotations = []
    observed_ranges = []
    crossings = []
    for ax, (family, metric, title, limits) in zip(axes, PANELS):
        ax.set(xlim=limits, ylim=(-2, 115))
        ax.set_title(title, loc="left", pad=9)
        ax.set_xlabel("Training compute (PF-days)", labelpad=6)
        ax.set_ylabel("Accuracy (%)", labelpad=6)
        ax.set_yticks([0, 25, 50, 75, 100])
        ticks = [-3, -1, 1, 3, 5, 7] if limits[1] == 7 else [-3, 0, 3, 6, 9, 12, 15]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"1e{int(value)}"))
        ax.grid(axis="y", color="#e6ecf2", linewidth=.65)
        ax.set_axisbelow(True)
        observed = [row for row in checkpoints if row["family"] == family]
        observed_x = np.log10([float(row["training_pf_days"]) for row in observed])
        minimum, maximum = float(observed_x.min()), float(observed_x.max())
        observed_ranges.append({"family": family, "metric": metric,
                                "min_pf_days": 10 ** minimum, "max_pf_days": 10 ** maximum})
        ax.axvspan(minimum, maximum, color="#edf1f4", zorder=0)
        ax.axvline(maximum, color="#bdc8d1", linewidth=.7, zorder=1)
        ax.scatter(observed_x, [100 * float(row[metric]) for row in observed],
                   s=10, color="#9ca9b5", alpha=.38, edgecolor="none", zorder=2)
        ax.axhline(98, color="#8f9ba7", linewidth=.75, linestyle=(0, (1.3, 2.5)), zorder=1)
        for segment in SEGMENTS:
            fit = next(row for row in fits if (row["family"], row["metric"], row["segment"]) == (family, metric, segment))
            color = COLORS[segment]
            for section, style in (("fit", "-"), ("extrapolation", (0, (4, 3)))):
                selected = [row for row in curves if (row["family"], row["metric"], row["segment"], row["section"]) == (family, metric, segment, section)]
                assert selected
                x = np.array([float(row["log10_pf_days"]) for row in selected])
                y = np.array([float(row["accuracy"]) for row in selected])
                expected = 1 - np.exp(fit["log_A"] - fit["alpha"] * np.log(10) * x)
                assert np.allclose(y, expected, atol=1e-12, rtol=1e-12)
                ax.plot(x, y * 100, color=color, linewidth=1.9, linestyle=style, zorder=4)
            group = [row for row in frontier if (row["family"], row["metric"], row["segment"]) == (family, metric, segment)]
            assert len(group) == fit["points"]
            ax.scatter(np.log10([float(row["training_pf_days"]) for row in group]),
                       [float(row["accuracy"]) * 100 for row in group],
                       s=22, color=color, edgecolor="white", linewidth=.45, zorder=5)
            target = fit["target_98_log10_pf_days"]
            assert abs(1 - np.exp(fit["log_A"] - fit["alpha"] * np.log(10) * target) - .98) < 1e-12
            ax.vlines(target, 0, 98, colors=color, linewidth=.7, alpha=.5,
                      linestyles=(0, (1.2, 3)), zorder=1)
            ax.scatter([target], [98], marker="D", s=37, facecolor="white",
                       edgecolor=color, linewidth=1.4, zorder=6)
            annotations.append(ax.annotate(scientific_label(target), (target, 98),
                                          xytext=(0, 7), textcoords="offset points",
                                          ha="center", va="bottom", color=color, fontsize=12.5))
            crossings.append({"family": family, "metric": metric, "segment": segment,
                              "training_pf_days": 10 ** target, "accuracy": .98})

    # Pin every text artist to the checked Newsreader face, including ticks.
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_text():
            text.set_fontproperties(font_manager.FontProperties(fname=str(FONT), size=text.get_fontsize()))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [legend.get_window_extent(renderer)] + [ax.get_tightbbox(renderer) for ax in axes]
    for index, box in enumerate(boxes):
        assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height
        assert all(not box.overlaps(other) for other in boxes[index + 1:]), "Overlapping panel layout"
    for annotation in annotations:
        box = annotation.get_window_extent(renderer)
        assert annotation.axes.bbox.contains(box.x0, box.y0) and annotation.axes.bbox.contains(box.x1, box.y1)
    destination = OUT / "figures/compute-extrapolation.svg"
    fig.savefig(destination, metadata={"Date": None, "Creator": "Matplotlib; Newsreader embedded as glyph paths"})
    plt.close(fig)
    svg = destination.read_text()
    assert "Newsreader" in svg and "DejaVu" not in svg and "<text" not in svg
    metadata = {
        "source_sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()},
        "fits": fits, "observed_ranges": observed_ranges, "crossings": crossings,
        "fit_parameters_unchanged": True, "frontier_assignments_unchanged": True,
        "curves_verified_against_coefficients": True, "font": "Newsreader; embedded glyph outlines",
        "font_sha256": hashlib.sha256(FONT.read_bytes()).hexdigest(),
        "layout": {"width": 720, "height": 1050, "panels": 3, "rows": 3},
        "omitted_panel": "Bidir full-trajectory accuracy is zero at every evaluated checkpoint.",
    }
    (OUT / "data/compute-extrapolation-fits.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"figure": str(destination), "crossings": len(crossings),
                      "font": "Newsreader", "layout_checked": True}))


if __name__ == "__main__":
    main()

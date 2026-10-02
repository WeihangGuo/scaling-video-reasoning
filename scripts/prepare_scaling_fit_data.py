"""Bundle the paper's unchanged scaling fits for an interactive web figure.

Only Python's standard library is required. Run from any working directory:
    python scripts/prepare_scaling_fit_data.py

The public CSV/JSON snapshots are the inputs. This script independently checks
the strict-record frontiers, equal-count window assignments, and ordinary
least-squares fits, then exports the original coefficients without refitting
or rounding them. The current website's PAPER_DATA supplies raw checkpoints.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math


ROOT = Path(__file__).resolve().parents[1]
SEGMENTS = {"First third": "first", "Middle third": "middle", "Last third": "final"}
DISPLAY_METRICS = {"mean_frame": "frame_accuracy", "full_trajectory": "full_trajectory"}


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def strict_frontier(checkpoints, metric):
    selected = []
    best = -math.inf
    for checkpoint in sorted(checkpoints, key=lambda row: float(row["training_pf_days"])):
        accuracy = float(checkpoint[metric])
        if accuracy > best + 1e-12:
            selected.append(checkpoint)
            best = accuracy
    return selected


def assert_close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-12), (actual, expected)


def check_fit(group, original):
    x = [math.log(float(row["training_pf_days"])) for row in group]
    y = [math.log1p(-float(row["accuracy"])) for row in group]
    mean_x, mean_y = sum(x) / len(x), sum(y) / len(y)
    slope = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y)) / sum((a - mean_x) ** 2 for a in x)
    intercept = mean_y - slope * mean_x
    assert original["points"] == len(group)
    assert_close(intercept, original["log_A"])
    assert_close(-slope, original["alpha"])
    assert_close(math.exp(intercept), original["A"])
    assert_close(float(group[0]["training_pf_days"]), original["min_pf_days"])
    assert_close(float(group[-1]["training_pf_days"]), original["max_pf_days"])
    assert_close((intercept - math.log(.02)) / (-slope * math.log(10)), original["target_98_log10_pf_days"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dist/static/data")
    parser.add_argument("--output", type=Path, default=ROOT / "dist/static/js/scaling-fit-data.js")
    args = parser.parse_args()
    sources = {name: args.data_dir / name for name in (
        "compute-extrapolation-checkpoints.csv",
        "compute-extrapolation-frontier.csv",
        "compute-extrapolation-fits.json",
    )}
    checkpoints = rows(sources["compute-extrapolation-checkpoints.csv"])
    assignments = rows(sources["compute-extrapolation-frontier.csv"])
    original_fits = json.loads(sources["compute-extrapolation-fits.json"].read_text())["fits"]
    assert len(checkpoints) == 424 and len(original_fits) == 6
    assert all(int(row["episodes"]) == 100 for row in checkpoints)
    assert all(math.isclose(float(row["frame_accuracy"]), float(row["mean_frame"]), abs_tol=1e-12) for row in checkpoints)
    panels = []
    for family in ("ar", "bidir"):
        measured = [row for row in checkpoints if row["family"] == family]
        observed_compute = [float(row["training_pf_days"]) for row in measured]
        for source_metric, metric in DISPLAY_METRICS.items():
            records = strict_frontier(measured, source_metric)
            membership = [row for row in assignments if row["family"] == family and row["metric"] == source_metric]
            panel = {
                "family": family,
                "metric": metric,
                "source_metric": source_metric,
                "observed_min_pf_days": min(observed_compute),
                "observed_max_pf_days": max(observed_compute),
                "checkpoint_count": len(measured),
                "frontier": [],
                "fits": [],
                "empty_fit_reason": None,
            }
            if (family, source_metric) == ("bidir", "full_trajectory"):
                assert not membership and all(float(row[source_metric]) == 0 for row in measured)
                panel["empty_fit_reason"] = "Full-trajectory accuracy is zero at all 265 evaluated Bidir checkpoints; no power-law fit is available."
                membership_by_key = {}
            else:
                assert [row["key"] for row in records] == [row["key"] for row in membership]
                assert len(records) % 3 == 0
                third_size = len(records) // 3
                for index, row in enumerate(membership):
                    assert row["segment"] == tuple(SEGMENTS)[index // third_size]
                membership_by_key = {row["key"]: SEGMENTS[row["segment"]] for row in membership}
            for row in records:
                panel["frontier"].append({
                    "key": row["key"],
                    "model": row["model"],
                    "training_videos": int(row["training_videos"]),
                    "training_pf_days": float(row["training_pf_days"]),
                    "accuracy": float(row[source_metric]),
                    "segment": membership_by_key.get(row["key"]),
                })
            for original in original_fits:
                if (original["family"], original["metric"]) != (family, source_metric):
                    continue
                selected = [row for row in membership if row["segment"] == original["segment"]]
                check_fit(selected, original)
                fit = {name: original[name] for name in ("log_A", "A", "alpha", "points", "min_pf_days", "max_pf_days", "target_98_log10_pf_days")}
                fit.update({
                    "segment": SEGMENTS[original["segment"]],
                    "label": "Final third" if original["segment"] == "Last third" else original["segment"],
                    "target_98_pf_days": 10 ** original["target_98_log10_pf_days"],
                })
                panel["fits"].append(fit)
            panels.append(panel)
    payload = {
        "formula": {
            "accuracy": "1 - exp(log_A - alpha * log(C))",
            "compute_for_accuracy": "exp((log_A - log(1 - q)) / alpha)",
            "compute_unit": "PF-days",
            "flops_per_pf_day": 8.64e19,
            "logarithm": "natural",
        },
        "method": {
            "frontier": "Initial baseline and strict record improvements after sorting checkpoints by training compute, separately for each family and metric.",
            "windows": "Equal-count thirds of the complete observed frontier; the paper fits the middle and final thirds separately.",
            "fit": "Ordinary least squares of log(1 - accuracy) against log(training PF-days).",
            "solid_curve": "Within the selected fit window.",
            "dashed_curve": "Outside the selected fit window.",
            "shading": "Compute range covered by evaluated checkpoints.",
        },
        "panels": panels,
        "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()},
        "checks": {"frontiers_verified": True, "window_assignments_verified": True, "ols_coefficients_verified": True, "original_coefficients_preserved": True},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("window.SCALING_FITS = " + json.dumps(payload, separators=(",", ":"), allow_nan=False) + ";\n")
    print(json.dumps({"panels": len(panels), "fits": sum(len(panel["fits"]) for panel in panels), "frontier_points": sum(len(panel["frontier"]) for panel in panels), "checks": payload["checks"]}))


if __name__ == "__main__":
    main()

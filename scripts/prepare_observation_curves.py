"""Bundle the paper's three-face / six-face observation-control curves.

The source CSV contains nine action-boundary accuracies for each observation
setting and generation family. Intervals reproduce the paper figure's clipped
normal intervals over 100 paired episodes. Training exposure is matched across
views within each family, not between AR-k4 and Bidir.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


SITE = Path(__file__).resolve().parents[1]
STATIC = SITE / "dist" / "static"
EPISODES = 100
COLUMNS = (
    "observation", "generation", "action", "frame_accuracy_percent",
    "ci95_low_percent", "ci95_high_percent",
)
SERIES = (
    ("single", "causal", "three-ar", "3-face · AR-k4", "three", "ar", 7_680_000),
    ("dual", "causal", "six-ar", "6-face · AR-k4", "six", "ar", 7_680_000),
    ("single", "bidir", "three-bidir", "3-face · Bidir", "three", "bidir", 1_440_000),
    ("dual", "bidir", "six-bidir", "6-face · Bidir", "six", "bidir", 1_440_000),
)


def bundle(source: Path) -> dict:
    with source.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError("Unexpected observation-curve CSV columns")
        rows = list(reader)
    if len(rows) != 36:
        raise ValueError("Expected 36 observations: four series of nine actions")
    by_key = {}
    for row in rows:
        key = (row["observation"], row["generation"], int(row["action"]))
        if key in by_key:
            raise ValueError(f"Duplicate observation curve point: {key}")
        accuracy, lower, upper = (
            float(row[column]) / 100 for column in COLUMNS[3:]
        )
        if not all(math.isfinite(value) for value in (accuracy, lower, upper)):
            raise ValueError(f"Nonfinite observation curve point: {key}")
        if not 0 <= lower <= accuracy <= upper <= 1:
            raise ValueError(f"Invalid accuracy / confidence interval: {key}")
        if not math.isclose(accuracy * EPISODES, round(accuracy * EPISODES), abs_tol=1e-10):
            raise ValueError(f"Accuracy is not a count out of {EPISODES}: {key}")
        # For binary outcomes, the sample SE is sqrt(p(1-p)/(n-1)).
        error = 1.96 * math.sqrt(accuracy * (1 - accuracy) / (EPISODES - 1))
        if not (math.isclose(lower, max(0, accuracy - error), abs_tol=1e-12)
                and math.isclose(upper, min(1, accuracy + error), abs_tol=1e-12)):
            raise ValueError(f"Confidence interval does not match the paper figure: {key}")
        by_key[key] = {"action": key[2], "accuracy": accuracy, "lower": lower, "upper": upper}

    expected = {(view, generation, action)
                for view, generation, *_ in SERIES for action in range(1, 10)}
    if set(by_key) != expected:
        raise ValueError("Expected every action from 1 to 9 for each observation / generation pair")
    series = [{
        "id": identifier, "label": label, "observation": observation,
        "family": family, "training_videos": training_videos,
        "points": [by_key[(view, generation, action)] for action in range(1, 10)],
    } for view, generation, identifier, label, observation, family, training_videos in SERIES]
    return {
        "episodes": EPISODES,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "confidence_interval": "95% normal interval, clipped to [0, 1]",
        "series": series,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, required=True,
                        help="CSV underlying the paper's observation-control figure")
    args = parser.parse_args()
    data = bundle(args.source_data)
    output_csv = STATIC / "data" / "observation-curves.csv"
    output_js = STATIC / "js" / "observation-curves-data.js"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_js.parent.mkdir(parents=True, exist_ok=True)
    # Explicitly allowlisted columns contain public scientific values only.
    with args.source_data.open(newline="") as stream, output_csv.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in csv.DictReader(stream):
            writer.writerow({column: row[column] for column in COLUMNS})
    output_js.write_text("window.OBSERVATION_CURVES = " +
                         json.dumps(data, separators=(",", ":"), allow_nan=False) + ";\n")
    print(f"Bundled {len(data['series'])} observation-control curves and 36 verified points.")


if __name__ == "__main__":
    main()

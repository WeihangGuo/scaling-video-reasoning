"""Export the verified observation-control example as lossless RGB atlases.

Reads existing EMA predictions and their recorded boundary scores. Only the
three-face simulator reference is rendered; no model inference is performed.
Local provenance is accepted through arguments and never bundled into the
public JavaScript metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image


SITE = Path(__file__).resolve().parents[1]
STATIC = SITE / "dist" / "static"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atlas(frames, name):
    assert frames.dtype == np.uint8 and frames.shape == (81, 256, 256, 3)
    columns = 9
    rows = math.ceil(len(frames) / columns)
    canvas = Image.new("RGB", (256 * columns, 256 * rows))
    for index, frame in enumerate(frames):
        canvas.paste(Image.fromarray(frame), ((index % columns) * 256, (index // columns) * 256))
    relative = f"observation/{name}.webp"
    output = STATIC / relative
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, lossless=True, method=6)
    # Lossless matters for these scored pixels: verify every exported frame.
    with Image.open(output) as image:
        decoded = np.asarray(image.convert("RGB"))
        for index, frame in enumerate(frames):
            x, y = (index % columns) * 256, (index // columns) * 256
            assert np.array_equal(decoded[y:y + 256, x:x + 256], frame)
    return {"image": f"static/{relative}", "width": 256, "height": 256,
            "columns": columns, "rows": rows, "atlas_width": 256 * columns,
            "atlas_height": 256 * rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", type=Path, required=True)
    parser.add_argument("--observation-manifest", type=Path, required=True)
    parser.add_argument("--historical-manifest", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.research_root / "src"))
    from memory_bench.config import RenderConfig
    from memory_bench.games.rubiks_cube import RubiksCubeGame

    figure = read(args.observation_manifest)
    historical = read(args.historical_manifest)
    assert sha(args.historical_manifest) == figure["source_manifest_sha256"]
    models = {"three": "single-causal", "six": "dual-causal"}
    selected, sources, episodes, scored = {}, {}, {}, {}
    for key, model in models.items():
        source = next(item for item in figure["sources"] if item["model"] == model)
        task = next(item for item in historical["tasks"] if item["group"] == "observation" and item["model"] == model)
        folder = Path(task["output_dir"])
        assert sha(folder / "summary.json") == source["summary_sha256"]
        assert sha(folder / "result.json") == source["result_sha256"]
        assert sha(source["rgb_path"]) == source["rgb_sha256"]
        summary, result = read(folder / "summary.json"), read(folder / "result.json")
        assert summary["complete"] and not summary["pilot"] and summary["weights"] == "ema"
        assert summary["loaded_weights"]["parameter_key"] == "ema_parameter_state"
        assert summary["loaded_weights"]["checkpoint_sha256"] == source["checkpoint_sha256"]
        assert result["checkpoint"]["uses_ema"] and result["inference_intervention"] is None
        record = next(item for item in result["records"] if item["seed"] == figure["episode_seed"])
        assert record["inference_intervention"] is None
        assert str(Path(record["generated_video"]).with_suffix(".rgb.npy")) == source["rgb_path"]
        dataset_path = Path(task["manifest"])
        dataset = read(dataset_path)
        entry = next(item for item in dataset["records"] if item["record_id"] == record["record_id"])
        episode_path = dataset_path.parent / entry["record_json"]
        assert sha(episode_path) == entry["record_json_sha256"]
        episode = read(episode_path)
        assert episode["episode"]["actions"] == figure["actions"]
        assert record["state_score"]["frame_schedule"]["action_boundaries"] == figure["settled_frame_indices"]
        assert record["state_score"]["frame_schedule"]["expected_frame_count"] == 81
        selected[key] = {"task": task, "summary": summary, "record": record}
        episodes[key] = episode
        sources[key] = np.load(source["rgb_path"], allow_pickle=False)
        scored[key] = [item for item in record["state_score"]["visible_state"]["by_boundary"]
                       if item["after_action_index"] is not None]
        assert len(scored[key]) == 9

    # View layouts have different dataset identities, so check physical episode
    # and render schedules rather than claiming their record IDs are identical.
    for field in ("game", "seed", "steps", "initial_state", "actions"):
        assert episodes["three"]["episode"][field] == episodes["six"]["episode"][field]
    assert episodes["three"]["render"] == episodes["six"]["render"]
    record = episodes["three"]
    game = RubiksCubeGame(**record["episode"]["game_config"])
    episode = game.make_episode(record["episode"]["seed"], record["episode"]["steps"])
    assert list(episode.actions) == figure["actions"]
    render_fields = ("width", "height", "fps", "intro_seconds", "action_seconds",
                     "outro_seconds", "quality", "show_action_label")
    config = RenderConfig(**{key: record["render"][key] for key in render_fields})
    sources["gt"] = np.asarray(list(game.render_frames(episode, config)))

    metadata = {
        "fps": record["render"]["fps"], "frames": record["render"]["frame_count"],
        "actions": figure["actions"], "boundaries": [], "model_size": "270M",
        "generation": "AR-k4", "weights": "EMA", "latent_frames_per_chunk": 4,
        "reference": "Simulator ground truth (three-face view)",
        "pairing": "Same initial cube state, action sequence, frame schedule, and evaluation episode.",
        "scoring": "Boundary scores measure all visible stickers: 12 in the three-face view and 24 in the six-face view. Intermediate rotation frames are not scored.",
        "selection": "The example used in the paper's observation-control comparison.",
        "compute_note": "PF-days are not reported for these historical observation-control checkpoints. Training exposure counts video presentations, not distinct videos.",
    }
    for index, frame in enumerate(figure["settled_frame_indices"]):
        boundary = {"frame": frame, "action": index + 1}
        for key, total in (("three", 12), ("six", 24)):
            score = scored[key][index]
            assert score["frame_index"] == frame and score["after_action_index"] == index
            assert score["expected_visible_components"] == total
            assert bool(score["exact"]) == (score["correct_components"] == total)
            boundary[key] = {"exact_frame": bool(score["exact"]),
                             "correct_stickers": score["correct_components"],
                             "total_stickers": total}
        metadata["boundaries"].append(boundary)

    for key in ("gt", "three", "six"):
        metadata[key] = atlas(sources[key], key)
        if key == "gt":
            continue
        row = selected[key]
        training = row["task"]["training"]
        model = row["summary"]["loaded_weights"]["model_config"]
        batch = training["effective_batch_size"]
        assert batch == training["microbatch_size_per_gpu"] * training["world_size"] * training["gradient_accumulation"] == 256
        presentations = training["completed_updates"] * batch
        assert presentations == training["samples_this_run"] == 7_680_000
        assert model["segment_frames"] == 4 and model["history_mode"] == "uncompressed_prefix"
        parameters = row["summary"]["loaded_weights"]["ema_parameter_count"]
        assert parameters == 273_872_912
        metadata[key].update({"model_size": "270M", "parameters": parameters,
                              "generation": "AR-k4", "training_updates": training["completed_updates"],
                              "training_video_presentations": presentations,
                              "training_exposure_unit": "video presentations", "training_pf_days": None,
                              "visible_faces": 3 if key == "three" else 6})

    output = STATIC / "js" / "observation-data.js"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("window.OBSERVATION_ROLLOUT = " + json.dumps(metadata, separators=(",", ":"), allow_nan=False) + ";\n")
    assert "/scratch/" not in output.read_text() and "seed" not in output.read_text().lower()
    print(json.dumps({"frames": metadata["frames"], "fps": metadata["fps"],
                      "parameters": metadata["three"]["parameters"], "training_video_presentations": presentations,
                      "three_incorrect_actions": [row["action"] for row in metadata["boundaries"] if not row["three"]["exact_frame"]],
                      "six_incorrect_actions": [row["action"] for row in metadata["boundaries"] if not row["six"]["exact_frame"]],
                      "outputs": [metadata[key]["image"] for key in ("gt", "three", "six")]}))


if __name__ == "__main__":
    main()

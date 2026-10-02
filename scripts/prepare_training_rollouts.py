"""Export a fixed saved episode at successive training checkpoints.

Only saved evaluation RGB arrays and metadata are read. No model is loaded and
no inference is run. Every lossless strip is verified against its source pixels;
all action-boundary scores are recomputed using the registered sticker metric.
Public metadata contains scientific measurements, never local paths or seeds.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from prepare_rollout_gallery import BOUNDARY_FRAMES, load_frames, require, sticker_count


def save_strips(frames: np.ndarray, output: Path, name: str, dist: Path) -> list[str]:
    """Save nine horizontal strips, each containing nine consecutive frames."""
    urls = []
    for row in range(9):
        strip = np.concatenate(frames[row * 9:(row + 1) * 9], axis=1)
        target = output / f"{name}-{row:02d}.webp"
        Image.fromarray(strip).save(target, lossless=True, method=4)
        with Image.open(target) as image:
            require(np.array_equal(np.asarray(image), strip), "Lossless encoding changed source pixels")
        urls.append(target.relative_to(dist).as_posix())
    return urls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progression-manifest", type=Path, required=True,
                        help="Existing checkpoint-progression manifest with evaluation record paths")
    parser.add_argument("--episode-seed", type=int, required=True,
                        help="A saved episode with complete RGB arrays at every selected checkpoint")
    parser.add_argument("--research-root", type=Path, required=True,
                        help="Repository providing src/memory_bench")
    parser.add_argument("--site-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    manifest = json.loads(args.progression_manifest.read_text())
    require(manifest["family"] == "ar" and manifest["model"] == "L"
            and manifest["latent_frames_per_chunk"] == 1 and manifest["weights"] == "ema",
            "Expected the 270M autoregressive EMA progression")

    sys.path.insert(0, str(args.research_root / "src"))
    from memory_bench.config import RenderConfig
    from memory_bench.games.rubiks_cube import RubiksCubeGame

    game = RubiksCubeGame(order=2, reveal_hidden_faces=False, camera_fov_degrees=26.0)
    config = RenderConfig(width=256, height=256, fps=24, intro_seconds=10 / 24,
                          action_seconds=7 / 24, outro_seconds=8 / 24)
    episode = game.make_episode(args.episode_seed, 9)
    actions = list(episode.actions)
    state, masks = tuple(episode.initial_state), []
    for action in actions:
        state = game.apply_action(state, action)
        masks.append(game.visible_sticker_masks(state, config, view="front"))

    dist = args.site_root / "dist"
    output = dist / "static/training-rollouts"
    output.mkdir(parents=True, exist_ok=True)
    with (dist / "static/data/compute-checkpoints.csv").open() as stream:
        compute = {int(row["training_videos"]): row for row in csv.DictReader(stream)
                   if row["family"] == "ar" and row["model"] == "L"}

    entries, jobs, digests = [], [], []
    reference = None
    noise_seed = None
    for ordinal, panel in enumerate(manifest["panels"], start=1):
        exposure = int(panel["training_videos"])
        evaluation = Path(panel["record"]).parents[1]
        record_path = evaluation / f"records/seed-{args.episode_seed}-start-0.json"
        record = json.loads(record_path.read_text())
        summary = json.loads((evaluation / "summary.json").read_text())
        training = summary["loaded_weights"]["training"]
        configuration = summary["loaded_weights"]["model_config"]
        require(summary["complete"] and not summary["pilot"] and summary["weights"] == "ema",
                "Checkpoint evaluation is not a completed non-pilot EMA evaluation")
        require(summary["model"] == "L" and summary["videos"] == exposure
                and training["records_seen"] == exposure, "Checkpoint exposure mismatch")
        require(configuration["segment_frames"] == 1
                and not configuration["context_joint_state_conditioning"],
                "Expected video-only autoregressive k=1 model")
        require(record["weights"] == "ema" and record["identity"]["episode_seed"] == args.episode_seed
                and record["manifest_sha256"] == summary["manifest_sha256"], "Record identity mismatch")
        require(record["window"]["first_action"] == 0 and record["window"]["last_action"] == 9
                and record["window"]["prefix_last_rgb"] == 0
                and record["window"]["prefix_last_latent"] == 0, "Expected a complete free-running rollout")
        require([item["commanded_action"] for item in record["actions"]] == actions,
                "The action sequence must remain fixed across checkpoints")
        require([item["boundary_rgb"] for item in record["actions"]] == BOUNDARY_FRAMES,
                "Action boundaries differ from the frozen renderer schedule")
        current_noise = record["identity"]["sampling_noise_seed"]
        if noise_seed is None:
            noise_seed = current_noise
        require(current_noise == noise_seed, "Sampling noise must remain fixed across checkpoints")

        source = evaluation / f"samples/seed-{args.episode_seed}-start-0-rgb.npy"
        frames = load_frames(source)
        if reference is None:
            reference = load_frames(evaluation / f"samples/seed-{args.episode_seed}-gt-rgb.npy")
            for frame, mask in zip(BOUNDARY_FRAMES, masks):
                require(sticker_count(reference[frame], mask) == (12, True),
                        "Ground-truth reconstruction contains an incorrect sticker state")
        boundaries = []
        for frame, step, mask in zip(BOUNDARY_FRAMES, record["actions"], masks):
            score = sticker_count(frames[frame], mask)
            require(score == (step["correct_stickers"], step["exact_frame"]),
                    "Saved RGB frame does not reproduce the recorded boundary score")
            boundaries.append(dict(frame=frame, action=step["commanded_action"],
                                   correct_stickers=score[0], total_stickers=12, exact_frame=score[1]))
        accounting = compute[exposure]
        require(abs(float(accounting["training_pf_days"]) - panel["training_pf_days"]) < 1e-12,
                "Checkpoint compute differs from the published accounting")
        identifier = f"checkpoint-{ordinal:02d}"
        entries.append(dict(
            id=identifier, model="270M", parameters=int(accounting["parameters"]),
            step=int(training["completed_updates"]), training_videos=exposure,
            training_pf_days=float(accounting["training_pf_days"]),
            fps=24, frames=81, width=256, height=256, columns=9,
            actions=actions, boundaries=boundaries,
            gt_label="Ground-truth reconstruction (VAE)", weights="EMA", generation="AR-k1",
        ))
        jobs.append((frames, identifier))
        digests.append(dict(id=identifier, decoded_rgb_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),
                            record_sha256=hashlib.sha256(record_path.read_bytes()).hexdigest()))

    require(reference is not None and len(entries) == len(manifest["panels"]), "Missing saved checkpoints")
    require(all(a["step"] < b["step"] for a, b in zip(entries, entries[1:])), "Training steps must increase")
    gt_strips = save_strips(reference, output, "ground-truth", dist)
    with ThreadPoolExecutor(max_workers=4) as pool:
        predicted_urls = list(pool.map(lambda item: save_strips(item[0], output, item[1], dist), jobs))
    for entry, urls in zip(entries, predicted_urls):
        entry["gt_strips"] = gt_strips
        entry["pred_strips"] = urls
    (dist / "static/js/training-rollouts-data.js").write_text(
        "window.TRAINING_ROLLOUTS = " + json.dumps(entries, separators=(",", ":"), allow_nan=False) + ";\n")
    provenance = dict(
        model="270M", parameters=manifest["parameters"], frame_count=81, fps=24,
        episode_selection="First episode in the saved evaluation order; fixed across all checkpoints.",
        checkpoint_selection="Fifteen existing checkpoint exposures spanning the saved training progression.",
        fixed_inputs="The same simulator episode, initial observation, action sequence, and sampling-noise seed.",
        image_processing="Unmodified saved RGB pixels, losslessly encoded as nine-frame strips.",
        reference="One shared ground-truth VAE reconstruction of the same simulator episode.",
        training_step_definition="Completed optimizer updates, read from the loaded checkpoint metadata.",
        earliest_available_step=entries[0]["step"], latest_available_step=entries[-1]["step"],
        limitations="No saved step-0 or step-100 full rollout is available in this checkpoint progression.",
        validation="All 135 action-boundary scores reproduced from RGB; all encoded pixels verified lossless.",
        source_digests=digests, checkpoints=entries,
    )
    (dist / "static/data/training-rollouts.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(dict(checkpoints=len(entries), earliest_step=entries[0]["step"],
                         latest_step=entries[-1]["step"], frames_per_checkpoint=81,
                         strips=(len(entries) + 1) * 9,
                         total_bytes=sum(path.stat().st_size for path in output.glob("*.webp")))))


if __name__ == "__main__":
    main()

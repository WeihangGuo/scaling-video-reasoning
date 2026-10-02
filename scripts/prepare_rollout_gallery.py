"""Export 20 saved AR rollouts as lossless full-frame WebP atlases.

Requires numpy, Pillow (with WebP support), and the research renderer package.
Only saved RGB arrays and evaluation records are read; no model is loaded.
The public data contains neither source filesystem paths nor episode seeds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, features


MODEL_LABELS = dict(zip(
    ("XS", "S", "M", "B", "L", "XL", "XXL"),
    ("20M", "30M", "70M", "120M", "270M", "550M", "1B"),
))
# Fixed choices of model, training exposure, and episode ordinal. Selection
# does not inspect the predictions or their accuracy; the paper example leads.
SELECTION = [
    ("XXL", 3_000_000, 0),
    ("XS", 1_000_000, 0), ("XS", 3_000_000, 1), ("XS", 5_000_000, 2),
    ("S", 1_000_000, 1), ("S", 3_000_000, 2), ("S", 5_000_000, 0),
    ("M", 1_000_000, 2), ("M", 3_000_000, 0), ("M", 5_000_000, 1),
    ("B", 1_000_000, 0), ("B", 3_000_000, 1), ("B", 5_000_000, 2),
    ("L", 1_000_000, 1), ("L", 3_000_000, 2), ("L", 5_000_000, 0),
    ("XL", 1_000_000, 2), ("XL", 3_000_000, 0), ("XL", 5_000_000, 1),
    ("XXL", 5_000_000, 2),
]
BOUNDARY_FRAMES = list(range(16, 73, 7))
FRAME_SHAPE = (81, 256, 256, 3)
FACE_NAMES = ("R", "L", "U", "D", "F", "B")
PALETTE = np.asarray([
    (196, 35, 48), (245, 116, 25), (242, 242, 235),
    (247, 205, 42), (27, 150, 76), (35, 91, 183),
], dtype=np.float32)
PALETTE /= np.linalg.norm(PALETTE, axis=1, keepdims=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sticker_count(frame: np.ndarray, masks: tuple) -> tuple[int, bool]:
    """Registered RGB metric, matching evaluate_scaling_rollouts.py exactly."""
    require(len(masks) == 12, "Scored observation must contain 12 stickers")
    correct, decoded = 0, 0
    for expected, mask in masks:
        require(mask.shape == frame.shape[:2] and bool(mask.any()), "Invalid sticker mask")
        median = np.median(frame[mask], axis=0).astype(np.float32)
        norm = float(np.linalg.norm(median))
        scores = (median / (norm + 1e-6)) @ PALETTE.T
        predicted = int(scores.argmax())
        readable = norm >= 40 and float(scores[predicted]) >= .90
        decoded += int(readable)
        correct += int(readable and FACE_NAMES[predicted] == expected)
    return correct, bool(decoded == 12 and correct == 12)


def load_frames(path: Path) -> np.ndarray:
    frames = np.load(path, mmap_mode="r", allow_pickle=False)
    require(frames.shape == FRAME_SHAPE and frames.dtype == np.uint8,
            "Every saved rollout must contain 81 full 256 x 256 RGB frames")
    return frames


def save_atlas(frames: np.ndarray, path: Path) -> None:
    atlas = frames.reshape(9, 9, 256, 256, 3).transpose(0, 2, 1, 3, 4).reshape(2304, 2304, 3)
    # In lossless mode quality controls compression effort, not pixel fidelity.
    Image.fromarray(atlas).save(path, format="WEBP", lossless=True, method=6, quality=80)
    # Validate every source pixel after the actual browser-format encoding.
    with Image.open(path) as decoded:
        require(decoded.mode == "RGB" and decoded.size == (2304, 2304), "Invalid atlas geometry")
        require(np.array_equal(np.asarray(decoded), atlas), "Lossless atlas changed source pixels")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-b-root", type=Path, required=True,
                        help="Directory containing model/training_videos/evaluation folders")
    parser.add_argument("--research-root", type=Path, required=True,
                        help="Research repository providing src/memory_bench")
    parser.add_argument("--episode-seeds", type=int, nargs=3, required=True,
                        help="Three saved evaluation episode seeds in original evaluation order")
    parser.add_argument("--site-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--checkpoints", type=Path,
                        help="Defaults to site-root/dist/static/data/compute-checkpoints.csv")
    args = parser.parse_args()
    require(features.check("webp"), "Pillow must include WebP support")
    require(len(set(args.episode_seeds)) == 3, "Provide three distinct episode seeds")
    sys.path.insert(0, str(args.research_root / "src"))
    from memory_bench.config import RenderConfig
    from memory_bench.games.rubiks_cube import RubiksCubeGame

    static = args.site_root / "dist/static"
    out = static / "rollouts"
    out.mkdir(parents=True, exist_ok=True)
    (static / "js").mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoints or static / "data/compute-checkpoints.csv"
    with checkpoint_path.open() as stream:
        checkpoints = {(row["model"], int(row["training_videos"])): row
                       for row in csv.DictReader(stream) if row["family"] == "ar"}

    game = RubiksCubeGame(order=2, reveal_hidden_faces=False, camera_fov_degrees=26.0)
    config = RenderConfig(width=256, height=256, fps=24, intro_seconds=10 / 24,
                          action_seconds=7 / 24, outro_seconds=8 / 24)
    require((config.intro_frames, config.action_frames, config.outro_frames) == (10, 7, 8),
            "Renderer must match the saved evaluation frame schedule")
    mask_sets = {}
    for seed in args.episode_seeds:
        episode = game.make_episode(seed, 9)
        state, masks = tuple(episode.initial_state), []
        for action in episode.actions:
            state = game.apply_action(state, action)
            masks.append(game.visible_sticker_masks(state, config, view="front"))
        mask_sets[seed] = (list(episode.actions), masks)

    gallery, references, source_identities = [], {}, set()
    for index, (model, exposure, ordinal) in enumerate(SELECTION, start=1):
        seed = args.episode_seeds[ordinal]
        evaluation = args.stage_b_root / model / str(exposure) / "evaluation"
        prefix = f"seed-{seed}"
        record = read_json(evaluation / f"records/{prefix}-start-0.json")
        summary = read_json(evaluation / "summary.json")
        checkpoint = checkpoints[(model, exposure)]
        require(summary["complete"] and not summary["pilot"] and summary["weights"] == "ema",
                "Only completed non-pilot EMA evaluation is eligible")
        require(summary["model"] == model and summary["videos"] == exposure,
                "Evaluation metadata differs from selected checkpoint")
        require(summary["loaded_weights"]["training"]["records_seen"] == exposure,
                "Training exposure must match loaded weights")
        configuration = summary["loaded_weights"]["model_config"]
        require(configuration["segment_frames"] == 1 and not configuration["context_joint_state_conditioning"],
                "Gallery expects video-only autoregressive k=1 models")
        require(record["weights"] == "ema" and record["identity"]["episode_seed"] == seed,
                "Record identity or weight selection mismatch")
        require(record["manifest_sha256"] == summary["manifest_sha256"], "Evaluation manifest mismatch")
        require(record["window"]["first_action"] == 0 and record["window"]["last_action"] == 9
                and record["window"]["prefix_last_rgb"] == 0 and record["window"]["prefix_last_latent"] == 0,
                "Only full nine-action free-running rollouts are eligible")
        identity = (model, exposure, seed)
        require(identity not in source_identities, "Duplicate saved example")
        source_identities.add(identity)

        generated = load_frames(evaluation / f"samples/{prefix}-start-0-rgb.npy")
        # The reference depends only on the original episode, not on the video
        # model. Reuse one saved VAE reconstruction per episode from the paper's
        # default checkpoint; separate decodes can differ by rounding pixels.
        reference_source = args.stage_b_root / "XXL/3000000/evaluation"
        reference_record = read_json(reference_source / f"records/{prefix}-start-0.json")
        require(reference_record["identity"]["episode_seed"] == seed,
                "Shared ground-truth reference episode mismatch")
        reference = load_frames(reference_source / f"samples/{prefix}-gt-rgb.npy")
        actions, masks = mask_sets[seed]
        require([step["action_index"] for step in record["actions"]] == list(range(1, 10)),
                "Evaluation action indices must cover all nine actions")
        require([step["boundary_rgb"] for step in record["actions"]] == BOUNDARY_FRAMES,
                "Evaluation boundaries differ from the frozen frame schedule")
        require([step["commanded_action"] for step in record["actions"]] == actions,
                "Saved record action prompt differs from renderer episode")
        require([step["commanded_action"] for step in reference_record["actions"]] == actions,
                "Shared ground-truth reference action prompt mismatch")
        boundaries = []
        for frame, step, mask in zip(BOUNDARY_FRAMES, record["actions"], masks):
            score = sticker_count(generated[frame], mask)
            require(score == (step["correct_stickers"], step["exact_frame"]),
                    "Saved generated frame does not reproduce recorded sticker score")
            require(sticker_count(reference[frame], mask) == (12, True),
                    "Ground-truth VAE reconstruction has incorrect visible states")
            boundaries.append(dict(frame=frame, action=step["commanded_action"],
                                   correct_stickers=score[0], total_stickers=12, exact_frame=score[1]))

        reference_hash = hashlib.sha256(reference.tobytes()).hexdigest()
        if reference_hash not in references:
            gt_name = f"ground-truth-{len(references) + 1:02d}.webp"
            save_atlas(reference, out / gt_name)
            references[reference_hash] = gt_name
        gt_name = references[reference_hash]
        identifier = f"rollout-{index:02d}"
        pred_name = identifier + ".webp"
        save_atlas(generated, out / pred_name)
        gallery.append(dict(
            id=identifier, model=MODEL_LABELS[model], training_videos=exposure,
            training_pf_days=float(checkpoint["training_pf_days"]),
            fps=24, frames=81, width=256, height=256, columns=9,
            gt=f"static/rollouts/{gt_name}", pred=f"static/rollouts/{pred_name}",
            actions=actions, boundaries=boundaries,
            gt_label="Ground-truth reconstruction (VAE)", weights="EMA", generation="AR-k1",
        ))
        print(f"{identifier}: {MODEL_LABELS[model]}, {exposure:,} videos, "
              f"{(out / pred_name).stat().st_size / 1_000_000:.2f} MB lossless", flush=True)

    require(len(gallery) == 20 and len(source_identities) == 20, "Expected 20 distinct saved examples")
    require(gallery[0]["model"] == "1B" and gallery[0]["training_videos"] == 3_000_000,
            "First gallery entry must be the paper's 1B/3M example")
    require(len({tuple(item["actions"]) for item in gallery}) == 3, "Expected three original action prompts")
    (static / "js/rollout-gallery-data.js").write_text(
        "window.ROLLOUT_GALLERY = " + json.dumps(gallery, separators=(",", ":"), allow_nan=False) + ";\n")
    asset_names = {Path(item[k]).name for item in gallery for k in ("gt", "pred")}
    total_bytes = sum((out / name).stat().st_size for name in asset_names)
    print(json.dumps(dict(examples=20, gt_atlases=len(references), atlas_bytes=total_bytes,
                          atlas_mb=round(total_bytes / 1_000_000, 2),
                          validated_boundary_pairs=180, lossless=True)))


if __name__ == "__main__":
    main()

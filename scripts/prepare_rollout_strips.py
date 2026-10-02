"""Split the gallery's existing RGB atlases into lossless nine-frame strips.

No scientific images are generated or modified: each output is one horizontal
row cropped from an existing atlas, with decoded RGB equality checked before
the gallery metadata is updated. Original atlas URLs remain available.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
from PIL import Image


SITE = Path(__file__).resolve().parents[1]
DIST = SITE / "dist"
GALLERY = DIST / "static/js/rollout-gallery-data.js"
PREFIX = "window.ROLLOUT_GALLERY = "


def export_strips(relative):
    source = DIST / relative
    with Image.open(source) as image:
        original = np.asarray(image.convert("RGB")).copy()
    assert original.shape == (2304, 2304, 3)
    outputs = []
    recovered = []
    total_bytes = 0
    for index in range(9):
        row = original[index * 256:(index + 1) * 256]
        output = DIST / "static/rollouts/strips" / f"{source.stem}-{index:02d}.webp"
        Image.fromarray(row).save(output, lossless=True, method=4)
        with Image.open(output) as image:
            decoded = np.asarray(image.convert("RGB")).copy()
        assert decoded.shape == (256, 2304, 3)
        assert np.array_equal(decoded, row), output
        recovered.append(decoded)
        outputs.append(output.relative_to(DIST).as_posix())
        total_bytes += output.stat().st_size
    # This compares all 81 frames and every pixel, including unscored frames.
    assert np.array_equal(np.concatenate(recovered, axis=0), original), source
    return relative, outputs, source.stat().st_size, total_bytes


def main():
    text = GALLERY.read_text()
    assert text.startswith(PREFIX)
    gallery = json.loads(text[len(PREFIX):].strip().removesuffix(";"))
    assert len(gallery) == 20 and len({row["id"] for row in gallery}) == 20
    for row in gallery:
        assert (row["frames"], row["width"], row["height"], row["columns"]) == (81, 256, 256, 9)
    sources = sorted({row[key] for row in gallery for key in ("gt", "pred")})
    assert len(sources) == 23
    (DIST / "static/rollouts/strips").mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(export_strips, sources))
    urls = {source: strips for source, strips, _, _ in results}
    for row in gallery:
        row["gt_strips"] = urls[row["gt"]]
        row["pred_strips"] = urls[row["pred"]]
    GALLERY.write_text(PREFIX + json.dumps(gallery, separators=(",", ":"), allow_nan=False) + ";\n")
    print(json.dumps({
        "gallery_entries": len(gallery), "source_atlases": len(sources),
        "lossless_strips": len(sources) * 9, "strip_dimensions": [2304, 256],
        "all_81_frames_pixel_identical": True,
        "original_total_bytes": sum(row[2] for row in results),
        "strip_total_bytes": sum(row[3] for row in results),
        "largest_strip_bytes": max((DIST / path).stat().st_size for paths in urls.values() for path in paths),
    }))


if __name__ == "__main__":
    main()

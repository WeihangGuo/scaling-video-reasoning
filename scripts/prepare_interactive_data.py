"""Bundle measured results and unlabelled frames for the interactive website."""
from pathlib import Path
import argparse
import csv
import hashlib
import itertools
import json
import sys
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / 'dist/static'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--research-root', type=Path, required=True)
parser.add_argument('--progression-manifest', type=Path, required=True)
args = parser.parse_args()
RESEARCH = args.research_root
sys.path.insert(0, str(RESEARCH / 'src'))
from memory_bench.config import RenderConfig
from memory_bench.games.rubiks_cube import RubiksCubeGame

def read_csv(name):
    with (STATIC / 'data' / name).open() as stream:
        return list(csv.DictReader(stream))

numeric = ['parameters', 'training_videos', 'training_pf_days', 'mse',
           'frame_accuracy', 'full_trajectory', 'action_following', 'sticker_accuracy']
checkpoints = []
for record in read_csv('compute-checkpoints.csv'):
    row = {key: record[key] for key in ['family', 'model', 'key']}
    row.update({key: float(record[key]) for key in numeric})
    checkpoints.append(row)

guidance = []
for record in read_csv('state-guidance.csv'):
    row = {'method': record['method']}
    row.update({key: float(record[key]) for key in record if key not in ['method', 'weights']})
    row['actions'] = [
        {'action': int(item['action_index']), 'accuracy': float(item['frame_accuracy'])}
        for item in read_csv('state-guidance-per-action.csv') if item['method'] == record['method']
    ]
    guidance.append(row)

manifest_path = args.progression_manifest
manifest = json.loads(manifest_path.read_text())
progression = []
for index, panel in enumerate(manifest['panels']):
    relative = f'images/training-{index}.webp'
    with Image.open(panel['rgb']) as image:
        image.save(STATIC / relative, lossless=True)
    progression.append({key: panel[key] for key in ['training_videos', 'training_pf_days', 'correct_stickers', 'exact_frame']})
    progression[-1]['image'] = f'static/{relative}'

game = RubiksCubeGame(order=2, reveal_hidden_faces=False, camera_fov_degrees=26.0)
config = RenderConfig(width=256, height=256, fps=24, intro_seconds=10/24, action_seconds=7/24, outro_seconds=8/24)
episode = game.make_episode(manifest['episode_seed'], 9)
record = json.loads(Path(manifest['panels'][0]['record']).read_text())
assert list(episode.actions) == [action['commanded_action'] for action in record['actions']]
frame = next(itertools.islice(game.render_frames(episode, config), 37, 38))
state = tuple(episode.initial_state)
for action in episode.actions[:4]:
    state = game.apply_action(state, action)
masks = game.visible_sticker_masks(state, config, view='front')
palette_names = ('R', 'L', 'U', 'D', 'F', 'B')
palette = np.array([(196,35,48),(245,116,25),(242,242,235),(247,205,42),(27,150,76),(35,91,183)], dtype=np.float32)
palette /= np.linalg.norm(palette, axis=1, keepdims=True)
def count_correct(image):
    count = 0
    for expected, mask in masks:
        median = np.median(image[mask], axis=0).astype(np.float32)
        norm = float(np.linalg.norm(median))
        scores = (median / (norm + 1e-6)) @ palette.T
        predicted = int(scores.argmax())
        count += int(norm >= 40 and scores[predicted] >= .90 and palette_names[predicted] == expected)
    return count
assert count_correct(frame) == 12
for panel in manifest['panels']:
    assert count_correct(np.array(Image.open(panel['rgb']))) == panel['correct_stickers']
Image.fromarray(frame).save(STATIC / 'images/training-gt.webp', lossless=True)

data = {'checkpoints': checkpoints, 'guidance': guidance, 'progression': progression}
(STATIC / 'js/results-data.js').write_text('window.PAPER_DATA = ' + json.dumps(data, separators=(',', ':'), allow_nan=False) + ';\n')
public_provenance = {
    'action_number': 4, 'action': 'L', 'model_parameters': 271760656,
    'reference': 'Simulator ground truth', 'selection': manifest['selection'],
    'reference_note': 'Re-rendered from the verified episode and renderer contract; all 12 states and all displayed checkpoint scores verified against the recorded evaluation.',
    'checkpoints': progression,
}
(STATIC / 'data/training-progression.json').write_text(json.dumps(public_provenance, indent=2) + '\n')
print(f'Bundled {len(checkpoints)} measured checkpoints, {len(guidance)} symbolic-guidance variants, and {len(progression)} progression frames.')

"""Prepare web-only figures in Newsreader; never modify manuscript assets.

Run with an environment providing numpy, Pillow, matplotlib and fontTools.
Source data snapshots are retained in dist/static/data for reproducibility.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import shutil

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter, LogLocator, FuncFormatter
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'dist/static'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--paper-root', type=Path, required=True)
parser.add_argument('--guidance-data', type=Path, required=True)
parser.add_argument('--rollout-dir', type=Path, required=True)
parser.add_argument('--episode-seed', type=int, required=True)
args = parser.parse_args()
PAPER, GUIDANCE, ROLLOUT = args.paper_root, args.guidance_data, args.rollout_dir
for folder in ['data', 'images', 'videos']:
    (OUT / folder).mkdir(parents=True, exist_ok=True)

sources = {
    'scaling-observations.csv': PAPER / 'data/scaling_results_v1/ar_bidir_observations.csv',
    'compute-checkpoints.csv': PAPER / 'data/scaling_results_v1/state_compute_segmented_checkpoints.csv',
    'state-guidance.csv': GUIDANCE / 'state_guidance_xs_k1_results.csv',
    'state-guidance-per-action.csv': GUIDANCE / 'state_guidance_xs_k1_per_action.csv',
}
provenance = {}
for name, source in sources.items():
    shutil.copyfile(source, OUT / 'data' / name)
    provenance[name] = {'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}

font = TTFont(OUT / 'fonts/Newsreader.ttf')
static_font = instantiateVariableFont(font, {'opsz': 24, 'wght': 400}, inplace=False)
static_path = ROOT / 'scripts/Newsreader-plot.ttf'
static_font.save(static_path)
font_manager.fontManager.addfont(str(static_path))
assert font_manager.FontProperties(fname=str(static_path)).get_name() == 'Newsreader'
plt.rcParams.update({
    'font.family': 'Newsreader', 'font.size': 13, 'axes.titlesize': 17,
    'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'text.color': '#172539', 'axes.labelcolor': '#172539',
    'xtick.color': '#506075', 'ytick.color': '#506075',
    'axes.edgecolor': '#c9d3df', 'axes.spines.top': False, 'axes.spines.right': False,
    'svg.fonttype': 'path', 'svg.hashsalt': 'scaling-video-reasoning',
    'axes.unicode_minus': False, 'figure.facecolor': 'white',
})
colors = ['#718096', '#3f728a', '#19669a', '#198175', '#bc8a26', '#c75b38', '#7e4994']
models = ['XS', 'S', 'M', 'B', 'L', 'XL', 'XXL']
labels = ['20M', '30M', '70M', '120M', '270M', '550M', '1B']

def rows(name):
    with (OUT / 'data' / name).open() as f:
        return list(csv.DictReader(f))

def finish(fig, name):
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_text():
            text.set_fontproperties(font_manager.FontProperties(fname=str(static_path), size=text.get_fontsize()))
    fig.savefig(OUT / 'images' / name, format='svg', metadata={'Date': None, 'Creator': 'Matplotlib; Newsreader embedded as glyph paths'})
    plt.close(fig)

def accuracy_axis(ax):
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.grid(axis='y', color='#e6ecf2', linewidth=.7)
    ax.set_axisbelow(True)

observations = [r for r in rows('scaling-observations.csv') if r['family'] == 'ar']
assert set(r['model'] for r in observations) == set(models)
fig, ax = plt.subplots(figsize=(11, 4.4))
fig.subplots_adjust(left=.09, right=.96, bottom=.17, top=.72)
for model, label, color in zip(models, labels, colors):
    selected = [r for r in observations if r['model'] == model]
    ax.scatter([float(r['mse']) for r in selected], [float(r['frame_accuracy']) * 100 for r in selected], s=22, color=color, alpha=.8, label=label, linewidths=0)
ax.set_xscale('log')
ax.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: f'{value:g}'))
ax.set_xlabel('Validation flow MSE')
ax.set_ylabel('Frame accuracy')
accuracy_axis(ax)
fig.legend(*ax.get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.52,.98), ncol=7, frameon=False, title='Model size (parameters)', columnspacing=1.1, handletextpad=.3)
finish(fig, 'loss-accuracy.svg')

checkpoints = [r for r in rows('compute-checkpoints.csv') if r['family'] == 'ar']
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
fig.subplots_adjust(left=.075, right=.97, bottom=.19, top=.68, wspace=.15)
for ax, metric, title in zip(axes, ['mean_frame', 'full_trajectory'], ['Average frame accuracy', 'Full-trajectory accuracy']):
    for model, label, color in zip(models, labels, colors):
        selected = sorted([r for r in checkpoints if r['model'] == model], key=lambda r: float(r['training_pf_days']))
        ax.plot([float(r['training_pf_days']) for r in selected], [float(r[metric]) * 100 for r in selected], color=color, label=label, marker='o', markersize=2.6, linewidth=1.4, alpha=.9)
    accuracy_axis(ax)
    ax.set_xscale('log')
    ax.set_xlabel('Training compute (PF-days)')
    ax.set_title(title, pad=12)
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: f'{value:g}'))
fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.53,1), ncol=7, frameon=False, title='Model size (parameters)', columnspacing=1.0, handlelength=1.4)
finish(fig, 'compute-accuracy.svg')

results = rows('state-guidance.csv')
per_action = rows('state-guidance-per-action.csv')
method_colors = ['#7b8797', '#19669a', '#9b6539']
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={'width_ratios': [1, 1.1]})
fig.subplots_adjust(left=.075, right=.96, bottom=.25, top=.8, wspace=.24)
for i, (r, color) in enumerate(zip(results, method_colors)):
    value = float(r['frame_accuracy']) * 100
    axes[0].bar(i, value, width=.57, color=color)
    axes[0].text(i, value+3, f'{value:.1f}%', ha='center', fontsize=18)
axes[0].set_xticks([0,1,2], ['Video\nonly', 'Front12\nfeedback', 'Full24\nfeedback'])
axes[0].set_title('Average frame accuracy', pad=14)
for r, color in zip(results, method_colors):
    selected = sorted([item for item in per_action if item['method'] == r['method']], key=lambda item: int(item['action_index']))
    axes[1].plot([int(item['action_index']) for item in selected], [float(item['frame_accuracy'])*100 for item in selected], marker='o', markersize=4, color=color, linewidth=2, label=r['method'])
axes[1].set_xticks(range(1,10))
axes[1].set_xlabel('Action number')
axes[1].set_title('Accuracy through the sequence', pad=14)
axes[1].legend(loc='upper center', bbox_to_anchor=(.47,-.31), ncol=3, frameon=False, fontsize=10, handlelength=1, columnspacing=.9)
for ax in axes:
    accuracy_axis(ax)
finish(fig, 'state-guidance.svg')

prefix = f'seed-{args.episode_seed}'
record = json.loads((ROLLOUT / f'records/{prefix}-start-0.json').read_text())
generated = np.load(ROLLOUT / f'samples/{prefix}-start-0-rgb.npy', mmap_mode='r')
ground_truth = np.load(ROLLOUT / f'samples/{prefix}-gt-rgb.npy', mmap_mode='r')
assert generated.shape == ground_truth.shape == (81, 256, 256, 3)
steps = []
for action in record['actions']:
    index, boundary = action['action_index'], action['boundary_rgb']
    for name, frames in [('pred', generated), ('gt', ground_truth)]:
        Image.fromarray(frames[boundary]).save(OUT / 'images' / f'{name}-{index}.webp', lossless=True)
    steps.append({'action': action['commanded_action'], 'correct_stickers': action['correct_stickers'], 'exact_frame': action['exact_frame']})
assert all(step['exact_frame'] for step in steps[:5]) and not any(step['exact_frame'] for step in steps[5:])
(OUT / 'js/rollout-data.js').write_text('window.ROLLOUT_DATA = ' + json.dumps(steps) + ';\n')
shutil.copyfile(ROLLOUT / f'samples/{prefix}-start-0.mp4', OUT / 'videos/rollout.mp4')
(ROOT / 'asset-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
print('Prepared three Newsreader figures, 18 boundary frames, one original rollout video, and source CSVs.')

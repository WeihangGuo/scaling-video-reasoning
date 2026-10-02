# Scaling Video Generation for Reasoning: At What Cost?

Academic project website for [arXiv:2609.36599](https://arxiv.org/abs/2609.36599).

Weihang Guo, Xiaoyu Wu, Yifei Wang, Niloofar Mireshghallah, Lydia E. Kavraki. Rice University and Carnegie Mellon University.

## Website

The static website lives in `dist/`. It uses a single-column article layout with descriptive headings, a contents list, and figures next to their discussion. The page retains the author links, attribution, and static structure adapted from [Academic-project-page-template](https://github.com/eliahuhorwitz/Academic-project-page-template). The exact upstream revision is recorded in `template-provenance.json`.

Page typography uses locally hosted **Newsreader**, including buttons, citations, and interactive SVG chart labels. Static fallback figures embed Newsreader glyph outlines. At the user's request, the symbolic-guidance workflow uses the original paper artwork with its original embedded typography. Original manuscript figures and manuscript files are not modified.

Preview locally:

```sh
python3 -m http.server 8000 --directory dist
```

Open `http://localhost:8000`.

Edit `dist/index.html`, the styles in `dist/static/css/`, and interaction code in `dist/static/js/`. No frontend build or package installation is needed to serve the site.

GitHub Pages publishes the contents of `dist/` using `.github/workflows/pages.yml`. In the repository's **Settings → Pages**, set **Source** to **GitHub Actions**. Pushing to `main` then deploys the website; the workflow can also be run manually from the Actions tab. The project URL is [weihangguo.github.io/scaling-video-reasoning](https://weihangguo.github.io/scaling-video-reasoning/). All bundled asset paths are relative to support this project subdirectory.

## Content and evidence

The article introduces the task and evaluation, then the interactive compute-frontier extrapolation near the start. The prediction player follows, then loss versus correctness, compute comparisons by model size, representation and visibility controls, and symbolic guidance. It ends with the open question of learning generalizable state abstractions: discrete, continuous, or hybrid. JEPA is mentioned as a possible direction for learned representation-space prediction, not as an established symbolic-state solution.

Interactive views cover complete frame-by-frame rollouts, 15 complete 270M training rollouts, loss versus accuracy for AR and bidirectional models, measured compute curves, a synchronized observation comparison, and three symbolic-guidance variants. Sliders support pointer, touch and keyboard use. The compute curves connect 424 evaluated checkpoints; their highlighted best-within-budget points select actual measurements without interpolation. The scaling-law explorer preserves the paper's fitted coefficients and fitting windows. Readers can change the generation family, accuracy criterion, fit region, compute probe, and target accuracy. The original static extrapolation plot remains available in a disclosure below it.

- Random switches among 20 saved AR-k1 rollouts, spanning seven model sizes, three training exposures, and three action sequences. It does not generate new video. Each example preserves all 81 original RGB frames at 24 fps, including the initial image. Metadata records its training exposure, actual PF-days, action sequence, and nine boundary scores. Selection is independent of accuracy. The alternate model/example gallery opens on the paper's 1B model after 3M training videos; the page's default view is now the 270M training-checkpoint comparison. Ground-truth atlases are labeled VAE reconstructions; three shared references are drawn from the default checkpoint's saved reference videos and verified against every paired episode's states.
- Incorrect predictions receive a red outline only on scored action-boundary frames. Intermediate rotation frames are viewable but unscored. Every gallery boundary was rescored against the simulator and matched to its saved evaluation; every lossless atlas was decoded and checked against the source RGB pixels.
- The observation comparison synchronizes three-face simulator ground truth, three-face prediction, and six-face prediction from the paper's separate 270M AR-k4 experiment. Both models received 7.68M training video presentations. Its historical evaluation does not report PF-days, so none are invented. Three-face correctness checks 12 stickers; six-face correctness checks 24.
- The compact observation chart includes all four paper curves: three-face and six-face accuracy for AR-k4 and bidirectional generation, with the original 95% confidence intervals across 100 paired episodes. Training exposure is matched between views within each generation family (7.68M for AR-k4; 1.44M for bidirectional). Pointer and keyboard inspection report all four measured accuracies at a selected action. `scripts/prepare_observation_curves.py` exports its CSV and browser data from the paper's curve snapshot.
- Plot CSVs retain exact parameter counts. Display labels are 20M, 30M, 70M, 120M, 270M, 550M, and 1B.
- Symbolic guidance compares the 20M AR-k1 baseline with Front12 and Full24 feedback after 3M training videos, using the same 100 evaluation episodes. Both guided variants supervise all 24 stickers.
- The training progression uses the same saved episode, action sequence, and sampling noise at 15 checkpoints (steps 2,048–500,000; 32,768–8M training video presentations). The training comparison opens on the last saved checkpoint (step 500,000), with a step selector, Previous/Next buttons, and a training slider above the video. Returning from the other-models tab preserves the checkpoint the reader selected. A separate frame slider remains below the video. Changing checkpoint preserves the selected frame. All 81 frames are available at each step; the nine boundary scores determine the red error outlines. Step 0 and step 100 rollouts were not saved. `scripts/prepare_training_rollouts.py` exports lossless nine-frame strips and verifies all pixels and boundary scores. The earlier single-frame progression exports remain as source assets but are no longer the training interaction.
- Author order and affiliations follow the public arXiv v1 manuscript, submitted 29 September 2026.

`asset-provenance.json` records hashes of the bundled data snapshots; `scripts/prepare_assets.py` regenerates the web figures and frame exports from user-supplied research paths. It requires NumPy, Pillow, Matplotlib, and fontTools and does not run model training or inference. Source CSV snapshots are bundled under `dist/static/data/`. Run each preparation script with `--help` for its required input paths; internal filesystem locations and run identifiers are excluded from this public repository.

`scripts/prepare_interactive_data.py` bundles chart data and the original single-frame training exports. `dist/static/data/training-progression.json` records the example's selection and rendering provenance. Existing `state-guidance` filenames are retained as source-data identifiers; visible website terminology is **symbolic guidance**.

`scripts/prepare_rollout_gallery.py` exports the 20-example gallery, `scripts/prepare_observation_rollout.py` exports the synchronized observation control, and `scripts/prepare_extrapolation.py` redraws the paper's fitted curves with Newsreader. Run each with `--help` for source inputs. These scripts only read saved research outputs; they never run inference or training.

`scripts/prepare_rollout_strips.py` splits the gallery atlases into lossless nine-frame strips and verifies every decoded pixel against the original exports. Random initially loads only the pair of strips containing the current frame (at most about 0.92 MB, versus about 8 MB for two complete atlases), then loads subsequent strips as needed. The toolbar immediately identifies the requested example; another click supersedes pending requests. Loads have a timeout and recoverable error state, and the cache is bounded. The observation comparison loads when it approaches the viewport. Full gallery atlases remain available as source exports and as a compatibility fallback.

The representation comparison uses native HTML/CSS. The symbolic-guidance workflow displays panel (a) extracted from the paper's original `joint_state_video_guidance.svg`, preserving its vector paths, arrows, labels, colors, and font outlines. The complete unchanged original is available through the figure's full-size link. Interactive method selection below the figure controls the result curves and readouts.

## Attribution

Author homepage links follow the paper entry on [Weihang Guo's personal website](https://www.whguo.me/). The Rice shield preserves the original shield paths and colors from [Rice's official horizontal university logo](https://www.rice.edu/sites/g/files/bxs2566/files/2019-08/Rice_University_Horizontal_Blue.svg), with the adjacent wordmark omitted. The CMU square logo is the user's supplied image, copied unchanged. The Newsreader typography setting applies to page text.

Website template adaptations: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), following Academic-project-page-template, which is partly based on [Nerfies](https://nerfies.github.io/). Bulma is MIT licensed. Newsreader uses the SIL Open Font License, included in `dist/static/fonts/OFL.txt`. These website notices do not relicense the paper, research data, or scientific media.

## Article and scaling explorer

The October 2026 article revision keeps Newsreader and the existing scientific media, with a plain reading column and smaller headings. It removes the poster slogans and section color bands. The broad reading-layout reference is [Lil’Log](https://lilianweng.github.io/); all article prose is original.

`dist/static/css/blog.css` supplies the article layout. `scripts/prepare_scaling_fit_data.py` derives `dist/static/js/scaling-fit-data.js` from the public CSV and original fit snapshot, independently checking the frontier/window assignments and ordinary-least-squares coefficients. The browser uses `a(C) = 1 − exp(log_A − alpha × log(C))` and its inverse. The compute frontier is a staircase of measured record improvements, never interpolated accuracy. Solid curves cover each fit window; dashed portions extend it. Negative values produced outside a fit window are not clamped into valid accuracies. The bidirectional full-trajectory setting has no fitted curve because all its measured values are zero.

The training player manifest is `dist/static/js/training-rollouts-data.js`; `dist/static/data/training-rollouts.json` records its selection, source verification, and asset hashes. Its initial load requests only the two strips needed for the selected frame. The complete 15-checkpoint sequence is not downloaded at page load.

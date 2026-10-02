'use strict';
(() => {
  const button = document.querySelector('#copy-citation');
  if (!button || !navigator.clipboard) return;
  button.hidden = false;
  button.addEventListener('click', async () => {
    const status = document.querySelector('#citation-status');
    try {
      await navigator.clipboard.writeText(document.querySelector('#citation').textContent);
      status.textContent = 'BibTeX copied.';
    } catch {
      status.textContent = 'Copy was unavailable. Select the citation text to copy it.';
    }
  });
})();
(() => {
  const $ = (id) => document.getElementById(id);
  const compact = (n) => new Intl.NumberFormat('en-US', {notation: 'compact', maximumFractionDigits: 2}).format(n);
  const precision = (n) => Number(n.toPrecision(3)).toString();
  // Decode only requested atlases. The small cache bounds memory as Random is used.
  const atlases = new Map();
  function loadAtlas(source, signal) {
    const cached = atlases.get(source);
    if (cached && (cached.loaded || !cached.abortSignal?.aborted)) return cached;
    if (cached) atlases.delete(source);
    const promise = new Promise((resolve, reject) => {
      const image = new Image();
      let settled = false;
      const cleanup = () => {
        clearTimeout(timeout); image.onload = null; image.onerror = null;
        signal?.removeEventListener('abort', abort);
      };
      const fail = (error) => {
        if (settled) return;
        settled = true; cleanup(); image.src = ''; reject(error);
      };
      const abort = () => fail(new DOMException('Superseded by another example.', 'AbortError'));
      const timeout = setTimeout(() => fail(new Error('The saved video took too long to load.')), 20000);
      image.onload = () => {
        if (settled) return;
        settled = true; cleanup(); resolve(image);
      };
      image.onerror = () => fail(new Error('The saved video could not be loaded.'));
      signal?.addEventListener('abort', abort, {once: true});
      if (signal?.aborted) { abort(); return; }
      image.src = source;
    });
    promise.abortSignal = signal;
    promise.then(() => { promise.loaded = true; }, () => {});
    atlases.set(source, promise);
    promise.catch(() => { if (atlases.get(source) === promise) atlases.delete(source); });
    while (atlases.size > 16) atlases.delete(atlases.keys().next().value);
    return promise;
  }
  function paint(canvas, image, layout, frame, description, incorrect = false) {
    if (canvas.width !== layout.width || canvas.height !== layout.height) {
      canvas.width = layout.width; canvas.height = layout.height;
    }
    const x = (frame % layout.columns) * layout.width;
    const y = Math.floor(frame / layout.columns) * layout.height;
    canvas.getContext('2d').drawImage(image, x, y, layout.width, layout.height, 0, 0, canvas.width, canvas.height);
    canvas.setAttribute('aria-label', description);
    canvas.classList.toggle('incorrect-boundary', incorrect);
    canvas.dataset.frame = String(frame);
  }
  function boundaryText(boundary, actions) {
    return `After move ${boundary.index + 1} · ${actions[boundary.index]}`;
  }
  function makePlayer(prefix, render) {
    const slider = $(`${prefix}-frame`), play = $(`${prefix}-play`);
    let frame = Number(slider.value), frames = 81, fps = 24, playing = false, raf = null;
    let startTime = null, startFrame = frame, playbackSession = 0;
    function show(next) {
      frame = Math.max(0, Math.min(frames - 1, next));
      slider.value = String(frame);
      slider.style.setProperty('--played', `${frame / (frames - 1) * 100}%`);
      $(`${prefix}-time`).textContent = `Frame ${frame + 1} / ${frames} · ${(frame / fps).toFixed(2)} s`;
      slider.setAttribute('aria-valuetext', `Frame ${frame + 1} of ${frames}, ${(frame / fps).toFixed(2)} seconds`);
      const rendering = render(frame);
      $(`${prefix}-actions`).querySelectorAll('button').forEach((button) => button.setAttribute('aria-pressed', String(Number(button.dataset.frame) === frame)));
      return rendering;
    }
    function pause() {
      playing = false; playbackSession++; cancelAnimationFrame(raf); play.textContent = 'Play';
      play.setAttribute('aria-label', `Play ${prefix === 'rollout' ? 'rollout' : 'observation comparison'}`);
    }
    async function tick(time) {
      if (!playing) return;
      const session = playbackSession;
      if (startTime === null) startTime = time;
      const next = Math.min(frames - 1, startFrame + Math.floor((time - startTime) * fps / 1000));
      if (next !== frame) {
        const started = performance.now();
        await show(next);
        if (!playing || session !== playbackSession) return;
        // Buffer at a strip boundary instead of jumping over unloaded frames.
        if (performance.now() - started > 1000 / fps) { startTime = performance.now(); startFrame = frame; }
      }
      if (next === frames - 1) pause(); else raf = requestAnimationFrame(tick);
    }
    play.addEventListener('click', () => {
      if (playing) { pause(); return; }
      if (frame === frames - 1) show(0);
      document.dispatchEvent(new CustomEvent('pause-other-rollout', {detail: prefix}));
      playing = true; startTime = null; startFrame = frame; play.textContent = 'Pause';
      play.setAttribute('aria-label', `Pause ${prefix === 'rollout' ? 'rollout' : 'observation comparison'}`);
      raf = requestAnimationFrame(tick);
    });
    slider.addEventListener('input', () => { pause(); show(Number(slider.value)); });
    document.addEventListener('visibilitychange', () => { if (document.hidden) pause(); });
    document.addEventListener('pause-other-rollout', (event) => { if (event.detail !== prefix) pause(); });
    document.addEventListener('rollout-tab-change', pause);
    return {
      pause, currentFrame: () => frame, refresh: () => show(frame),
      configure(data, verdict) {
        pause(); frames = data.frames; fps = data.fps; slider.max = String(frames - 1);
        const container = $(`${prefix}-actions`); container.replaceChildren();
        const spacing = Math.min(...data.boundaries.slice(1).map((boundary, i) => boundary.frame - data.boundaries[i].frame));
        container.style.setProperty('--boundary-width', `${spacing / (frames - 1) * 100}%`);
        data.boundaries.forEach((boundary, index) => {
          const button = document.createElement('button'); button.type = 'button';
          const number = document.createElement('span'); number.className = 'boundary-number'; number.textContent = String(index + 1);
          const separator = document.createElement('span'); separator.className = 'boundary-separator'; separator.textContent = ' · '; separator.setAttribute('aria-hidden', 'true');
          const action = document.createElement('span'); action.className = 'boundary-action'; action.textContent = data.actions[index];
          button.append(number, separator, action);
          button.dataset.frame = String(boundary.frame);
          button.style.left = `${boundary.frame / (frames - 1) * 100}%`;
          const incorrect = verdict(boundary);
          button.classList.toggle('incorrect-move', incorrect);
          button.setAttribute('aria-label', `Move ${index + 1}: ${data.actions[index]}, ${incorrect ? 'incorrect boundary prediction' : 'correct boundary prediction'}`);
          button.addEventListener('click', () => { pause(); show(boundary.frame); });
          container.append(button);
        });
        $(`${prefix}-controls`).hidden = false;
        return show(frame);
      }
    };
  }
  const gallery = window.ROLLOUT_GALLERY;
  if (Array.isArray(gallery) && gallery.length) {
    const training = window.TRAINING_ROLLOUTS || [];
    let selected = -1, requested = -1, entry = null, bag = [];
    let mode = training.length ? 'training' : 'examples', trainingIndex = Math.max(0, training.length - 1), exampleIndex = 0;
    const rows = () => mode === 'training' ? training : gallery;
    const stepNumber = (value) => new Intl.NumberFormat('en-US').format(value);
    const modelText = (row) => `${row.model} model${row.step != null ? ` · step ${stepNumber(row.step)}` : ''} · ${compact(row.training_videos)} training videos · ${precision(row.training_pf_days)} PF-days`;
    const tabs = [$('training-tab'), $('rollout-tab')];
    function chooseMode(next) {
      if (next === mode) return;
      mode = next;
      document.dispatchEvent(new Event('rollout-tab-change'));
      tabs.forEach((tab, i) => {
        const active = i === (mode === 'training' ? 0 : 1);
        tab.setAttribute('aria-selected', String(active)); tab.tabIndex = active ? 0 : -1;
      });
      $('rollout-panel').setAttribute('aria-labelledby', mode === 'training' ? 'training-tab' : 'rollout-tab');
      $('training-checkpoint-controls').hidden = mode !== 'training';
      $('random-rollout').hidden = mode === 'training';
      $('rollout-caption').textContent = mode === 'training'
        ? 'The same episode and sampling noise at 15 training checkpoints. The frame slider covers the complete video; red outlines mark incorrect predictions at action boundaries. Ground truth is reconstructed through the frozen VAE.'
        : 'The slider covers all 81 frames. Action labels mark the nine evaluated frames; a red outline means at least one sticker is incorrect. Random selects one of 20 saved examples. The ground-truth video passes through the same frozen video autoencoder (VAE) as the predictions.';
      select(mode === 'training' ? trainingIndex : exampleIndex);
    }
    tabs.forEach((tab, index) => {
      tab.addEventListener('click', () => chooseMode(index ? 'examples' : 'training'));
      tab.addEventListener('keydown', (event) => {
        if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
        event.preventDefault();
        const next = event.key === 'Home' ? 0 : event.key === 'End' ? 1 : 1 - index;
        chooseMode(next ? 'examples' : 'training'); tabs[next].focus();
      });
    });
    if (training.length) {
      document.querySelector('.view-tabs').hidden = false;
      $('training-step').max = String(training.length - 1);
      training.forEach((row, index) => {
        const option = document.createElement('option'); option.value = String(index);
        option.textContent = `Step ${stepNumber(row.step)}`; $('training-checkpoint').append(option);
      });
      function chooseCheckpoint(index, load = true) {
        trainingIndex = index;
        $('training-step').value = String(index); $('training-checkpoint').value = String(index);
        $('training-step').setAttribute('aria-valuetext', `Step ${stepNumber(training[index].step)}, ${compact(training[index].training_videos)} training videos`);
        $('previous-checkpoint').disabled = index === 0;
        $('next-checkpoint').disabled = index === training.length - 1;
        if (load && mode === 'training') select(index);
      }
      $('training-step').addEventListener('input', () => chooseCheckpoint(Number($('training-step').value)));
      $('training-checkpoint').addEventListener('change', () => chooseCheckpoint(Number($('training-checkpoint').value)));
      $('previous-checkpoint').addEventListener('click', () => chooseCheckpoint(Math.max(0, trainingIndex - 1)));
      $('next-checkpoint').addEventListener('click', () => chooseCheckpoint(Math.min(training.length - 1, trainingIndex + 1)));
      chooseCheckpoint(trainingIndex, false);
    }
    let selectionVersion = 0, frameVersion = 0, selectionController = null, frameController = null;
    const status = $('rollout-status');
    function stripSources(row, frame) {
      const strip = Math.floor(frame / row.columns);
      return row.gt_strips && row.pred_strips
        ? {gt: row.gt_strips[strip], pred: row.pred_strips[strip], frame: frame % row.columns}
        : {gt: row.gt, pred: row.pred, frame};
    }
    function controlsDisabled(disabled) {
      $('rollout-controls').querySelectorAll('button, input').forEach((control) => { control.disabled = disabled; });
    }
    function frameBusy(busy) {
      $('rollout-pair').classList.toggle('loading-frame', busy);
      $('rollout-pair').setAttribute('aria-busy', String(busy));
    }
    const player = makePlayer('rollout', async (frame) => {
      if (!entry) return;
      const row = entry, version = ++frameVersion;
      const source = stripSources(row, frame);
      const uncached = !atlases.has(source.gt) || !atlases.has(source.pred);
      if (uncached) {
        frameBusy(true);
        $('state-score').textContent = 'Loading frame…';
        $('state-score').classList.remove('correct', 'incorrect');
        $('prediction').classList.remove('incorrect-boundary');
      }
      try {
        const images = await Promise.all([loadAtlas(source.gt, frameController.signal), loadAtlas(source.pred, frameController.signal)]);
        if (version !== frameVersion || row !== entry) return;
        const index = row.boundaries.findIndex((b) => b.frame === frame);
        const boundary = index < 0 ? null : {...row.boundaries[index], index};
        paint($('ground-truth'), images[0], row, source.frame, `Ground-truth VAE reconstruction, frame ${frame + 1}.`);
        paint($('prediction'), images[1], row, source.frame, `${row.model} prediction, frame ${frame + 1}${boundary ? `, ${boundary.correct_stickers}/12 stickers correct` : ', unscored frame'}.`, boundary !== null && !boundary.exact_frame);
        // Keep the public frame index independent of the strip's local coordinates.
        $('ground-truth').dataset.frame = String(frame); $('prediction').dataset.frame = String(frame);
        const score = $('state-score');
        score.textContent = boundary ? `${boundaryText(boundary, row.actions)} · ${boundary.correct_stickers}/12 correct` : frame === 0 ? 'Initial image · given' : 'Between scored boundaries';
        score.classList.toggle('correct', boundary !== null && boundary.exact_frame);
        score.classList.toggle('incorrect', boundary !== null && !boundary.exact_frame);
        frameBusy(false);
        // Load only the following strip ahead, giving playback time to buffer.
        const nextStrip = Math.floor(frame / row.columns) + 1;
        if (row.gt_strips && nextStrip < row.gt_strips.length && frame % row.columns >= 5) {
          [row.gt_strips[nextStrip], row.pred_strips[nextStrip]].forEach((url) => loadAtlas(url, frameController.signal).catch(() => {}));
        }
      } catch (error) {
        if (version !== frameVersion || row !== entry || error.name === 'AbortError') return;
        frameBusy(false); player.pause();
        $('state-score').textContent = 'Frame unavailable. Move the slider to retry.';
      }
    });
    async function select(index) {
      const version = ++selectionVersion;
      requested = index;
      player.pause(); controlsDisabled(true);
      selectionController?.abort(); frameController?.abort(); ++frameVersion;
      // Remove canceled promises before another example can reuse a shared GT strip.
      await Promise.resolve();
      if (version !== selectionVersion) return;
      selectionController = new AbortController();
      const controller = selectionController, next = rows()[index];
      $('random-rollout').disabled = false;
      $('random-rollout').textContent = 'Random';
      status.textContent = mode === 'training'
        ? `Loading step ${stepNumber(next.step)} · checkpoint ${index + 1} / ${training.length}…`
        : `Loading example ${index + 1} / ${gallery.length} · ${next.model} · ${compact(next.training_videos)} videos…`;
      $('retry-rollout').hidden = true;
      status.classList.remove('load-error');
      $('rollout-panel').setAttribute('aria-busy', 'true');
      frameBusy(true);
      try {
        const source = stripSources(next, player.currentFrame());
        await Promise.all([loadAtlas(source.gt, controller.signal), loadAtlas(source.pred, controller.signal)]);
        if (version !== selectionVersion) return;
        entry = next; frameController = new AbortController();
        await player.configure(entry, (b) => !b.exact_frame);
        if (version !== selectionVersion) return;
        selected = index;
        if (mode === 'examples') exampleIndex = index;
        $('rollout-model').textContent = modelText(entry);
        $('rollout-panel').dataset.rolloutId = entry.id;
        status.textContent = mode === 'training'
          ? `Checkpoint ${index + 1} / ${training.length} · same initial image and actions`
          : `Example ${index + 1} / ${gallery.length}`;
      } catch (error) {
        if (version !== selectionVersion) return;
        // Preserve a usable previous example when a network request fails.
        frameController = new AbortController();
        status.textContent = 'Could not load the selected video. Retry or choose another checkpoint or example.';
        $('retry-rollout').hidden = false;
        status.classList.add('load-error');
      } finally {
        if (version === selectionVersion) {
          controlsDisabled(selected < 0);
          frameBusy(false);
          $('rollout-panel').setAttribute('aria-busy', 'false');
        }
      }
    }
    $('retry-rollout').addEventListener('click', () => select(requested));
    $('random-rollout').addEventListener('click', () => {
      if (!bag.length) {
        bag = gallery.map((_, index) => index).filter((index) => index !== requested);
        for (let i = bag.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1)); [bag[i], bag[j]] = [bag[j], bag[i]];
        }
      }
      select(bag.pop());
    });
    if (!training.length) {
      $('training-checkpoint-controls').hidden = true;
      $('random-rollout').hidden = false;
      $('rollout-panel').setAttribute('aria-labelledby', 'teaser-title');
    }
    select(mode === 'training' ? trainingIndex : exampleIndex);
  } else {
    $('rollout-status').textContent = 'The saved examples could not be loaded. Reload the page to retry.';
  }
  const comparison = window.OBSERVATION_ROLLOUT;
  if (comparison) {
    let images = null;
    const player = makePlayer('observation', (frame) => {
      if (!images) return;
      const index = comparison.boundaries.findIndex((b) => b.frame === frame);
      const boundary = index < 0 ? null : {...comparison.boundaries[index], index};
      paint($('observation-gt'), images[0], comparison.gt, frame, `Three-face simulator ground truth, frame ${frame + 1}.`);
      ['three', 'six'].forEach((key, position) => {
        const result = boundary?.[key];
        paint($(`observation-${key}`), images[position + 1], comparison[key], frame, `${key === 'three' ? 'Three' : 'Six'}-face prediction, frame ${frame + 1}${result ? `, ${result.correct_stickers}/${result.total_stickers} stickers correct` : ', unscored frame'}.`, !!result && !result.exact_frame);
        const score = $(`observation-${key}-score`);
        score.textContent = result ? `${result.correct_stickers}/${result.total_stickers} stickers correct` : frame === 0 ? 'Initial image · given' : 'Unscored frame';
        score.classList.toggle('correct', !!result && result.exact_frame);
        score.classList.toggle('incorrect', !!result && !result.exact_frame);
      });
      $('observation-action').textContent = boundary ? boundaryText(boundary, comparison.actions) : 'Between scored boundaries';
    });
    async function loadComparison() {
      try {
        images = await Promise.all(['gt','three','six'].map((key) => loadAtlas(comparison[key].image)));
        player.configure(comparison, (b) => !b.three.exact_frame || !b.six.exact_frame);
        $('observation-loading').hidden = true;
      } catch {
        $('observation-loading').textContent = 'The comparison could not be loaded. Reload the page to try again.';
      }
    }
    if ('IntersectionObserver' in window) {
      const observer = new IntersectionObserver((entries) => {
        if (entries.some((entry) => entry.isIntersecting)) { observer.disconnect(); loadComparison(); }
      }, {rootMargin: '400px'});
      observer.observe($('observation-player'));
    } else loadComparison();
  }
})();

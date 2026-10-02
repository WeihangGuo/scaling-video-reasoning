'use strict';
(() => {
  const data = window.PAPER_DATA;
  if (!data || !data.checkpoints.length) return;
  const $ = (selector) => document.querySelector(selector);
  const models = ['XS', 'S', 'M', 'B', 'L', 'XL', 'XXL'];
  const labels = ['20M', '30M', '70M', '120M', '270M', '550M', '1B'];
  const colors = ['#64748b', '#457b94', '#1767a2', '#218071', '#a37a1b', '#c15432', '#814c9e'];
  const methodColors = ['#718096', '#1767a2', '#a46d31'];
  const label = (model) => labels[models.indexOf(model)];
  const color = (model) => colors[models.indexOf(model)];
  const percent = (value) => `${(value * 100).toFixed(1)}%`;
  const compact = (value) => new Intl.NumberFormat('en-US', {notation: 'compact', maximumFractionDigits: 2}).format(value);
  const number = (value) => new Intl.NumberFormat('en-US').format(value);
  const precision = (value) => Number(value.toPrecision(3)).toString();
  const ns = 'http://www.w3.org/2000/svg';
  function svgElement(tag, attrs = {}, text = null) {
    const element = document.createElementNS(ns, tag);
    Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, String(value)));
    if (text !== null) element.textContent = text;
    return element;
  }
  function makeChart(container, title, options = {}) {
    const width = Math.max(280, Math.floor(container.clientWidth));
    const height = options.height ?? (width < 550 ? 330 : 385);
    const margin = {left: 56, right: 20, top: 26, bottom: 59};
    const svg = svgElement('svg', {viewBox: `0 0 ${width} ${height}`, width, height, role: 'img', 'aria-label': title});
    svg.append(svgElement('title', {}, title));
    container.replaceChildren(svg);
    const box = {svg, width, height, left: margin.left, right: width - margin.right, top: margin.top, bottom: height - margin.bottom};
    box.y = (value) => box.bottom - value * (box.bottom - box.top);
    [0, .25, .5, .75, 1].forEach((value) => {
      svg.append(svgElement('line', {x1: box.left, x2: box.right, y1: box.y(value), y2: box.y(value), class: 'chart-grid'}));
      svg.append(svgElement('text', {x: box.left - 10, y: box.y(value) + 5, 'text-anchor': 'end', class: 'chart-tick'}, `${value * 100}%`));
    });
    return box;
  }
  function xAxis(box, ticks, scale, title) {
    ticks.forEach((tick) => {
      const x = scale(tick);
      if (x < box.left - 1 || x > box.right + 1) return;
      box.svg.append(svgElement('text', {x, y: box.bottom + 24, 'text-anchor': 'middle', class: 'chart-tick'}, String(tick)));
    });
    box.svg.append(svgElement('text', {x: (box.left + box.right) / 2, y: box.height - 12, 'text-anchor': 'middle', class: 'chart-axis-title'}, title));
  }
  function stat(title, value, detail = '') {
    const item = document.createElement('div');
    const term = document.createElement('span'); term.textContent = title;
    const amount = document.createElement('strong'); amount.textContent = value;
    item.append(term, amount);
    if (detail) {const note = document.createElement('small'); note.textContent = detail; item.append(note);}
    return item;
  }

  // Loss versus capability: dots always represent measured paired checkpoints.
  let lossPoints = [], selectedLoss = null, comparison = false;
  const lossLegend = $('#loss-legend');
  models.forEach((model) => {
    const item = document.createElement('span');
    item.style.setProperty('--model-color', color(model));
    item.textContent = label(model); lossLegend.append(item);
  });
  function updateLossReadout(point) {
    selectedLoss = point;
    $('#loss-checkpoint').value = String(lossPoints.findIndex((row) => row.key === point.key));
    $('#loss-checkpoint').setAttribute('aria-valuetext', `${label(point.model)} model, ${number(point.training_videos)} training videos, MSE ${point.mse.toFixed(4)}, frame accuracy ${percent(point.frame_accuracy)}`);
    $('#loss-readout').replaceChildren(
      stat('Model size', label(point.model), `${number(point.parameters)} parameters`),
      stat('Training videos', compact(point.training_videos), `${precision(point.training_pf_days)} PF-days`),
      stat('Validation MSE', point.mse.toFixed(4)),
      stat('Frame accuracy', percent(point.frame_accuracy))
    );
    $('#loss-chart').querySelectorAll('circle').forEach((circle) => {
      const selected = circle.dataset.key === point.key;
      const paired = comparison && ['ar:M:1500000','ar:XXL:1500000'].includes(circle.dataset.key);
      circle.setAttribute('r', selected || paired ? '7' : '4');
      circle.setAttribute('stroke', selected || paired ? '#172539' : 'white');
      circle.setAttribute('stroke-width', selected ? '2.3' : paired ? '1.5' : '.6');
    });
  }
  function renderLoss() {
    const family = $('#loss-family').value;
    const chosen = $('#loss-model').value;
    const points = data.checkpoints.filter((row) => row.family === family);
    lossPoints = points.filter((row) => chosen === 'all' || row.model === chosen).sort((a,b) => a.mse - b.mse || models.indexOf(a.model) - models.indexOf(b.model) || a.training_videos - b.training_videos);
    if (comparison) lossPoints = lossPoints.filter((row) => ['ar:M:1500000','ar:XXL:1500000'].includes(row.key));
    const oldKey = selectedLoss?.key;
    const fallback = lossPoints.find((row) => row.model === 'M' && row.training_videos === 1500000) || lossPoints[0];
    selectedLoss = lossPoints.find((row) => row.key === oldKey) || fallback;
    $('#loss-checkpoint').max = String(lossPoints.length - 1);
    const box = makeChart($('#loss-chart'), 'Validation flow MSE versus frame accuracy. Each colored point is an evaluated checkpoint. Use the checkpoint slider for exact values.');
    const minimum = Math.min(...points.map((row) => row.mse)) * .87;
    const maximum = Math.max(...points.map((row) => row.mse)) * 1.12;
    const x = (value) => box.left + (Math.log(value) - Math.log(minimum)) / (Math.log(maximum) - Math.log(minimum)) * (box.right - box.left);
    const ticks = [];
    for (let power = Math.floor(Math.log10(minimum)); power <= Math.ceil(Math.log10(maximum)); power += 1) {
      const tick = 10 ** power;
      if (tick >= minimum && tick <= maximum) ticks.push(Number(tick.toPrecision(1)));
    }
    xAxis(box, ticks, x, 'Validation flow MSE · logarithmic scale');
    box.svg.append(svgElement('text', {x: box.left, y: 16, class: 'chart-axis-title'}, 'Frame accuracy'));
    points.forEach((point) => {
      const muted = (chosen !== 'all' && point.model !== chosen) || (comparison && !['ar:M:1500000','ar:XXL:1500000'].includes(point.key));
      const circle = svgElement('circle', {cx: x(point.mse), cy: box.y(point.frame_accuracy), r: 4, fill: color(point.model), opacity: muted ? '.12' : '.8', stroke: 'white', 'stroke-width': '.6', 'data-key': point.key});
      circle.append(svgElement('title', {}, `${label(point.model)} · ${compact(point.training_videos)} videos · MSE ${point.mse.toFixed(4)} · ${percent(point.frame_accuracy)} frame accuracy`));
      if (!muted) {
        circle.style.cursor = 'pointer';
        circle.addEventListener('pointerenter', () => updateLossReadout(point));
        circle.addEventListener('click', () => updateLossReadout(point));
      }
      box.svg.append(circle);
    });
    updateLossReadout(selectedLoss);
  }
  ['#loss-family','#loss-model'].forEach((id) => $(id).addEventListener('change', () => {
    comparison = false; $('#loss-example').setAttribute('aria-pressed','false'); $('#loss-comparison').hidden = true; renderLoss();
  }));
  $('#loss-checkpoint').addEventListener('input', () => updateLossReadout(lossPoints[Number($('#loss-checkpoint').value)]));
  $('#loss-example').addEventListener('click', () => {
    if (comparison) {comparison = false; $('#loss-example').setAttribute('aria-pressed','false'); $('#loss-comparison').hidden = true; renderLoss(); return;}
    $('#loss-family').value = 'ar'; $('#loss-model').value = 'all'; comparison = true;
    $('#loss-example').setAttribute('aria-pressed','true');
    const small = data.checkpoints.find((row) => row.key === 'ar:M:1500000');
    const large = data.checkpoints.find((row) => row.key === 'ar:XXL:1500000');
    const note = $('#loss-comparison');
    note.textContent = `At 1.5M training videos, the 70M model reaches ${percent(small.frame_accuracy)} frame accuracy (MSE ${small.mse.toFixed(4)}). The 1B model reaches ${percent(large.frame_accuracy)} (MSE ${large.mse.toFixed(4)}). The larger model has lower loss but less accurate states at this data budget.`;
    note.hidden = false; selectedLoss = small; renderLoss();
  });
  $('#loss-explorer').hidden = false;
  renderLoss();

  // Compute budgets select checkpoints, never interpolated performance.
  let budgets = [], currentBudget = .1;
  models.forEach((model) => {
    const item = document.createElement('span');
    item.style.setProperty('--model-color', color(model));
    item.textContent = label(model); $('#compute-legend').append(item);
  });
  function renderComputeCurves(points, choices, metric) {
    const box = makeChart($('#compute-chart'), 'Measured accuracy versus training compute, with one curve per model size. The vertical line marks the selected budget; large dots identify the best measured checkpoints within it.');
    const minimum = Math.min(...points.map((row) => row.training_pf_days)) * .8;
    const maximum = Math.max(...points.map((row) => row.training_pf_days)) * 1.3;
    const x = (value) => box.left + Math.log(value / minimum) / Math.log(maximum / minimum) * (box.right - box.left);
    const ticks = [];
    for (let power = Math.ceil(Math.log10(minimum)); power <= Math.floor(Math.log10(maximum)); power++) ticks.push(Number((10 ** power).toPrecision(1)));
    xAxis(box, ticks, x, 'Training compute (PF-days) · logarithmic scale');
    box.svg.append(svgElement('text', {x: box.left, y: 16, class: 'chart-axis-title'}, metric === 'frame_accuracy' ? 'Average frame accuracy' : 'Full-trajectory accuracy'));
    const budgetX = x(currentBudget);
    box.svg.append(svgElement('rect', {x: box.left, y: box.top, width: budgetX - box.left, height: box.bottom - box.top, fill: '#173e7a', opacity: '.035'}));
    models.forEach((model) => {
      const series = points.filter((row) => row.model === model).sort((a,b) => a.training_pf_days - b.training_pf_days);
      const path = series.map((point, i) => `${i ? 'L' : 'M'}${x(point.training_pf_days)},${box.y(point[metric])}`).join(' ');
      box.svg.append(svgElement('path', {d: path, fill: 'none', stroke: color(model), 'stroke-width': 2, 'stroke-linejoin': 'round'}));
      series.forEach((point) => {
        const circle = svgElement('circle', {cx: x(point.training_pf_days), cy: box.y(point[metric]), r: 2.8, fill: color(model), opacity: point.training_pf_days <= currentBudget ? .95 : .4});
        circle.append(svgElement('title', {}, `${label(model)} · ${compact(point.training_videos)} videos · ${precision(point.training_pf_days)} PF-days · ${percent(point[metric])}`));
        box.svg.append(circle);
      });
    });
    box.svg.append(svgElement('line', {x1: budgetX, x2: budgetX, y1: box.top, y2: box.bottom, stroke: '#172539', 'stroke-width': 1.5, 'stroke-dasharray': '5 4'}));
    choices.filter(Boolean).forEach((point) => {
      const circle = svgElement('circle', {cx: x(point.training_pf_days), cy: box.y(point[metric]), r: 6.5, fill: color(point.model), stroke: 'white', 'stroke-width': 2});
      circle.append(svgElement('title', {}, `Best within budget: ${label(point.model)}, ${percent(point[metric])} at ${precision(point.training_pf_days)} PF-days`));
      box.svg.append(circle);
    });
  }
  function configureBudget() {
    const family = $('#compute-family').value;
    const points = data.checkpoints.filter((row) => row.family === family);
    const maximum = Math.max(...points.map((row) => row.training_pf_days));
    budgets = [...new Set([...points.map((row) => row.training_pf_days), .01,.1,1].filter((value) => value <= maximum))].sort((a,b) => a-b);
    $('#compute-budget').max = String(budgets.length - 1);
    setBudget(Math.min(currentBudget, maximum));
  }
  function setBudget(target) {
    let index = budgets.findIndex((budget) => budget >= target - 1e-12);
    if (index < 0) index = budgets.length - 1;
    $('#compute-budget').value = String(index);
    renderBudget();
  }
  function renderBudget() {
    currentBudget = budgets[Number($('#compute-budget').value)];
    const family = $('#compute-family').value;
    const metric = $('#compute-metric').value;
    $('#budget-value').textContent = `${precision(currentBudget)} PF-days`;
    $('#compute-budget').setAttribute('aria-valuetext', `${precision(currentBudget)} PF-days`);
    const choices = models.map((model) => data.checkpoints.filter((row) => row.family === family && row.model === model && row.training_pf_days <= currentBudget + 1e-12).sort((a,b) => b[metric]-a[metric] || a.training_pf_days-b.training_pf_days)[0]);
    const table = $('#compute-table'); table.replaceChildren();
    choices.forEach((row, index) => {
      const tr = document.createElement('tr');
      [labels[index], row ? number(row.parameters) : '—', row ? number(row.training_videos) : '—', row ? precision(row.training_pf_days) : '—', row ? percent(row[metric]) : 'Not evaluated'].forEach((text, i) => {
        const cell = document.createElement(i ? 'td' : 'th'); if (!i) cell.scope = 'row'; cell.textContent = text; tr.append(cell);
      }); table.append(tr);
    });
    renderComputeCurves(data.checkpoints.filter((row) => row.family === family), choices, metric);
    const best = choices.filter(Boolean).sort((a,b) => b[metric]-a[metric] || a.training_pf_days-b.training_pf_days)[0];
    $('#budget-takeaway').textContent = best[metric] === 0
      ? `No evaluated checkpoint within this budget achieves a ${metric === 'frame_accuracy' ? 'fully correct post-action frame' : 'fully correct trajectory'}.`
      : `${label(best.model)} reaches the highest observed ${metric === 'frame_accuracy' ? 'average frame' : 'full-trajectory'} accuracy within this budget: ${percent(best[metric])}.`;
    document.querySelectorAll('[data-budget]').forEach((button) => {
      const target = button.dataset.budget === 'max' ? budgets[budgets.length-1] : Number(button.dataset.budget);
      button.setAttribute('aria-pressed', String(Math.abs(currentBudget-target) < 1e-10));
    });
  }
  $('#compute-family').addEventListener('change', configureBudget);
  $('#compute-metric').addEventListener('change', renderBudget);
  $('#compute-budget').addEventListener('input', renderBudget);
  document.querySelectorAll('[data-budget]').forEach((button) => button.addEventListener('click', () => setBudget(button.dataset.budget === 'max' ? budgets[budgets.length-1] : Number(button.dataset.budget))));
  $('#compute-explorer').hidden = false;
  configureBudget();

  // Observation control: the paper's four measured curves and confidence bands.
  const observationData = window.OBSERVATION_CURVES;
  let observationAction = 7;
  const observationColor = (series) => series.observation === 'three' ? '#b64847' : '#287d8e';
  function renderObservationCurves() {
    if (!observationData) return;
    const container = $('#observation-curve-chart');
    const box = makeChart(container, 'Frame accuracy across nine actions for three-face and six-face observation, with AR-k4 and bidirectional generation. Use the left and right arrow keys to inspect an action.', {height: 250});
    const x = (action) => box.left + (action - 1) / 8 * (box.right - box.left);
    xAxis(box, [1,2,3,4,5,6,7,8,9], x, 'Action number');
    box.svg.append(svgElement('text', {x: box.left, y: 16, class: 'chart-axis-title'}, 'Frame accuracy'));
    observationData.series.forEach((series) => {
      const upper = series.points.map((point, i) => `${i ? 'L' : 'M'}${x(point.action)},${box.y(point.upper)}`);
      const lower = [...series.points].reverse().map((point) => `L${x(point.action)},${box.y(point.lower)}`);
      box.svg.append(svgElement('path', {d: [...upper, ...lower, 'Z'].join(' '), fill: observationColor(series), opacity: '.09', class: 'observation-confidence'}));
    });
    observationData.series.forEach((series) => {
      const path = series.points.map((point, i) => `${i ? 'L' : 'M'}${x(point.action)},${box.y(point.accuracy)}`).join(' ');
      box.svg.append(svgElement('path', {d: path, fill: 'none', stroke: observationColor(series), 'stroke-width': 2.3, 'stroke-dasharray': series.family === 'ar' ? 'none' : '6 4', 'stroke-linejoin': 'round', class: 'observation-curve', 'data-series': series.id}));
      series.points.forEach((point) => {
        const dot = svgElement('circle', {cx: x(point.action), cy: box.y(point.accuracy), r: 3.2, fill: observationColor(series), stroke: 'white', 'stroke-width': .8, 'data-action': point.action});
        dot.append(svgElement('title', {}, `${series.label}, action ${point.action}: ${percent(point.accuracy)} (95% CI ${percent(point.lower)}–${percent(point.upper)})`));
        box.svg.append(dot);
      });
    });
    const cursor = svgElement('line', {y1: box.top, y2: box.bottom, stroke: '#8fa1b7', 'stroke-dasharray': '3 4', 'pointer-events': 'none'});
    box.svg.append(cursor);
    function inspect(action) {
      observationAction = action;
      cursor.setAttribute('x1', x(action)); cursor.setAttribute('x2', x(action));
      box.svg.querySelectorAll('circle[data-action]').forEach((dot) => dot.setAttribute('r', Number(dot.dataset.action) === action ? '5' : '3.2'));
      const heading = document.createElement('strong'); heading.textContent = `Action ${action}`;
      const values = observationData.series.map((series) => {
        const value = document.createElement('span');
        value.textContent = `${series.label}: ${percent(series.points.find((point) => point.action === action).accuracy)}`;
        value.style.setProperty('--curve-color', observationColor(series)); return value;
      });
      $('#observation-curve-readout').replaceChildren(heading, ...values);
    }
    function inspectPointer(event) {
      const bounds = box.svg.getBoundingClientRect();
      const chartX = (event.clientX - bounds.left) * box.width / bounds.width;
      const action = Math.max(1, Math.min(9, Math.round(1 + (chartX - box.left) / (box.right - box.left) * 8)));
      if (action !== observationAction) inspect(action);
    }
    box.svg.setAttribute('tabindex', '0');
    box.svg.setAttribute('aria-describedby', 'observation-curve-readout');
    box.svg.addEventListener('pointermove', (event) => { if (event.pointerType !== 'touch') inspectPointer(event); });
    box.svg.addEventListener('click', inspectPointer);
    box.svg.addEventListener('keydown', (event) => {
      if (!['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) return;
      event.preventDefault();
      inspect(event.key === 'Home' ? 1 : event.key === 'End' ? 9 : Math.max(1, Math.min(9, observationAction + (event.key === 'ArrowRight' ? 1 : -1))));
    });
    inspect(observationAction);
  }
  if (observationData) {
    observationData.series.forEach((series) => {
      const item = document.createElement('span'); item.textContent = series.label;
      item.style.setProperty('--curve-color', observationColor(series));
      item.classList.toggle('dashed', series.family !== 'ar'); $('#observation-legend').append(item);
    });
    $('#observation-results').hidden = false;
    renderObservationCurves();
  }

  // Symbolic guidance: keep all methods visible, emphasize the selected one.
  function renderGuidance() {
    const selected = Number(document.querySelector('[name="guidance-method"]:checked').value);
    const action = Number($('#guidance-action').value);
    const method = data.guidance[selected];
    $('#guidance-average').textContent = percent(method.frame_accuracy);
    $('#guidance-trajectory').textContent = percent(method.full_trajectory_accuracy);
    $('#guidance-action-number').textContent = String(action);
    $('#guidance-explanation').textContent = [
      'The video-only baseline learns with the video flow objective and uses no symbolic feedback.',
      'All 24 stickers are supervised. Only the 12 visible-sticker probability distributions are fed back.',
      'All 24 stickers are supervised, and both visible and hidden sticker distributions are fed back.'
    ][selected];
    const box = makeChart($('#guidance-chart'), 'Frame accuracy after each action for video-only, Front12 and Full24 models. Use the action slider for exact values.', {height: 250});
    const x = (value) => box.left + (value-1)/8*(box.right-box.left);
    xAxis(box,[1,2,3,4,5,6,7,8,9],x,'Action number');
    box.svg.append(svgElement('text', {x: box.left, y: 16, class: 'chart-axis-title'}, 'Frame accuracy'));
    box.svg.append(svgElement('line', {x1:x(action),x2:x(action),y1:box.top,y2:box.bottom,stroke:'#8fa1b7','stroke-dasharray':'4 4'}));
    data.guidance.forEach((row, index) => {
      const path = row.actions.map((item,i) => `${i?'L':'M'}${x(item.action)},${box.y(item.accuracy)}`).join(' ');
      box.svg.append(svgElement('path',{d:path,fill:'none',stroke:methodColors[index],'stroke-width':selected === index ? 3.5 : 1.8,opacity:selected === index ? 1 : .4}));
      row.actions.forEach((item) => {
        const dot = svgElement('circle',{cx:x(item.action),cy:box.y(item.accuracy),r:item.action===action?6:3.5,fill:methodColors[index],stroke:'white','stroke-width':1,opacity:selected === index ? 1 : .55});
        dot.append(svgElement('title',{},`${row.method}, action ${item.action}: ${percent(item.accuracy)}`));
        dot.style.cursor='pointer';dot.addEventListener('click',()=>{$('#guidance-action').value=String(item.action);document.querySelector(`[name="guidance-method"][value="${index}"]`).checked=true;renderGuidance();});
        box.svg.append(dot);
      });
    });
    $('#guidance-readout').replaceChildren(...data.guidance.map((row,i) => {
      const item=stat(row.method,percent(row.actions.find((r)=>r.action===action).accuracy),`After action ${action}`);
      item.style.borderTop = `3px solid ${methodColors[i]}`; item.classList.toggle('selected-result',i===selected);return item;
    }));
    $('#guidance-action').setAttribute('aria-valuetext',`Action ${action}: ${percent(method.actions.find((r)=>r.action===action).accuracy)} frame accuracy for ${method.method}`);
  }
  document.querySelectorAll('[name="guidance-method"]').forEach((input)=>input.addEventListener('change',renderGuidance));
  $('#guidance-action').addEventListener('input',renderGuidance);
  $('#guidance-explorer').hidden=false;
  renderGuidance();
  let lastWidth=window.innerWidth;
  window.addEventListener('resize',()=>{if(window.innerWidth!==lastWidth){lastWidth=window.innerWidth;renderLoss();renderBudget();renderObservationCurves();renderGuidance();}});
})();

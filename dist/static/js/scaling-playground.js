'use strict';
(() => {
  const data = window.SCALING_FITS, observations = window.PAPER_DATA?.checkpoints;
  if (!data || !observations) return;
  const $ = (id) => document.getElementById(id);
  const ns = 'http://www.w3.org/2000/svg';
  const colors = {middle: '#b0612e', final: '#245783'};
  const accuracy = (fit, compute) => 1 - Math.exp(fit.log_A - fit.alpha * Math.log(compute));
  const crossing = (fit, target) => Math.exp((fit.log_A - Math.log1p(-target)) / fit.alpha);
  const format = (n) => n >= 1e4 || n < .001 ? n.toExponential(2) : Number(n.toPrecision(3)).toString();
  const percent = (n) => `${(n * 100).toFixed(1)}%`;
  const svg = (tag, attrs = {}, text) => {
    const node = document.createElementNS(ns, tag);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
    if (text !== undefined) node.textContent = text;
    return node;
  };
  let panel, selectedFits = [], lower, upper, compute = null;
  function configure() {
    panel = data.panels.find((p) => p.family === $('fit-family').value && p.metric === $('fit-metric').value);
    const window = $('fit-window').value;
    selectedFits = panel.fits.filter((fit) => window === 'both' || fit.segment === window);
    const target = Number($('fit-target').value) / 100;
    lower = Math.log10(panel.observed_min_pf_days);
    upper = Math.log10(Math.max(panel.observed_max_pf_days * 1.3, ...selectedFits.map((fit) => crossing(fit, target) * 1.5)));
    compute = Math.max(10 ** lower, Math.min(10 ** upper, compute ?? panel.observed_max_pf_days));
    const empty = !selectedFits.length;
    ['fit-compute', 'fit-target', 'fit-window'].forEach((id) => { $(id).disabled = empty; });
    $('fit-results').closest('.table-scroll').hidden = empty;
    document.querySelector('.fit-legend').hidden = empty;
    $('fit-compute').value = String(Math.round((Math.log10(compute) - lower) / (upper - lower) * 1000));
    render();
  }
  function drawChart(target) {
    const container = $('fit-chart');
    const width = Math.max(260, Math.floor(container.clientWidth)), height = width < 550 ? 320 : 365;
    const left = 53, right = width - 18, top = 28, bottom = height - 58;
    const x = (c) => left + (Math.log10(c) - lower) / (upper - lower) * (right - left);
    const y = (a) => bottom - a * (bottom - top);
    const chart = svg('svg', {viewBox: `0 0 ${width} ${height}`, width, height, role: 'img', 'aria-label': 'Measured accuracy and power-law fits versus training compute. Shading marks evaluated compute. Use the compute slider and fit controls to inspect values.'});
    chart.append(svg('title', {}, 'Accuracy and fitted scaling curves'));
    const defs = svg('defs'), clip = svg('clipPath', {id: 'fit-plot-clip'});
    clip.append(svg('rect', {x: left, y: top, width: right - left, height: bottom - top})); defs.append(clip); chart.append(defs);
    const plot = svg('g', {'clip-path': 'url(#fit-plot-clip)'});
    plot.append(svg('rect', {x: left, y: top, width: x(panel.observed_max_pf_days) - left, height: bottom - top, fill: '#efefef', 'data-role': 'observed-range'}));
    [0,.25,.5,.75,1].forEach((a) => {
      plot.append(svg('line', {x1: left, x2: right, y1: y(a), y2: y(a), class: 'chart-grid'}));
      chart.append(svg('text', {x: left - 9, y: y(a) + 4, 'text-anchor': 'end', class: 'chart-tick'}, `${a * 100}%`));
    });
    const tickStep = Math.max(1, Math.ceil((upper - lower) / (width < 550 ? 4 : 7)));
    for (let power = Math.ceil(lower / tickStep) * tickStep; power <= upper; power += tickStep) {
      const pointX = x(10 ** power);
      plot.append(svg('line', {x1: pointX, x2: pointX, y1: top, y2: bottom, class: 'chart-grid'}));
      const label = svg('text', {x: pointX, y: bottom + 24, 'text-anchor': 'middle', class: 'chart-tick'}, '10');
      label.append(svg('tspan', {'baseline-shift': 'super', 'font-size': '10'}, power)); chart.append(label);
    }
    chart.append(svg('text', {x: left, y: 17, class: 'chart-axis-title'}, panel.metric === 'frame_accuracy' ? 'Frame accuracy' : 'Full-trajectory accuracy'));
    chart.append(svg('text', {x: (left + right) / 2, y: height - 10, 'text-anchor': 'middle', class: 'chart-axis-title'}, 'Training compute (PF-days, log scale)'));
    observations.filter((p) => p.family === panel.family).forEach((point) => {
      const dot = svg('circle', {cx: x(point.training_pf_days), cy: y(point[panel.metric]), r: 2.4, fill: '#aaa', opacity: .6});
      dot.append(svg('title', {}, `${format(point.training_pf_days)} PF-days; ${percent(point[panel.metric])}`)); plot.append(dot);
    });
    plot.append(svg('path', {d: panel.frontier.map((point, i) => i ? `H${x(point.training_pf_days)} V${y(point.accuracy)}` : `M${x(point.training_pf_days)},${y(point.accuracy)}`).join(' ') + ` H${x(panel.observed_max_pf_days)}`, fill: 'none', stroke: '#777', 'stroke-width': 1.1}));
    panel.frontier.forEach((point) => {
      const fit = selectedFits.find((f) => f.segment === point.segment);
      plot.append(svg('circle', {cx: x(point.training_pf_days), cy: y(point.accuracy), r: fit ? 3.8 : 2.5, fill: fit ? colors[fit.segment] : '#777', 'data-window': point.segment ?? 'none'}));
    });
    function curve(fit, start, end, dashed) {
      if (end <= start) return;
      const logStart = Math.log10(start), logEnd = Math.log10(end);
      const path = Array.from({length: 241}, (_, i) => {
        const c = 10 ** (logStart + (logEnd - logStart) * i / 240);
        return `${i ? 'L' : 'M'}${x(c)},${y(accuracy(fit, c))}`;
      }).join(' ');
      plot.append(svg('path', {d: path, fill: 'none', stroke: colors[fit.segment], 'stroke-width': 2.2, 'stroke-dasharray': dashed ? '6 4' : 'none', 'data-fit': fit.segment, 'data-extrapolated': String(dashed)}));
    }
    selectedFits.forEach((fit) => {
      curve(fit, 10 ** lower, fit.min_pf_days, true);
      curve(fit, fit.min_pf_days, fit.max_pf_days, false);
      curve(fit, fit.max_pf_days, 10 ** upper, true);
      const targetCompute = crossing(fit, target), targetX = x(targetCompute), targetY = y(target);
      const marker = svg('path', {d: `M${targetX},${targetY-5} l5,5 -5,5 -5,-5 Z`, fill: colors[fit.segment], 'data-target-compute': targetCompute, 'data-fit': fit.segment});
      marker.append(svg('title', {}, `${fit.label}: ${format(targetCompute)} PF-days for ${percent(target)} (fitted curve)`)); plot.append(marker);
      plot.append(svg('circle', {cx: x(compute), cy: y(accuracy(fit, compute)), r: 5, fill: colors[fit.segment], stroke: 'white', 'stroke-width': 1.5, 'data-probe-fit': fit.segment}));
    });
    plot.append(svg('line', {x1: left, x2: right, y1: y(target), y2: y(target), stroke: '#9b9b9b', 'stroke-dasharray': '2 4'}));
    plot.append(svg('line', {x1: x(compute), x2: x(compute), y1: top, y2: bottom, stroke: '#333', 'stroke-dasharray': '4 4', 'data-role': 'compute-cursor'}));
    chart.append(plot);
    chart.style.cursor = 'crosshair';
    chart.addEventListener('click', (event) => {
      const bounds = chart.getBoundingClientRect();
      const fraction = Math.max(0, Math.min(1, ((event.clientX - bounds.left) * width / bounds.width - left) / (right - left)));
      $('fit-compute').value = String(Math.round(fraction * 1000)); readCompute();
    });
    container.replaceChildren(chart);
  }
  function render() {
    const target = Number($('fit-target').value) / 100;
    $('fit-target-value').textContent = `${Math.round(target * 100)}%`;
    $('fit-compute-value').textContent = `${format(compute)} PF-days`;
    $('fit-target-heading').textContent = `Compute for ${Math.round(target * 100)}%`;
    $('fit-compute').setAttribute('aria-valuetext', `${format(compute)} PF-days`);
    $('fit-target').setAttribute('aria-valuetext', `${Math.round(target * 100)} percent accuracy`);
    if (!selectedFits.length) {
      const message = document.createElement('p'); message.className = 'fit-chart-message'; message.textContent = panel.empty_fit_reason;
      $('fit-chart').replaceChildren(message); $('fit-status').textContent = 'There is no nonzero accuracy frontier to fit for this setting.'; return;
    }
    drawChart(target);
    $('fit-results').replaceChildren(...selectedFits.map((fit) => {
      const row = document.createElement('tr'), predicted = accuracy(fit, compute), projected = crossing(fit, target);
      row.dataset.fit = fit.segment; row.dataset.accuracy = String(predicted); row.dataset.targetCompute = String(projected);
      const values = [fit.label, fit.alpha.toFixed(4), fit.A.toFixed(4), predicted >= 0 && predicted <= 1 ? percent(predicted) : 'Outside accuracy range', `${format(projected)} PF-days`];
      values.forEach((value, index) => {
        const cell = document.createElement(index ? 'td' : 'th'); cell.textContent = value;
        if (!index) {cell.scope = 'row'; cell.className = `fit-${fit.segment}`;}
        if (index === 3) {const note = document.createElement('small'); note.textContent = predicted < 0 ? `Curve value: ${percent(predicted)}` : compute < fit.min_pf_days || compute > fit.max_pf_days ? 'Outside fitted interval' : 'Within fitted interval'; cell.append(note);}
        if (index === 4) {const note = document.createElement('small'); note.textContent = `${format(projected / panel.observed_max_pf_days)}× maximum evaluated compute`; cell.append(note);}
        row.append(cell);
      }); return row;
    }));
    $('fit-table-caption').textContent = `Fitted accuracy at ${format(compute)} PF-days and compute at the selected target`;
    $('fit-status').textContent = `Evaluated compute: ${format(panel.observed_min_pf_days)}–${format(panel.observed_max_pf_days)} PF-days. Selected compute is ${compute > panel.observed_max_pf_days ? 'beyond' : 'within'} this range.`;
  }
  function readCompute() {
    compute = 10 ** (lower + Number($('fit-compute').value) / 1000 * (upper - lower)); render();
  }
  ['fit-family', 'fit-metric', 'fit-window'].forEach((id) => $(id).addEventListener('change', configure));
  $('fit-target').addEventListener('input', configure);
  $('fit-compute').addEventListener('input', readCompute);
  let lastWidth = window.innerWidth;
  window.addEventListener('resize', () => {if (window.innerWidth !== lastWidth) {lastWidth = window.innerWidth; render();}});
  $('scaling-playground').hidden = false; configure();
})();

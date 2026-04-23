/* KOLF2.1J Perturbation Atlas viewer
   Three tabs: MDE (#mde), Clustermap (#clustermap), Clusters (#clusters).
*/

// ============================================================================
// Router
// ============================================================================
const TABS = ['mde', 'clustermap', 'clusters'];
const tabButtons = document.querySelectorAll('.tab');
const views      = Object.fromEntries(
  TABS.map(name => [name, document.getElementById(`view-${name}`)])
);

function showTab(name) {
  if (!TABS.includes(name)) name = 'mde';
  for (const t of TABS) {
    const active = (t === name);
    views[t].hidden = !active;
    views[t].classList.toggle('active', active);
  }
  tabButtons.forEach(b => {
    const active = (b.dataset.tab === name);
    b.classList.toggle('active', active);
    b.setAttribute('aria-selected', active);
  });
  if (location.hash !== `#${name}`) {
    history.replaceState(null, '', `#${name}`);
  }
  if (name === 'mde')        MDE.onShow();
  if (name === 'clustermap') Clustermap.onShow();
  if (name === 'clusters')   Clusters.onShow();
}

tabButtons.forEach(b => b.addEventListener('click', () => showTab(b.dataset.tab)));
window.addEventListener('hashchange', () => showTab(location.hash.replace('#', '')));

// ============================================================================
// MDE tab
// ============================================================================
const MDE = (() => {
  const PLOT   = document.getElementById('plot');
  const SEARCH = document.getElementById('search');
  const EMPTY  = document.getElementById('empty');
  const RESET  = document.getElementById('reset');
  const PANEL  = document.getElementById('panel');
  const PCLOSE = document.getElementById('panel-close');

  const P = {
    gene:    document.getElementById('p-gene'),
    leiden:  document.getElementById('p-leiden'),
    hdbText: document.getElementById('p-hdb-text'),
    hdbBtn:  document.getElementById('p-hdb-hl'),
    ndegs:   document.getElementById('p-ndegs'),
    edist:   document.getElementById('p-edist'),
    nbr:     document.getElementById('p-nbr'),
  };

  let data = [];
  let leidenLabels = {}, hdbscanLabels = {};
  let indexByGene = new Map();
  let nDegsRank = new Map(), edistRank = new Map();
  let totalRanked = 0;
  let xRange = null, yRange = null;
  let selectedIdx = null;
  let hdbscanHighlightOn = false;
  let hdbscanMembers = new Map();
  let plotted = false;

  const GRAY = '#c7c7c7';
  function clusterColor(v) {
    if (v === -1) return GRAY;
    const golden = 0.61803398875;
    const hue = (v * golden * 360) % 360;
    return `hsl(${hue.toFixed(1)},62%,48%)`;
  }

  function panelClusterText(p, which) {
    const cid = p[which];
    if (which === 'h' && cid === -1) return 'unclustered';
    const labels = which === 'l' ? leidenLabels : hdbscanLabels;
    const lab = labels[cid];
    return `cluster ${cid}: ${lab && lab.length ? lab : 'unlabeled'}`;
  }

  function buildTrace(records) {
    const x = new Array(records.length);
    const y = new Array(records.length);
    const text = new Array(records.length);
    const hovertext = new Array(records.length);
    const color = new Array(records.length);
    for (let i = 0; i < records.length; i++) {
      const r = records[i];
      x[i] = r.x; y[i] = r.y;
      text[i] = r.g;
      hovertext[i] = `<b>${r.g}</b>`;
      color[i] = clusterColor(r.l);
    }
    return {
      type: 'scattergl', mode: 'markers',
      x, y, text, hovertext,
      hovertemplate: '%{hovertext}<extra></extra>',
      marker: { color, size: 6, opacity: 0.85, line: { width: 0 } },
      selected:   { marker: { color: '#d62728', size: 9 } },
      unselected: { marker: { opacity: 0.85 } },
    };
  }

  const layout = {
    margin: { l: 24, r: 24, t: 12, b: 24 },
    xaxis: { visible: false, scaleanchor: 'y', scaleratio: 1 },
    yaxis: { visible: false },
    hovermode: 'closest', showlegend: false, dragmode: 'pan',
    plot_bgcolor: '#fff', paper_bgcolor: '#fff', annotations: [],
  };
  const config = {
    responsive: true, displaylogo: false, scrollZoom: true,
    modeBarButtonsToRemove: ['lasso2d','select2d','autoScale2d','toggleSpikelines'],
    toImageButtonOptions: { filename: 'KOLF_MDE', scale: 2, format: 'png' },
  };

  function initialAutorange() {
    const xs = data.map(r => r.x), ys = data.map(r => r.y);
    const pad = 0.04;
    const xmin = Math.min(...xs), xmax = Math.max(...xs);
    const ymin = Math.min(...ys), ymax = Math.max(...ys);
    const dx = (xmax - xmin) * pad, dy = (ymax - ymin) * pad;
    xRange = [xmin - dx, xmax + dx];
    yRange = [ymin - dy, ymax + dy];
  }
  function render() {
    Plotly.react(PLOT, [buildTrace(data)], {
      ...layout,
      xaxis: { ...layout.xaxis, range: xRange },
      yaxis: { ...layout.yaxis, range: yRange },
    }, config);
    plotted = true;
  }

  function buildHdbscanIndex() {
    hdbscanMembers = new Map();
    for (let i = 0; i < data.length; i++) {
      const h = data[i].h;
      if (h === -1) continue;
      if (!hdbscanMembers.has(h)) hdbscanMembers.set(h, []);
      hdbscanMembers.get(h).push(i);
    }
  }

  function computeRanks() {
    const byN = [...data].filter(r => r.n >= 0).sort((a,b) => b.n - a.n);
    byN.forEach((r,i) => nDegsRank.set(r.g, i+1));
    const byE = [...data].filter(r => r.e != null).sort((a,b) => b.e - a.e);
    byE.forEach((r,i) => edistRank.set(r.g, i+1));
    totalRanked = Math.max(byN.length, byE.length);
  }

  function annotateSelected(r) {
    return [{
      x: r.x, y: r.y, text: r.g, showarrow: true, arrowhead: 0,
      ax: 0, ay: -28, font: { size: 13, color: '#111' },
      bgcolor: 'rgba(255,255,255,0.9)', bordercolor: '#111', borderwidth: 1, borderpad: 3,
    }];
  }

  function focusGene(idx, { zoom = false } = {}) {
    const r = data[idx];
    selectedIdx = idx;
    hdbscanHighlightOn = false;
    Plotly.restyle(PLOT, { selectedpoints: [null] });
    const relayout = { annotations: annotateSelected(r) };
    if (zoom) {
      const pad = Math.max((xRange[1] - xRange[0]), (yRange[1] - yRange[0])) * 0.06;
      relayout['xaxis.range'] = [r.x - pad, r.x + pad];
      relayout['yaxis.range'] = [r.y - pad, r.y + pad];
    }
    Plotly.relayout(PLOT, relayout);
    openPanel(idx);
  }

  function clearHighlight() {
    selectedIdx = null;
    hdbscanHighlightOn = false;
    Plotly.restyle(PLOT, { selectedpoints: [null] });
    Plotly.relayout(PLOT, { annotations: [] });
  }

  function setHdbBtnState() {
    const r = selectedIdx != null ? data[selectedIdx] : null;
    const eligible = r && r.h !== -1 && hdbscanMembers.has(r.h);
    P.hdbBtn.hidden = !eligible;
    P.hdbBtn.classList.toggle('on', hdbscanHighlightOn);
    P.hdbBtn.textContent = hdbscanHighlightOn ? 'clear' : 'highlight cluster';
  }

  function toggleHdbHighlight() {
    if (selectedIdx == null) return;
    const r = data[selectedIdx];
    if (r.h === -1 || !hdbscanMembers.has(r.h)) return;
    hdbscanHighlightOn = !hdbscanHighlightOn;
    Plotly.restyle(PLOT, {
      selectedpoints: hdbscanHighlightOn
        ? [hdbscanMembers.get(r.h).slice()] : [null],
    });
    setHdbBtnState();
  }

  function searchHighlight(query) {
    const q = query.trim().toUpperCase();
    if (!q) { EMPTY.hidden = true; return; }
    let idx = data.findIndex(r => r.g.toUpperCase() === q);
    if (idx < 0) idx = data.findIndex(r => r.g.toUpperCase().startsWith(q));
    if (idx < 0) idx = data.findIndex(r => r.g.toUpperCase().includes(q));
    if (idx < 0) { EMPTY.hidden = false; return; }
    EMPTY.hidden = true;
    focusGene(idx, { zoom: true });
  }

  function resetView() {
    SEARCH.value = '';
    EMPTY.hidden = true;
    closePanel();
    clearHighlight();
    Plotly.relayout(PLOT, { 'xaxis.range': xRange, 'yaxis.range': yRange });
  }

  const fmtInt   = v => v == null || v < 0 ? '—' : v.toLocaleString();
  const fmtFloat = (v, d=2) => v == null ? '—' : Number(v).toFixed(d);
  const fmtRank  = (rank, total) => rank ? `rank ${rank} / ${total}` : '—';
  const fmtWithRank = (s, rank, total) => rank ? `${s} · ${fmtRank(rank, total)}` : s;

  function nearestNeighbors(idx, k = 10) {
    const a = data[idx]; const out = [];
    for (let i = 0; i < data.length; i++) {
      if (i === idx) continue;
      const b = data[i]; const dx = a.x - b.x, dy = a.y - b.y;
      out.push({ i, d: Math.sqrt(dx*dx + dy*dy) });
    }
    out.sort((u,v) => u.d - v.d);
    return out.slice(0, k);
  }

  function renderNeighbors(ol, neighbors) {
    ol.innerHTML = '';
    for (const n of neighbors) {
      const r = data[n.i];
      const li = document.createElement('li');
      const g = document.createElement('span'); g.className = 'g'; g.textContent = r.g;
      g.addEventListener('click', () => focusGene(n.i, { zoom: true }));
      const d = document.createElement('span'); d.className = 'd'; d.textContent = n.d.toFixed(2);
      li.appendChild(g); li.appendChild(d); ol.appendChild(li);
    }
  }

  function openPanel(idx) {
    const r = data[idx];
    P.gene.textContent    = r.g;
    P.leiden.textContent  = panelClusterText(r, 'l');
    P.hdbText.textContent = panelClusterText(r, 'h');
    setHdbBtnState();
    P.ndegs.textContent = fmtWithRank(fmtInt(r.n), nDegsRank.get(r.g), totalRanked);
    P.edist.textContent = fmtWithRank(fmtFloat(r.e, 2), edistRank.get(r.g), totalRanked);
    renderNeighbors(P.nbr, nearestNeighbors(idx, 10));
    PANEL.classList.add('open');
    PANEL.setAttribute('aria-hidden', 'false');
  }
  function closePanel() {
    PANEL.classList.remove('open');
    PANEL.setAttribute('aria-hidden', 'true');
  }

  function attachEvents() {
    SEARCH.addEventListener('input',   e => searchHighlight(e.target.value));
    SEARCH.addEventListener('keydown', e => { if (e.key === 'Escape') { SEARCH.value = ''; resetView(); } });
    RESET.addEventListener('click', resetView);
    PCLOSE.addEventListener('click', () => { closePanel(); clearHighlight(); });
    P.hdbBtn.addEventListener('click', toggleHdbHighlight);
    PLOT.on('plotly_click', ev => {
      const pt = ev.points && ev.points[0]; if (!pt) return;
      focusGene(pt.pointIndex, { zoom: false });
    });
  }

  function init() {
    return fetch('data/mde.json?v=11').then(r => r.json()).then(payload => {
      data = payload.points;
      leidenLabels  = payload.leiden_labels  || {};
      hdbscanLabels = payload.hdbscan_labels || {};
      indexByGene = new Map(data.map((r,i) => [r.g, i]));
      buildHdbscanIndex();
      computeRanks();
      initialAutorange();
      render();
      attachEvents();
    }).catch(err => {
      PLOT.innerHTML = `<pre style="padding:24px;color:#a00">Failed to load data/mde.json: ${err}</pre>`;
    });
  }

  function onShow() {
    if (plotted) Plotly.Plots.resize(PLOT);
  }

  return { init, onShow, jumpTo: gene => {
    const idx = indexByGene.get(gene); if (idx != null) focusGene(idx, { zoom: true });
  }};
})();

// ============================================================================
// Clustermap tab — pert × pert Pearson on z-normed pseudobulks.
//   Main pane: hierarchically clustered 1655 x 1655 + HDBSCAN color strip.
//   Side pane: 314 perts with HDBSCAN != -1, sorted by cluster id, with
//              cluster-boundary separator lines + HDBSCAN color strip.
// ============================================================================
const Clustermap = (() => {
  const MAIN   = document.getElementById('cmap-main');
  const SIDE   = document.getElementById('cmap-side');
  const SEARCH = document.getElementById('csearch');
  const SUG    = document.getElementById('csuggest');
  const HINT   = document.getElementById('chint');
  const META   = document.getElementById('cmeta');
  const RESET  = document.getElementById('creset');

  let meta = null;                  // full metadata
  let perts = [];                   // main matrix order
  let M = null;                     // Float32Array length n*n
  let n = 0;
  let indexByGene = new Map();
  let plottedMain = false, plottedSide = false;
  let activeSugg = -1;

  // ---- color helpers ----
  const GRAY = '#c7c7c7';
  function hdbscanColor(h) {
    if (h === -1) return GRAY;
    const golden = 0.61803398875;
    const hue = (h * golden * 360) % 360;
    return `hsl(${hue.toFixed(1)},62%,48%)`;
  }
  // Discrete colorscale: each unique id -> a step in [0,1]; z values are
  // normalized indices (idx/N + 0.5/N), zmin=0, zmax=1.
  function discreteScaleAndZ(ids) {
    const uniq = [...new Set(ids)].sort((a,b) => a - b);
    const N = uniq.length;
    const idxOf = new Map(uniq.map((v,i) => [v, i]));
    const scale = [];
    for (let i = 0; i < N; i++) {
      const c = hdbscanColor(uniq[i]);
      scale.push([i / N, c]);
      const t1 = (i + 1) / N;
      scale.push([Math.min(t1, 1.0), c]);
    }
    const zNorm = ids.map(v => (idxOf.get(v) + 0.5) / N);
    return { scale, zNorm, uniq };
  }

  // ---- Plotly common config ----
  const config = {
    responsive: true, displaylogo: false, scrollZoom: false,
    modeBarButtonsToRemove: ['lasso2d','select2d','toggleSpikelines'],
    toImageButtonOptions: { filename: 'KOLF_clustermap', scale: 2, format: 'png' },
  };

  function init() {
    HINT.hidden = false;
    HINT.textContent = 'Loading correlation matrices (~2.8 MB)…';
    return Promise.all([
      fetch('data/clustermap/meta.json?v=11').then(r => r.json()),
      fetch('data/clustermap/corr_int8.bin?v=11').then(r => r.arrayBuffer()),
      fetch('data/clustermap/corr_side_int8.bin?v=11').then(r => r.arrayBuffer()),
    ]).then(([m, mainBuf, sideBuf]) => {
      meta  = m;
      perts = m.perts;
      n     = m.n;
      const scale = m.scale || 127;

      // Main matrix
      const mi8 = new Int8Array(mainBuf);
      if (mi8.length !== n * n) throw new Error(`main bin length ${mi8.length} != ${n*n}`);
      M = new Float32Array(n * n);
      for (let i = 0; i < mi8.length; i++) M[i] = mi8[i] / scale;
      indexByGene = new Map(perts.map((g, i) => [g, i]));

      // Side matrix
      const ns = m.side.n;
      const si8 = new Int8Array(sideBuf);
      if (si8.length !== ns * ns) throw new Error(`side bin length ${si8.length} != ${ns*ns}`);
      const Ms = new Float32Array(ns * ns);
      for (let i = 0; i < si8.length; i++) Ms[i] = si8[i] / scale;

      META.textContent = `${n} perts (left)  ·  ${ns} HDBSCAN-clustered (right)  ·  Pearson r`;
      attachEvents();
      renderMain(M);
      renderSide(Ms, ns);
      HINT.hidden = true;
    }).catch(err => {
      HINT.textContent = `Failed to load clustermap: ${err.message || err}`;
    });
  }

  function onShow() {
    if (plottedMain) Plotly.Plots.resize(MAIN);
    if (plottedSide) Plotly.Plots.resize(SIDE);
    if (SEARCH) SEARCH.focus();
  }

  // -------------------- MAIN pane --------------------
  function renderMain(M) {
    const z = new Array(n);
    for (let i = 0; i < n; i++) z[i] = M.subarray(i * n, (i + 1) * n);

    const { scale: hdbScale, zNorm: hdbZ } = discreteScaleAndZ(meta.hdbscan);
    const customLabels = meta.hdbscan.map(h => h === -1 ? 'noise' : `${h}`);

    const stripTrace = {
      type: 'heatmap',
      z: hdbZ.map(v => [v]),
      x: [0], y: perts,
      xaxis: 'x', yaxis: 'y',
      colorscale: hdbScale, zmin: 0, zmax: 1,
      showscale: false,
      customdata: customLabels.map(l => [l]),
      hovertemplate: '<b>%{y}</b><br>HDBSCAN %{customdata[0]}<extra></extra>',
    };
    const heatTrace = {
      type: 'heatmap',
      z, x: perts, y: perts,
      xaxis: 'x2', yaxis: 'y',
      colorscale: 'RdBu', reversescale: true,    // inverted: red=high, blue=low
      zmin: -0.2, zmax: 0.2,                      // tighter dynamic range for main
      hovertemplate: '<b>%{y}</b> × <b>%{x}</b><br>r = %{z:.3f}<extra></extra>',
      colorbar: {
        title: { text: 'Pearson r', font: { size: 10 } },
        thickness: 8, len: 0.5, x: 1.02, xanchor: 'left',
        tickfont: { size: 10 },
      },
    };
    const layout = {
      margin: { l: 12, r: 70, t: 12, b: 12 },
      xaxis:  { domain: [0, 0.025], showticklabels: false, ticks: '', fixedrange: true },
      xaxis2: { domain: [0.04, 1.0], showticklabels: false, ticks: '',
                scaleanchor: 'y', constrain: 'domain' },          // square cells
      yaxis:  { showticklabels: false, ticks: '', autorange: 'reversed', constrain: 'domain' },
      hovermode: 'closest', showlegend: false, dragmode: 'zoom',
      plot_bgcolor: '#fff', paper_bgcolor: '#fff',
      annotations: [], shapes: [],
    };
    Plotly.react(MAIN, [stripTrace, heatTrace], layout, config);
    plottedMain = true;
  }

  // -------------------- SIDE pane --------------------
  function renderSide(Ms, ns) {
    const sideMeta = meta.side;
    const z = new Array(ns);
    for (let i = 0; i < ns; i++) z[i] = Ms.subarray(i * ns, (i + 1) * ns);

    const { scale: hdbScale, zNorm: hdbZ } = discreteScaleAndZ(sideMeta.hdbscan);
    const customLabels = sideMeta.hdbscan.map(h => `${h}`);

    const stripTrace = {
      type: 'heatmap',
      z: hdbZ.map(v => [v]),
      x: [0], y: sideMeta.perts,
      xaxis: 'x', yaxis: 'y',
      colorscale: hdbScale, zmin: 0, zmax: 1,
      showscale: false,
      customdata: customLabels.map(l => [l]),
      hovertemplate: '<b>%{y}</b><br>HDBSCAN %{customdata[0]}<extra></extra>',
    };
    const heatTrace = {
      type: 'heatmap',
      z, x: sideMeta.perts, y: sideMeta.perts,
      xaxis: 'x2', yaxis: 'y',
      colorscale: 'RdBu', reversescale: true,    // inverted: red=high, blue=low
      zmin: -0.5, zmax: 0.5,                      // wider range for the cluster blocks
      hovertemplate: '<b>%{y}</b> × <b>%{x}</b><br>r = %{z:.3f}<extra></extra>',
      showscale: false,
    };

    // Cluster boundary lines on side heatmap (use xaxis2/yaxis)
    const shapes = [];
    for (const b of sideMeta.boundaries) {
      // horizontal across the heatmap subplot
      shapes.push({ type: 'line', xref: 'x2', yref: 'y',
        x0: -0.5, x1: ns - 0.5, y0: b - 0.5, y1: b - 0.5,
        line: { color: '#111', width: 0.5 } });
      shapes.push({ type: 'line', xref: 'x2', yref: 'y',
        x0: b - 0.5, x1: b - 0.5, y0: -0.5, y1: ns - 0.5,
        line: { color: '#111', width: 0.5 } });
    }

    const layout = {
      margin: { l: 12, r: 12, t: 12, b: 12 },
      xaxis:  { domain: [0, 0.04], showticklabels: false, ticks: '', fixedrange: true },
      xaxis2: { domain: [0.06, 1.0], showticklabels: false, ticks: '',
                scaleanchor: 'y', constrain: 'domain' },          // square cells
      yaxis:  { showticklabels: false, ticks: '', autorange: 'reversed', constrain: 'domain' },
      hovermode: 'closest', showlegend: false, dragmode: 'zoom',
      plot_bgcolor: '#fff', paper_bgcolor: '#fff',
      annotations: [], shapes,
    };
    Plotly.react(SIDE, [stripTrace, heatTrace], layout, config);
    plottedSide = true;
  }

  // -------------------- Search / suggest --------------------
  function attachEvents() {
    SEARCH.addEventListener('input',   () => updateSuggest(SEARCH.value));
    SEARCH.addEventListener('focus',   () => updateSuggest(SEARCH.value));
    SEARCH.addEventListener('blur',    () => setTimeout(closeSuggest, 120));
    SEARCH.addEventListener('keydown', onSearchKey);
    RESET.addEventListener('click', resetView);
  }
  function fuzzy(query) {
    const q = query.trim().toUpperCase();
    if (!q) return perts.slice(0, 12);
    const eq  = perts.filter(p => p.toUpperCase() === q);
    const pre = perts.filter(p => p.toUpperCase().startsWith(q) && p.toUpperCase() !== q);
    const sub = perts.filter(p => !p.toUpperCase().startsWith(q) && p.toUpperCase().includes(q));
    return [...eq, ...pre, ...sub].slice(0, 12);
  }
  function updateSuggest(q) {
    const list = fuzzy(q);
    if (!list.length) { closeSuggest(); return; }
    SUG.innerHTML = '';
    list.forEach(p => {
      const li = document.createElement('li');
      const idx = q ? p.toUpperCase().indexOf(q.trim().toUpperCase()) : -1;
      if (idx >= 0 && q.trim()) {
        const a = p.slice(0, idx);
        const m = p.slice(idx, idx + q.trim().length);
        const z = p.slice(idx + q.trim().length);
        li.innerHTML = `${a}<mark>${m}</mark>${z}`;
      } else {
        li.textContent = p;
      }
      li.addEventListener('mousedown', e => { e.preventDefault(); selectPert(p); });
      SUG.appendChild(li);
    });
    activeSugg = -1;
    SUG.hidden = false;
  }
  function closeSuggest() { SUG.hidden = true; activeSugg = -1; }
  function onSearchKey(e) {
    if (SUG.hidden) return;
    const items = SUG.querySelectorAll('li');
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      activeSugg = Math.min(items.length - 1, activeSugg + 1);
      items.forEach((it,i) => it.classList.toggle('active', i === activeSugg));
      items[activeSugg]?.scrollIntoView({ block: 'nearest' });
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      activeSugg = Math.max(0, activeSugg - 1);
      items.forEach((it,i) => it.classList.toggle('active', i === activeSugg));
      items[activeSugg]?.scrollIntoView({ block: 'nearest' });
    } else if (e.key === 'Enter') {
      e.preventDefault();
      const pick = activeSugg >= 0 ? items[activeSugg].textContent
                                   : (items[0] && items[0].textContent);
      if (pick) selectPert(pick);
    } else if (e.key === 'Escape') {
      closeSuggest();
    }
  }
  function selectPert(p) {
    const idx = indexByGene.get(p);
    if (idx == null) return;
    SEARCH.value = p;
    closeSuggest();
    const W = 60;
    const x0 = Math.max(0, idx - W);
    const x1 = Math.min(n - 1, idx + W);
    Plotly.relayout(MAIN, {
      shapes: [
        { type: 'line', xref: 'x2', yref: 'paper', x0: idx, x1: idx, y0: 0, y1: 1,
          line: { color: '#111', width: 1, dash: 'dot' } },
        { type: 'line', xref: 'paper', yref: 'y', x0: 0, x1: 1, y0: idx, y1: idx,
          line: { color: '#111', width: 1, dash: 'dot' } },
      ],
      annotations: [
        { x: idx, y: 0, xref: 'x2', yref: 'paper',
          text: p, showarrow: false, yanchor: 'bottom', yshift: 4,
          font: { size: 11, color: '#111' },
          bgcolor: 'rgba(255,255,255,0.85)', bordercolor: '#111',
          borderwidth: 1, borderpad: 2 },
      ],
      'xaxis2.range': [x0, x1],
      'yaxis.range':  [x1, x0],
    });
  }
  function resetView() {
    SEARCH.value = '';
    closeSuggest();
    Plotly.relayout(MAIN, {
      shapes: [], annotations: [],
      'xaxis2.autorange': true, 'yaxis.autorange': 'reversed',
    });
  }

  return { init, onShow };
})();

// ============================================================================
// Clusters tab (placeholder until z-score matrix lands)
// ============================================================================
const Clusters = (() => {
  function init() { /* nothing yet */ }
  function onShow() { /* nothing yet */ }
  return { init, onShow };
})();

// ============================================================================
// Boot
// ============================================================================
MDE.init();
Clustermap.init();
Clusters.init();

// Initial route
const initial = location.hash.replace('#', '');
showTab(TABS.includes(initial) ? initial : 'mde');

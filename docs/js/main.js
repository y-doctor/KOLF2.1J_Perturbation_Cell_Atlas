/* KOLF2.1J Perturbation Atlas viewer
   Tabs: Overview (#home), MDE (#mde), Clustermaps (#clustermap, toggle pert/gene),
         Gene query (#genequery), Clusters (#clusters), Gene panel (#panel).
*/

// ============================================================================
// Router
// ============================================================================
const TABS = ['home', 'mde', 'clustermap', 'genequery', 'clusters', 'panel'];
const tabButtons = document.querySelectorAll('.tab');
const views      = Object.fromEntries(
  TABS.map(name => [name, document.getElementById(`view-${name}`)])
);

// Clustermap subview mode: 'pert' or 'gene'. Toggled within the tab.
let cmapMode = 'pert';
function onCmapShow() {
  if (cmapMode === 'pert') Clustermap.onShow();
  else                     GeneClustermap.onShow();
}

function showTab(name) {
  if (!TABS.includes(name)) name = 'home';
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
  if (name === 'clustermap') onCmapShow();
  if (name === 'genequery')  GeneQuery.onShow();
  if (name === 'clusters')   Clusters.onShow();
  if (name === 'panel')      Panel.onShow();
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
    fit:     document.getElementById('p-fit'),
    nbr:     document.getElementById('p-nbr'),
  };

  let fitnessCache = null;
  function ensureFitness() { return fitnessCache || (fitnessCache = fetch('data/mde/fitness.json?v=35').then(r => r.json())); }

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

  function renderDegsTable(tbody, rows) {
    tbody.innerHTML = '';
    if (!rows || rows.length === 0) {
      const tr = document.createElement('tr');
      tr.innerHTML = '<td colspan="2" style="color:var(--muted)">—</td>';
      tbody.appendChild(tr);
      return;
    }
    for (const r of rows) {
      const tr = document.createElement('tr');
      const g = document.createElement('td'); g.textContent = r.g;
      const z = document.createElement('td');
      z.textContent = (r.z >= 0 ? '+' : '') + r.z.toFixed(2);
      tr.appendChild(g); tr.appendChild(z);
      tbody.appendChild(tr);
    }
  }
  function openPanel(idx) {
    const r = data[idx];
    const gene = r.g;
    P.gene.textContent    = gene;
    P.leiden.textContent  = panelClusterText(r, 'l');
    P.hdbText.textContent = panelClusterText(r, 'h');
    setHdbBtnState();
    P.ndegs.textContent = fmtWithRank(fmtInt(r.n), nDegsRank.get(gene), totalRanked);
    P.edist.textContent = fmtWithRank(fmtFloat(r.e, 2), edistRank.get(gene), totalRanked);
    P.fit.textContent   = '…';
    renderNeighbors(P.nbr, nearestNeighbors(idx, 10));
    PANEL.classList.add('open');
    PANEL.setAttribute('aria-hidden', 'false');

    ensureFitness().then(fit => {
      const f = fit.genes && fit.genes[gene];
      if (!f) { P.fit.textContent = '—'; return; }
      const sign = f.z >= 0 ? '+' : '';
      const rank = f.z >= 0 ? `pos #${f.rp.toLocaleString()}` : `neg #${f.rn.toLocaleString()}`;
      P.fit.textContent = `${sign}${f.z.toFixed(2)} · ${rank} / ${fit.n_total.toLocaleString()}`;
    }).catch(e => { console.error('fitness load failed', e); P.fit.textContent = '—'; });
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
    return fetch('data/mde.json?v=35').then(r => r.json()).then(payload => {
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
// Clustermap factory — produces a tab module that loads two int8 binary
// correlation matrices (main + hdbscan-filtered side), renders them as
// Plotly heatmaps with HDBSCAN color strips, and supports search + zoom.
// Used twice: once for pert x pert and once for gene x gene.
// ============================================================================
function makeClustermap(opts) {
  const MAIN   = document.getElementById(opts.mainId);
  const SIDE   = document.getElementById(opts.sideId);
  const SEARCH = document.getElementById(opts.searchId);
  const SUG    = document.getElementById(opts.suggestId);
  const HINT   = document.getElementById(opts.hintId);
  const META   = document.getElementById(opts.metaId);
  const RESET  = document.getElementById(opts.resetId);

  let meta = null;                  // full metadata
  let perts = [];                   // main matrix order (genes or perts)
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
    HINT.textContent = `Loading ${opts.label} matrices…`;
    return Promise.all([
      fetch(opts.metaPath).then(r => r.json()),
      fetch(opts.mainPath).then(r => r.arrayBuffer()),
      fetch(opts.sidePath).then(r => r.arrayBuffer()),
    ]).then(([m, mainBuf, sideBuf]) => {
      meta  = m;
      perts = m.perts;
      n     = m.n;
      const scale = m.scale || 127;

      const mi8 = new Int8Array(mainBuf);
      if (mi8.length !== n * n) throw new Error(`main bin length ${mi8.length} != ${n*n}`);
      M = new Float32Array(n * n);
      for (let i = 0; i < mi8.length; i++) M[i] = mi8[i] / scale;
      indexByGene = new Map(perts.map((g, i) => [g, i]));

      const ns = m.side.n;
      const si8 = new Int8Array(sideBuf);
      if (si8.length !== ns * ns) throw new Error(`side bin length ${si8.length} != ${ns*ns}`);
      const Ms = new Float32Array(ns * ns);
      for (let i = 0; i < si8.length; i++) Ms[i] = si8[i] / scale;

      META.textContent = `${n} ${opts.unit} (left) · ${ns} HDBSCAN-clustered (right) · Pearson r`;
      attachEvents();
      renderMain(M);
      renderSide(Ms, ns);
      HINT.hidden = true;
    }).catch(err => {
      HINT.textContent = `Failed to load ${opts.label}: ${err.message || err}`;
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
    const clLabels = (meta.side && meta.side.cluster_labels) || {};
    const customLabels = meta.hdbscan.map(h => {
      if (h === -1) return 'noise';
      const lbl = clLabels[String(h)];
      return lbl ? `${h} · ${lbl}` : `${h}`;
    });

    const stripTrace = {
      type: 'heatmap',
      z: hdbZ.map(v => [v]),
      x: [0], y: perts,
      xaxis: 'x', yaxis: 'y',
      colorscale: hdbScale, zmin: 0, zmax: 1,
      showscale: false,
      text: customLabels.map(l => [l]),
      hovertemplate: '<b>%{y}</b><br>HDBSCAN %{text}<extra></extra>',
    };
    const heatTrace = {
      type: 'heatmap',
      z, x: perts, y: perts,
      xaxis: 'x2', yaxis: 'y',
      colorscale: 'RdBu', reversescale: true,    // inverted: red=high, blue=low
      zmin: opts.mainZmin, zmax: opts.mainZmax,
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
    const cluster_labels = (sideMeta.cluster_labels) || {};
    const customLabels = sideMeta.hdbscan.map(h => {
      const lbl = cluster_labels[String(h)];
      return lbl ? `${h} · ${lbl}` : `${h}`;
    });

    const stripTrace = {
      type: 'heatmap',
      z: hdbZ.map(v => [v]),
      x: [0], y: sideMeta.perts,
      xaxis: 'x', yaxis: 'y',
      colorscale: hdbScale, zmin: 0, zmax: 1,
      showscale: false,
      text: customLabels.map(l => [l]),
      hovertemplate: '<b>%{y}</b><br>HDBSCAN %{text}<extra></extra>',
    };
    const heatTrace = {
      type: 'heatmap',
      z, x: sideMeta.perts, y: sideMeta.perts,
      xaxis: 'x2', yaxis: 'y',
      colorscale: 'RdBu', reversescale: true,    // inverted: red=high, blue=low
      zmin: opts.sideZmin, zmax: opts.sideZmax,
      hovertemplate: '<b>%{y}</b> × <b>%{x}</b><br>r = %{z:.3f}<extra></extra>',
      colorbar: {
        title: { text: 'Pearson r', font: { size: 10 } },
        thickness: 8, len: 0.5, x: 1.02, xanchor: 'left',
        tickfont: { size: 10 },
      },
    };

    // Cluster boundary lines (labels are hover-only now)
    const shapes = [];
    for (const b of sideMeta.boundaries) {
      shapes.push({ type: 'line', xref: 'x2', yref: 'y',
        x0: -0.5, x1: ns - 0.5, y0: b - 0.5, y1: b - 0.5,
        line: { color: '#111', width: 0.5 } });
      shapes.push({ type: 'line', xref: 'x2', yref: 'y',
        x0: b - 0.5, x1: b - 0.5, y0: -0.5, y1: ns - 0.5,
        line: { color: '#111', width: 0.5 } });
    }

    const layout = {
      margin: { l: 12, r: 70, t: 12, b: 12 },                   // labels are hover-only
      xaxis:  { domain: [0, 0.04], showticklabels: false, ticks: '', fixedrange: true },
      xaxis2: { domain: [0.06, 1.0], showticklabels: false, ticks: '',
                scaleanchor: 'y', constrain: 'domain' },          // square cells
      yaxis:  {
        side: 'left', showticklabels: false, ticks: '',
        autorange: 'reversed', constrain: 'domain',
      },
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
}

const Clustermap = makeClustermap({
  label:    'pert clustermap',
  unit:     'perts',
  mainId:   'cmap-main', sideId:   'cmap-side',
  searchId: 'csearch',   suggestId:'csuggest',
  hintId:   'chint',     metaId:   'cmeta',     resetId: 'creset',
  metaPath: 'data/clustermap/meta.json?v=35',
  mainPath: 'data/clustermap/corr_int8.bin?v=35',
  sidePath: 'data/clustermap/corr_side_int8.bin?v=35',
  mainZmin: -0.2, mainZmax: 0.2,
  sideZmin: -0.5, sideZmax: 0.5,
});

const GeneClustermap = makeClustermap({
  label:    'gene clustermap',
  unit:     'genes',
  mainId:   'gmap-main', sideId:   'gmap-side',
  searchId: 'gsearch',   suggestId:'gsuggest',
  hintId:   'ghint',     metaId:   'gmeta',     resetId: 'greset',
  metaPath: 'data/genemap/meta.json?v=35',
  mainPath: 'data/genemap/gene_corr_int8.bin?v=35',
  sidePath: 'data/genemap/gene_corr_side_int8.bin?v=35',
  mainZmin: -0.2, mainZmax: 0.2,
  sideZmin: -0.5, sideZmax: 0.5,
});

// ============================================================================
// Query tab — dual mode:
//   per-gene  : pick an expressed gene, see top 50 up/dn perts + histogram
//   per-pert  : pick a perturbation, see top 50 up/dn genes + histogram
// ============================================================================
const GeneQuery = (() => {
  const SEARCH = document.getElementById('qsearch');
  const SUG    = document.getElementById('qsuggest');
  const META   = document.getElementById('qmeta');
  const HINT   = document.getElementById('qhint');
  const RESULTS= document.getElementById('qresults');
  const GENE   = document.getElementById('q-gene');
  const SUB    = document.getElementById('q-sub');
  const HIST   = document.getElementById('q-hist');
  const UP     = document.getElementById('q-up');
  const DN     = document.getElementById('q-dn');
  const UP_H   = document.getElementById('q-up-h');
  const DN_H   = document.getElementById('q-dn-h');

  const MODES = {
    gene: {
      indexPath: 'data/genes/index.json?v=35',
      topkPath:  'data/genes/topk.json?v=35',
      indexKey:  'genes',       // index.json key holding the searchable list
      counterKey:'n_perts',     // index.json key for the "other-axis" size
      entity:    'feature gene',
      other:     'perturbations',
      titlePrefix: 'Feature Gene: ',
      placeholder: 'Search feature gene…',
      hintHtml:  '<p>Pick a <strong>feature gene</strong> to see the perturbations that most strongly <strong>up-regulate</strong> or <strong>down-regulate</strong> it (NTC z-score, post-basic-QC universe of all perturbations).</p>',
      upHeader:  'Top 50 perturbations · highest z-score',
      dnHeader:  'Top 50 perturbations · lowest z-score',
      axisX:     'NTC z-score',
      axisY:     '# perturbations',
      hoverUnit: 'perts',
      defaultPick: 'POU5F1',
    },
    pert: {
      indexPath: 'data/perts/index.json?v=35',
      topkPath:  'data/perts/topk.json?v=35',
      indexKey:  'perts',
      counterKey:'n_genes',
      entity:    'perturbed gene',
      other:     'genes',
      titlePrefix: 'Perturbed Gene: ',
      placeholder: 'Search perturbed gene…',
      hintHtml:  '<p>Pick a <strong>perturbed gene</strong> (CRISPRi knockdown target) to see the genes it most strongly <strong>up-regulates</strong> or <strong>down-regulates</strong> (NTC z-score across expressed genes).</p>',
      upHeader:  'Top 50 genes · highest z-score',
      dnHeader:  'Top 50 genes · lowest z-score',
      axisX:     'NTC z-score',
      axisY:     '# genes',
      hoverUnit: 'genes',
      defaultPick: 'DNMT1',
    },
  };

  // Per-mode state: each mode caches its own index + topk payload so switching
  // doesn't re-fetch.
  const state = {
    gene: { keys: [], other: 0, K: 50, binCenters: [], binEdges: [], topk: null, topkLoading: null, indexPromise: null },
    pert: { keys: [], other: 0, K: 50, binCenters: [], binEdges: [], topk: null, topkLoading: null, indexPromise: null },
  };
  let mode = 'gene';
  let activeSugg = -1;
  const autoOpened = { gene: false, pert: false };

  function cfg() { return MODES[mode]; }
  function S()   { return state[mode]; }

  function init() {
    // Kick off gene-mode index immediately (pre-fetch for snappy first-load).
    ensureIndex('gene');
    attachEvents();
    return state.gene.indexPromise;
  }

  function ensureIndex(m) {
    const s = state[m];
    if (s.indexPromise) return s.indexPromise;
    const c = MODES[m];
    s.indexPromise = fetch(c.indexPath).then(r => r.json()).then(idx => {
      s.keys       = idx[c.indexKey] || [];
      s.other      = idx[c.counterKey] || 0;
      s.K          = idx.k || 50;
      s.binCenters = idx.bin_centers || [];
      s.binEdges   = idx.bin_edges   || [];
      if (m === mode) refreshMeta();
    }).catch(err => {
      if (m === mode) META.textContent = `Failed to load ${c.entity} index: ${err.message || err}`;
      throw err;
    });
    return s.indexPromise;
  }

  function ensureTopk(m) {
    const s = state[m];
    if (s.topk) return Promise.resolve(s.topk);
    if (s.topkLoading) return s.topkLoading;
    const c = MODES[m];
    META.textContent = `Loading per-${c.entity} query data…`;
    s.topkLoading = fetch(c.topkPath).then(r => r.json()).then(data => {
      s.topk = data;
      if (m === mode) refreshMeta();
      return s.topk;
    });
    return s.topkLoading;
  }

  function refreshMeta() {
    const c = cfg(), s = S();
    if (!s.keys.length) return;
    META.textContent = `${s.keys.length.toLocaleString()} ${c.entity}s · ${s.other.toLocaleString()} ${c.other}`;
  }

  function autoOpenForCurrentMode() {
    if (autoOpened[mode]) return;
    const m = mode, c = MODES[m], s = state[m];
    const p = s.indexPromise || ensureIndex(m);
    p.then(() => {
      if (autoOpened[m] || mode !== m) return;
      if (s.keys.includes(c.defaultPick) && RESULTS && RESULTS.hidden) {
        autoOpened[m] = true;
        selectItem(c.defaultPick);
      }
    }).catch(() => {});
  }

  function onShow() {
    if (SEARCH) SEARCH.focus();
    autoOpenForCurrentMode();
  }

  function applyMode() {
    const c = cfg();
    SEARCH.value = '';
    SEARCH.placeholder = c.placeholder;
    HINT.innerHTML = c.hintHtml;
    HINT.hidden = false;
    RESULTS.hidden = true;
    UP_H.textContent = c.upHeader;
    DN_H.textContent = c.dnHeader;
    closeSuggest();
    Plotly.purge(HIST);
    fillTable(UP, []); fillTable(DN, []);
    ensureIndex(mode).then(refreshMeta).catch(() => {});
    SEARCH.focus();
    autoOpenForCurrentMode();
  }

  function attachEvents() {
    SEARCH.addEventListener('input',   () => updateSuggest(SEARCH.value));
    SEARCH.addEventListener('focus',   () => updateSuggest(SEARCH.value));
    SEARCH.addEventListener('blur',    () => setTimeout(closeSuggest, 120));
    SEARCH.addEventListener('keydown', onSearchKey);
    document.querySelectorAll('#view-genequery .seg').forEach(btn => {
      btn.addEventListener('click', () => {
        const newMode = btn.dataset.qmode;
        if (newMode === mode) return;
        document.querySelectorAll('#view-genequery .seg').forEach(b => {
          const on = (b === btn);
          b.classList.toggle('active', on);
          b.setAttribute('aria-checked', on);
        });
        mode = newMode;
        applyMode();
      });
    });
  }

  function fuzzy(q) {
    const keys = S().keys;
    q = q.trim().toUpperCase();
    if (!q) return keys.slice(0, 12);
    const eq  = keys.filter(g => g.toUpperCase() === q);
    const pre = keys.filter(g => g.toUpperCase().startsWith(q) && g.toUpperCase() !== q);
    const sub = keys.filter(g => !g.toUpperCase().startsWith(q) && g.toUpperCase().includes(q));
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
      } else { li.textContent = p; }
      li.addEventListener('mousedown', e => { e.preventDefault(); selectItem(p); });
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
      if (pick) selectItem(pick);
    } else if (e.key === 'Escape') {
      closeSuggest();
    }
  }

  function fillTable(tbody, rows) {
    tbody.innerHTML = '';
    if (!rows || rows.length === 0) {
      tbody.innerHTML = '<tr><td colspan="2" style="color:var(--muted)">—</td></tr>';
      return;
    }
    for (const r of rows) {
      const [name, z] = r;
      const tr = document.createElement('tr');
      const g = document.createElement('td'); g.textContent = name;
      const v = document.createElement('td'); v.textContent = (z >= 0 ? '+' : '') + z.toFixed(2);
      tr.appendChild(g); tr.appendChild(v);
      tbody.appendChild(tr);
    }
  }

  function selectItem(key) {
    const s = S(), c = cfg();
    if (!s.keys.includes(key)) return;
    SEARCH.value = key;
    closeSuggest();
    HINT.hidden = true;
    RESULTS.hidden = false;
    GENE.textContent = c.titlePrefix + key;
    SUB.textContent = 'loading…';
    fillTable(UP, []); fillTable(DN, []);
    const modeAtCall = mode;
    ensureTopk(modeAtCall).then(d => {
      if (mode !== modeAtCall) return;     // user switched modes mid-fetch
      const entry = d[key];
      if (!entry) {
        SUB.textContent = 'no data';
        fillTable(UP, []); fillTable(DN, []);
        Plotly.purge(HIST);
        return;
      }
      SUB.textContent = `top ${s.K} ${c.other} up · top ${s.K} ${c.other} down · NTC z-score across ${s.other.toLocaleString()} ${c.other}`;
      renderHist(key, entry);
      fillTable(UP, entry.up);
      fillTable(DN, entry.dn);
    }).catch(err => {
      SUB.textContent = `load failed: ${err.message || err}`;
    });
  }

  function renderHist(key, entry) {
    const s = S(), c = cfg();
    const binCenters = s.binCenters, binEdges = s.binEdges;
    if (!entry.h || !binCenters.length) { Plotly.purge(HIST); return; }
    const counts = entry.h;
    // Color bars by sign (red for positive, blue for negative, gray near zero)
    const colors = binCenters.map(c =>
      c >= 0.25 ? '#d62728' : (c <= -0.25 ? '#1f77b4' : '#bbb')
    );
    // Bucket the K up + K dn named entities into histogram bins so each bar's
    // hover tooltip lists the named entities falling in that z range (names
    // only exist for the top-K up / bottom-K down; other bins hover as counts).
    const nBins = counts.length;
    const perBin = Array.from({ length: nBins }, () => ({ up: [], dn: [] }));
    const binFor = (z) => {
      if (binEdges.length < 2) return -1;
      if (z < binEdges[0] || z > binEdges[binEdges.length - 1]) return -1;
      let lo = 0, hi = nBins - 1;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (z < binEdges[mid + 1]) hi = mid; else lo = mid + 1;
      }
      return lo;
    };
    (entry.up || []).forEach(([name, z]) => {
      const b = binFor(z); if (b >= 0) perBin[b].up.push([name, z]);
    });
    (entry.dn || []).forEach(([name, z]) => {
      const b = binFor(z); if (b >= 0) perBin[b].dn.push([name, z]);
    });
    // Two custom fields per bar: the count line + the optional name list.
    // Using two array-of-arrays avoids hovermode='x' mangling a single string.
    const customdata = perBin.map(({ up, dn }, i) => {
      const lines = [];
      up.forEach(([n, z]) => lines.push(`<span style="color:#d62728">▲ ${n} (${z.toFixed(2)})</span>`));
      dn.forEach(([n, z]) => lines.push(`<span style="color:#1f77b4">▼ ${n} (${z.toFixed(2)})</span>`));
      return [counts[i], lines.length ? '<br>' + lines.join('<br>') : ''];
    });

    Plotly.react(HIST, [{
      type: 'bar',
      x: binCenters, y: counts,
      marker: { color: colors, line: { width: 0 } },
      width: binCenters.length > 1 ? (binCenters[1] - binCenters[0]) * 0.9 : 0.2,
      customdata,
      hovertemplate: `z ≈ %{x:.2f}<br>%{customdata[0]} ${c.hoverUnit}%{customdata[1]}<extra></extra>`,
    }], {
      margin: { l: 50, r: 16, t: 22, b: 38 },
      xaxis: { title: { text: c.axisX, font: { size: 11 } }, zeroline: true, zerolinecolor: '#bbb' },
      yaxis: { title: { text: c.axisY, font: { size: 11 } } },
      bargap: 0.05,
      hovermode: 'closest',
      hoverlabel: { align: 'left', bgcolor: '#fff', bordercolor: '#ccc' },
      showlegend: false,
      plot_bgcolor: '#fff', paper_bgcolor: '#fff',
    }, {
      responsive: true, displaylogo: false,
      modeBarButtonsToRemove: ['lasso2d','select2d','autoScale2d','toggleSpikelines','zoom2d','pan2d','zoomIn2d','zoomOut2d'],
      toImageButtonOptions: { filename: `KOLF_query_${key}`, scale: 2, format: 'png' },
    });
  }

  return { init, onShow };
})();

// ============================================================================
// Clusters tab
// ============================================================================
const Clusters = (() => {
  const LIST   = document.getElementById('clist');
  const META   = document.getElementById('cmeta2');
  const HINT   = document.getElementById('cl-hint');
  const RES    = document.getElementById('cresults');
  const TITLE  = document.getElementById('c-title');
  const SUB    = document.getElementById('c-sub');
  const UP     = document.getElementById('c-up');
  const DN     = document.getElementById('c-dn');
  const NMEM   = document.getElementById('c-nmem');
  const MEMROW = document.getElementById('c-members');
  const FILTER = document.getElementById('cfilter');

  let data = null;              // { leiden: [...], hdbscan: [...], k_top }
  const ctype = 'hdbscan';      // Leiden removed from UI; HDBSCAN only
  const DEFAULT_LABEL = 'POU domain';
  let selectedId = null;
  let loaded = null;            // promise

  function ensureData() {
    if (loaded) return loaded;
    META.textContent = 'Loading cluster summaries…';
    loaded = fetch('data/clusters/summary.json?v=35').then(r => r.json()).then(d => {
      data = d;
      META.textContent = meta();
      return d;
    }).catch(err => {
      META.textContent = 'Failed to load clusters.';
      throw err;
    });
    return loaded;
  }

  function meta() {
    const nH = data.hdbscan.length;
    return `${nH} HDBSCAN clusters · top ${data.k_top} up/dn per cluster`;
  }

  function clusterLabel(c) {
    return c.label && c.label.length ? c.label : `Cluster ${c.id}`;
  }

  function renderList() {
    if (!data) return;
    // Sort: unclustered (id=-1) to the bottom; rest by id ascending.
    const arr = data[ctype].slice().sort((a, b) => {
      if (a.id === -1) return 1;
      if (b.id === -1) return -1;
      return a.id - b.id;
    });
    const q = (FILTER.value || '').trim().toLowerCase();
    LIST.innerHTML = '';
    for (const c of arr) {
      const lbl = c.id === -1 ? 'Unclustered (noise)' : clusterLabel(c);
      if (q && !(lbl.toLowerCase().includes(q) || String(c.id).includes(q))) continue;
      const row = document.createElement('div');
      row.className = 'cl-row' + (c.id === selectedId ? ' active' : '');
      row.innerHTML = `
        <span class="cid">${c.id}</span>
        <span class="clbl">${escapeHtml(lbl)}</span>
        <span class="cn">${c.n}</span>`;
      row.addEventListener('click', () => pick(c.id));
      LIST.appendChild(row);
    }
  }

  function pick(id) {
    selectedId = id;
    const c = data[ctype].find(x => x.id === id);
    LIST.querySelectorAll('.cl-row').forEach(r => r.classList.toggle(
      'active', r.querySelector('.cid').textContent === String(id)));
    if (!c) { HINT.hidden = false; RES.hidden = true; return; }
    HINT.hidden = true; RES.hidden = false;
    TITLE.textContent = c.id === -1 ? 'Unclustered (noise)' : clusterLabel(c);
    SUB.textContent = `${ctype.toUpperCase()} cluster ${c.id} · ${c.n} strong perturbations`;
    fillDegTable(UP, c.up);
    fillDegTable(DN, c.dn);
    NMEM.textContent = c.n;
    MEMROW.innerHTML = '';
    for (const name of c.members) {
      const a = document.createElement('a');
      a.href = `#mde:${encodeURIComponent(name)}`;
      a.className = 'chip';
      a.textContent = name;
      a.addEventListener('click', (e) => {
        e.preventDefault();
        jumpToMde(name);
      });
      MEMROW.appendChild(a);
    }
  }

  function fillDegTable(tbody, rows) {
    tbody.innerHTML = '';
    for (const r of rows) {
      const tr = document.createElement('tr');
      const z = typeof r.z === 'number' ? r.z.toFixed(2) : r.z;
      tr.innerHTML = `<td>${escapeHtml(r.g)}</td><td class="num">${z}</td>`;
      tbody.appendChild(tr);
    }
  }

  function jumpToMde(gene) {
    if (typeof showTab === 'function') showTab('mde');
    else location.hash = '#mde';
    if (MDE.jumpTo) MDE.jumpTo(gene);
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => (
      { '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[c]));
  }

  function attach() {
    FILTER.addEventListener('input', renderList);
  }

  function init() { attach(); }
  function onShow() {
    ensureData().then(() => {
      renderList();
      if (selectedId == null) {
        const def = data.hdbscan.find(c => (c.label || '').toLowerCase() === DEFAULT_LABEL.toLowerCase());
        if (def) pick(def.id);
      }
    });
  }
  return { init, onShow };
})();

// ============================================================================
// Gene panel tab — user picks up to 10 genes, see mean z across clusters
// ============================================================================
const Panel = (() => {
  const MAX = 10;
  const ZMAX = 0.5;                             // viz z scale ±0.5
  const DEFAULT_FEATURES = ['POU5F1','NANOG','SOX2','PRDM14','ZFP42','FZD8','SHISA2','PRDM1','OTX2'];
  const DEFAULT_PERTS    = ['POU3F2','PSMD8','POU5F1','RNF7','MARVELD1'];

  const META   = document.getElementById('pmeta');
  const HINT   = document.getElementById('pn-hint');
  const PLOT   = document.getElementById('pn-plot');
  const PERTS_PICKER = document.getElementById('pn-perts-picker');
  const GS     = {  // "gene slot" — feature gene picker
    input:  document.getElementById('psearch-gene'),
    sug:    document.getElementById('psuggest-gene'),
    chips:  document.getElementById('pchips-gene'),
  };
  const PS     = {  // "pert slot" — perturbation picker
    input:  document.getElementById('psearch-pert'),
    sug:    document.getElementById('psuggest-pert'),
    chips:  document.getElementById('pchips-pert'),
  };

  // Per-pert matrix — CHUNKED. meta.json carries perts/genes/scale/chunk_size.
  // Rows are fetched on demand via HTTP Range against chunk_{N}.bin files.
  let pmMeta = null;                            // {scale, perts, genes, chunk_size, n_perts, n_genes, row_bytes, n_chunks}
  let pmLoaded = null;
  const pertRowCache = new Map();               // pert_name -> Int8Array(n_genes)
  // Cluster-aggregated matrix (for HDBSCAN axis)
  let clMeta = null;                            // {scale, genes, leiden, hdbscan, n_genes}
  let clMatrix = null;                          // Int8Array row-major (nL+nH x n_genes)
  let clLoaded = null;
  // Cluster membership + labels (for hover enrichment)
  let clSummary = null;                         // {hdbscan: [{id,label,n,members:[...]}, ...]}
  let clSummaryLoaded = null;

  // Lookup maps (populated after pmMeta loads)
  const geneIdx  = new Map();
  const pertIdx  = new Map();
  let geneLower  = [];
  let pertLower  = [];

  let pickedGenes  = [];
  let pickedPerts  = [];
  let axis         = 'perts';                   // 'perts' | 'clusters'
  let activeSugG = -1, activeSugP = -1;
  let populated  = false;                       // defaults seeded on first show

  function ensurePertMeta() {
    if (pmLoaded) return pmLoaded;
    META.textContent = 'Loading perturbation index…';
    pmLoaded = fetch('data/panel/all/meta.json?v=35').then(r => r.json()).then(m => {
      pmMeta = m;
      pmMeta.genes.forEach((g, i) => geneIdx.set(g, i));
      pmMeta.perts.forEach((p, i) => pertIdx.set(p, i));
      geneLower = pmMeta.genes.map(g => g.toLowerCase());
      pertLower = pmMeta.perts.map(p => p.toLowerCase());
      refreshMeta();
    }).catch(err => {
      META.textContent = 'Failed to load pert index.'; throw err;
    });
    return pmLoaded;
  }

  function fetchPertRow(name) {
    if (pertRowCache.has(name)) return Promise.resolve(pertRowCache.get(name));
    const pi = pertIdx.get(name);
    if (pi == null) return Promise.resolve(null);
    const cs = pmMeta.chunk_size, rb = pmMeta.row_bytes;
    const chunkIdx = Math.floor(pi / cs);
    const offset   = (pi % cs) * rb;
    const url = `data/panel/all/chunk_${chunkIdx}.bin?v=35`;
    return fetch(url, { headers: { Range: `bytes=${offset}-${offset + rb - 1}` } })
      .then(r => {
        if (!r.ok && r.status !== 206) throw new Error(`HTTP ${r.status}`);
        return r.arrayBuffer().then(buf => ({ buf, status: r.status }));
      })
      .then(({ buf, status }) => {
        let row;
        if (status === 206 || buf.byteLength === rb) {
          row = new Int8Array(buf);
        } else {
          // Server ignored the Range header and returned the full chunk —
          // slice out just the requested row.
          row = new Int8Array(buf, offset, rb);
        }
        pertRowCache.set(name, row);
        return row;
      });
  }

  function prefetchPickedPertRows() {
    const missing = pickedPerts.filter(p => !pertRowCache.has(p));
    if (!missing.length) return Promise.resolve();
    META.textContent = `Fetching z for ${missing.length} perturbation${missing.length === 1 ? '' : 's'}…`;
    return Promise.all(missing.map(fetchPertRow)).then(() => refreshMeta());
  }

  function ensureClusterMeans() {
    if (clLoaded) return clLoaded;
    clLoaded = Promise.all([
      fetch('data/clusters/means_meta.json?v=35').then(r => r.json()),
      fetch('data/clusters/means.bin?v=35').then(r => r.arrayBuffer()),
    ]).then(([m, buf]) => {
      clMeta = m;
      clMatrix = new Int8Array(buf);
    }).catch(err => {
      META.textContent = 'Failed to load cluster means.'; throw err;
    });
    return clLoaded;
  }

  function ensureClusterSummary() {
    if (clSummaryLoaded) return clSummaryLoaded;
    clSummaryLoaded = fetch('data/clusters/summary.json?v=35').then(r => r.json()).then(s => {
      clSummary = s;
    }).catch(err => { console.warn('Cluster summary load failed', err); });
    return clSummaryLoaded;
  }

  function refreshMeta() {
    if (!pmMeta) return;
    META.textContent = `${pmMeta.genes.length.toLocaleString()} genes · ${pmMeta.perts.length.toLocaleString()} perturbations · z-scale ±${ZMAX}`;
  }

  function clusterRow(hi) {                     // hi = HDBSCAN cluster index (0..nH-1)
    const nL = clMeta.leiden.length;
    const ri = nL + hi;
    return clMatrix.subarray(ri * clMeta.n_genes, (ri + 1) * clMeta.n_genes);
  }

  function memberSummaryFor(cid) {
    if (!clSummary) return '';
    const c = (clSummary.hdbscan || []).find(x => x.id === cid);
    if (!c || !c.members) return '';
    const MAX = 12;
    const head = c.members.slice(0, MAX).join(', ');
    const tail = c.members.length > MAX ? ` +${c.members.length - MAX} more` : '';
    return `<br><span style="color:#555;font-size:11px">${head}${tail}</span>`;
  }

  // ---------- Rendering ----------
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => (
      { '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[c]));
  }

  function renderChips() {
    const fill = (row, list, onRemove) => {
      row.innerHTML = '';
      list.forEach(name => {
        const c = document.createElement('span');
        c.className = 'chip';
        c.innerHTML = `${escapeHtml(name)}<span class="x" title="Remove">×</span>`;
        c.querySelector('.x').addEventListener('click', () => onRemove(name));
        row.appendChild(c);
      });
    };
    fill(GS.chips, pickedGenes, name => { pickedGenes = pickedGenes.filter(g => g !== name); renderChips(); render(); });
    fill(PS.chips, pickedPerts, name => { pickedPerts = pickedPerts.filter(p => p !== name); renderChips(); render(); });
  }

  function render() {
    if (!pmMeta) return;

    const rows = pickedGenes.slice();           // row keys (feature gene names)
    if (!rows.length) return showHint();

    if (axis === 'perts') {
      if (!pickedPerts.length) return showHint();
      // Fetch missing rows first, then re-enter render()
      const missing = pickedPerts.filter(p => !pertRowCache.has(p));
      if (missing.length) {
        prefetchPickedPertRows().then(() => render());
        showHint('Fetching per-perturbation z rows…');
        return;
      }
      renderHeatmap(
        rows,
        pickedPerts.slice(),                                         // x labels
        pickedPerts.map(p => `Perturbed gene: ${p}`),                // per-col hover labels
        pickedPerts.map(p => pertRowCache.get(p)),                   // per-col z vectors
        pmMeta.scale,
      );
    } else {
      if (!clMeta || !clMatrix) { ensureClusterMeans().then(() => { ensureClusterSummary().then(render); }); showHint('Loading HDBSCAN means…'); return; }
      if (!clSummary)            { ensureClusterSummary().then(render); }   // enrich hover when members land
      const hdb = clMeta.hdbscan.slice().sort((a, b) => {
        if (a.id === -1) return 1; if (b.id === -1) return -1; return a.id - b.id;
      });
      const xLabels = hdb.map(c => c.id === -1 ? 'noise' : String(c.id));
      // Hover content per column: cluster name + strong-pert count + member list.
      // Kept out of the tick label so the x-axis stays legible.
      const colLabels = hdb.map(c => {
        const head = c.id === -1 ? 'Unclustered (noise)'
                   : (c.label ? `Cluster ${c.id} · ${c.label}` : `Cluster ${c.id}`);
        const n    = `<br><span style="color:#555;font-size:11px">${c.n} strong perts</span>`;
        const mem  = memberSummaryFor(c.id);
        return head + n + mem;
      });
      const colZ = hdb.map(c => clusterRow(clMeta.hdbscan.findIndex(x => x.id === c.id)));
      renderHeatmap(rows, xLabels, colLabels, colZ, clMeta.scale);
    }
  }

  function renderHeatmap(rows, xLabels, colLabels, colZ, scale) {
    HINT.hidden = true; PLOT.hidden = false;
    const z = rows.map(g => {
      const gi = geneIdx.get(g);
      return colZ.map(row => row[gi] * scale);
    });
    const cdata = rows.map(() => colLabels);
    const trace = {
      type: 'heatmap',
      x: xLabels, y: rows, z,
      customdata: cdata,
      colorscale: 'RdBu', reversescale: true, zmin: -ZMAX, zmax: ZMAX,
      hovertemplate: '<b>%{y}</b> × %{customdata}<br>mean z = %{z:.3f}<extra></extra>',
      colorbar: { title: { text: 'z', font: { size: 10 } }, thickness: 8, len: 0.6,
                  tickfont: { size: 10 } },
      xgap: 1, ygap: 1,
    };
    const height = Math.max(260, 44 * rows.length + 110);
    PLOT.style.height = height + 'px';
    const layout = {
      margin: { l: 100, r: 60, t: 16, b: 80 },
      xaxis: {
        title: { text: axis === 'perts' ? 'Perturbed gene' : 'HDBSCAN cluster id',
                 font: { size: 11 } },
        type: 'category', automargin: true,
        tickfont: { size: 11 }, side: 'bottom',
      },
      yaxis: {
        title: { text: 'Feature gene', font: { size: 11 } },
        type: 'category', autorange: 'reversed',
        tickfont: { size: 11 }, automargin: true,
      },
      plot_bgcolor: '#fff', paper_bgcolor: '#fff', showlegend: false,
      hoverlabel: { align: 'left' },
    };
    Plotly.react(PLOT, [trace], layout, {
      responsive: true, displaylogo: false,
      modeBarButtonsToRemove: ['lasso2d','select2d','autoScale2d','toggleSpikelines','zoom2d','pan2d'],
      toImageButtonOptions: { filename: `KOLF_panel_${axis}`, scale: 2, format: 'png' },
    });
  }

  function showHint(msg) {
    PLOT.hidden = true; HINT.hidden = false;
    if (msg) HINT.innerHTML = `<p>${escapeHtml(msg)}</p>`;
    Plotly.purge(PLOT);
  }

  // ---------- Suggest (reads module-level geneLower/pertLower at call time) ----------
  function namesFor(kind) { return kind === 'gene' ? pmMeta.genes : pmMeta.perts; }
  function lowerFor(kind) { return kind === 'gene' ? geneLower   : pertLower; }
  function getActive(kind) { return kind === 'gene' ? activeSugG : activeSugP; }
  function setActive(kind, v) { if (kind === 'gene') activeSugG = v; else activeSugP = v; }

  function updateSuggest(slot, kind) {
    if (!pmMeta) return;
    const q = (slot.input.value || '').trim().toLowerCase();
    if (!q) { slot.sug.hidden = true; setActive(kind, -1); return; }
    const lower = lowerFor(kind), names = namesFor(kind);
    const starts = [], contains = [];
    for (let i = 0; i < lower.length; i++) {
      const s = lower[i];
      if (s === q) { starts.unshift(i); continue; }
      if (s.startsWith(q)) { starts.push(i); if (starts.length > 30) break; continue; }
      if (s.includes(q) && contains.length < 30) contains.push(i);
    }
    const hits = starts.concat(contains).slice(0, 20).map(i => names[i]);
    if (!hits.length) { slot.sug.hidden = true; return; }
    slot.sug.innerHTML = '';
    hits.forEach((name, i) => {
      const li = document.createElement('li');
      li.textContent = name;
      li.dataset.item = name;
      if (i === 0) { li.classList.add('active'); setActive(kind, 0); }
      li.addEventListener('mousedown', (e) => { e.preventDefault(); pickSuggest(slot, kind, name); });
      slot.sug.appendChild(li);
    });
    slot.sug.hidden = false;
  }

  function pickSuggest(slot, kind, name) {
    addPicked(kind, name);
    slot.input.value = '';
    slot.sug.hidden = true; setActive(kind, -1);
  }

  function addPicked(kind, name) {
    if (!name || !pmMeta) return;
    name = name.trim();
    const canon = namesFor(kind).find(x => x.toUpperCase() === name.toUpperCase());
    if (!canon) return;
    const arr = kind === 'gene' ? pickedGenes : pickedPerts;
    if (arr.includes(canon)) return;
    if (arr.length >= MAX) { META.textContent = `Slot full (${MAX} max).`; return; }
    arr.push(canon);
    renderChips(); render(); refreshMeta();
  }

  function wireSlot(slot, kind) {
    slot.input.addEventListener('input',  () => updateSuggest(slot, kind));
    slot.input.addEventListener('focus',  () => updateSuggest(slot, kind));
    slot.input.addEventListener('blur',   () => setTimeout(() => { slot.sug.hidden = true; }, 120));
    slot.input.addEventListener('keydown', (e) => {
      const items = slot.sug.querySelectorAll('li');
      const cur = getActive(kind);
      if (e.key === 'Enter') {
        e.preventDefault();
        if (items.length && cur >= 0) pickSuggest(slot, kind, items[cur].dataset.item);
        else pickSuggest(slot, kind, slot.input.value);
      } else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        if (!items.length) return;
        e.preventDefault();
        items[cur >= 0 ? cur : 0].classList.remove('active');
        const next = (cur + (e.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length;
        items[next].classList.add('active');
        setActive(kind, next);
      } else if (e.key === 'Escape') {
        slot.sug.hidden = true; setActive(kind, -1);
      }
    });
  }

  function wireAxisToggle() {
    document.querySelectorAll('#view-panel .seg').forEach(btn => {
      btn.addEventListener('click', () => {
        const newAxis = btn.dataset.paxis;
        if (newAxis === axis) return;
        document.querySelectorAll('#view-panel .seg').forEach(b => {
          const on = (b === btn);
          b.classList.toggle('active', on);
          b.setAttribute('aria-checked', on);
        });
        axis = newAxis;
        PERTS_PICKER.style.opacity = axis === 'clusters' ? '0.45' : '1';
        PERTS_PICKER.style.pointerEvents = axis === 'clusters' ? 'none' : 'auto';
        if (axis === 'clusters') ensureClusterMeans().then(render);
        else render();
      });
    });
  }

  function populateDefaults() {
    if (populated || !pmMeta) return;
    populated = true;
    DEFAULT_FEATURES.forEach(g => { if (pmMeta.genes.includes(g) && !pickedGenes.includes(g)) pickedGenes.push(g); });
    DEFAULT_PERTS   .forEach(p => { if (pmMeta.perts.includes(p) && !pickedPerts.includes(p)) pickedPerts.push(p); });
    renderChips(); render();
  }

  function init() {
    wireSlot(GS, 'gene');
    wireSlot(PS, 'pert');
    wireAxisToggle();
  }

  function onShow() {
    ensurePertMeta().then(() => {
      populateDefaults();
      if (PLOT.offsetWidth) Plotly.Plots.resize(PLOT);
    });
  }
  return { init, onShow };
})();

// ============================================================================
// Boot
// ============================================================================
MDE.init();
Clustermap.init();
GeneClustermap.init();
GeneQuery.init();
Clusters.init();
Panel.init();

// Clustermap subview toggle (pert <-> gene) within the Clustermaps tab
(function wireClustermapToggle() {
  const pertView = document.getElementById('cmap-pert-view');
  const geneView = document.getElementById('cmap-gene-view');
  document.querySelectorAll('#view-clustermap .cmap-mode-row .seg').forEach(btn => {
    btn.addEventListener('click', () => {
      const newMode = btn.dataset.cmode;
      if (newMode === cmapMode) return;
      document.querySelectorAll('#view-clustermap .cmap-mode-row .seg').forEach(b => {
        const on = (b === btn);
        b.classList.toggle('active', on);
        b.setAttribute('aria-checked', on);
      });
      cmapMode = newMode;
      pertView.hidden = (cmapMode !== 'pert');
      geneView.hidden = (cmapMode !== 'gene');
      onCmapShow();                             // force resize of freshly visible plot
    });
  });
})();

// Initial route
const initial = location.hash.replace('#', '');
showTab(TABS.includes(initial) ? initial : 'home');

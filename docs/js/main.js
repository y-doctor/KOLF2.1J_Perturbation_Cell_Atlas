/* KOLF2.1J Perturbation Atlas viewer
   Tabs: MDE (#mde), Pert clustermap (#clustermap),
         Gene clustermap (#genemap), Gene query (#genequery),
         Clusters (#clusters).
*/

// ============================================================================
// Router
// ============================================================================
const TABS = ['mde', 'clustermap', 'genemap', 'genequery', 'clusters', 'panel'];
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
  if (name === 'genemap')    GeneClustermap.onShow();
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
    up:      document.getElementById('p-up').querySelector('tbody'),
    dn:      document.getElementById('p-dn').querySelector('tbody'),
    sim:     document.getElementById('p-sim'),
    nbr:     document.getElementById('p-nbr'),
  };

  let fitnessCache = null;
  let signaturesCache = null;
  function ensureFitness()    { return fitnessCache    || (fitnessCache    = fetch('data/mde/fitness.json?v=23').then(r => r.json())); }
  function ensureSignatures() { return signaturesCache || (signaturesCache = fetch('data/mde/signatures.json?v=23').then(r => r.json())); }

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
  function renderSimList(ol, names) {
    ol.innerHTML = '';
    if (!names || names.length === 0) {
      const li = document.createElement('li');
      li.innerHTML = '<span class="g" style="color:var(--muted)">—</span>';
      ol.appendChild(li);
      return;
    }
    for (const name of names) {
      const i = indexByGene.get(name);
      const li = document.createElement('li');
      const g = document.createElement('span');
      g.className = 'g';
      g.textContent = name;
      if (i != null) g.addEventListener('click', () => focusGene(i, { zoom: true }));
      li.appendChild(g);
      ol.appendChild(li);
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
    renderDegsTable(P.up, null);
    renderDegsTable(P.dn, null);
    renderSimList(P.sim, null);
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

    ensureSignatures().then(sigs => {
      const s = sigs[gene];
      if (!s) { renderDegsTable(P.up, []); renderDegsTable(P.dn, []); renderSimList(P.sim, []); return; }
      renderDegsTable(P.up, s.up);
      renderDegsTable(P.dn, s.dn);
      renderSimList(P.sim, s.neighbors);
    }).catch(e => {
      console.error('signatures load failed', e);
      renderDegsTable(P.up, []); renderDegsTable(P.dn, []); renderSimList(P.sim, []);
    });
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
    return fetch('data/mde.json?v=23').then(r => r.json()).then(payload => {
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
  metaPath: 'data/clustermap/meta.json?v=23',
  mainPath: 'data/clustermap/corr_int8.bin?v=23',
  sidePath: 'data/clustermap/corr_side_int8.bin?v=23',
  mainZmin: -0.2, mainZmax: 0.2,
  sideZmin: -0.5, sideZmax: 0.5,
});

const GeneClustermap = makeClustermap({
  label:    'gene clustermap',
  unit:     'genes',
  mainId:   'gmap-main', sideId:   'gmap-side',
  searchId: 'gsearch',   suggestId:'gsuggest',
  hintId:   'ghint',     metaId:   'gmeta',     resetId: 'greset',
  metaPath: 'data/genemap/meta.json?v=23',
  mainPath: 'data/genemap/gene_corr_int8.bin?v=23',
  sidePath: 'data/genemap/gene_corr_side_int8.bin?v=23',
  mainZmin: -0.2, mainZmax: 0.2,
  sideZmin: -0.5, sideZmax: 0.5,
});

// ============================================================================
// Gene query tab — for any expressed gene, show top 25 perts with highest
// and lowest mean NTC z-score (post-basic-QC universe of ~11,687 perts).
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

  let genes = [];
  let nPerts = 0, K = 25;
  let binCenters = [];
  let binEdges = [];
  let topk = null;          // lazy-loaded big payload
  let topkLoading = null;
  let activeSugg = -1;
  let pendingPick = null;   // gene to render after topk lands

  function init() {
    return fetch('data/genes/index.json?v=23').then(r => r.json()).then(idx => {
      genes      = idx.genes || [];
      nPerts     = idx.n_perts;
      K          = idx.k || 25;
      binCenters = idx.bin_centers || [];
      binEdges   = idx.bin_edges   || [];
      META.textContent = `${genes.length.toLocaleString()} expressed genes · ${nPerts.toLocaleString()} perturbations`;
      attachEvents();
    }).catch(err => {
      META.textContent = `Failed to load gene index: ${err.message || err}`;
    });
  }
  function onShow() { if (SEARCH) SEARCH.focus(); }

  function ensureTopk() {
    if (topk) return Promise.resolve(topk);
    if (topkLoading) return topkLoading;
    META.textContent = 'Loading per-gene query data (~17 MB)…';
    topkLoading = fetch('data/genes/topk.json?v=23').then(r => r.json()).then(data => {
      topk = data;
      META.textContent = `${genes.length.toLocaleString()} genes · ${nPerts.toLocaleString()} perts`;
      return topk;
    });
    return topkLoading;
  }

  function attachEvents() {
    SEARCH.addEventListener('input',   () => updateSuggest(SEARCH.value));
    SEARCH.addEventListener('focus',   () => updateSuggest(SEARCH.value));
    SEARCH.addEventListener('blur',    () => setTimeout(closeSuggest, 120));
    SEARCH.addEventListener('keydown', onSearchKey);
  }

  function fuzzy(q) {
    q = q.trim().toUpperCase();
    if (!q) return genes.slice(0, 12);
    const eq  = genes.filter(g => g.toUpperCase() === q);
    const pre = genes.filter(g => g.toUpperCase().startsWith(q) && g.toUpperCase() !== q);
    const sub = genes.filter(g => !g.toUpperCase().startsWith(q) && g.toUpperCase().includes(q));
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
      li.addEventListener('mousedown', e => { e.preventDefault(); selectGene(p); });
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
      if (pick) selectGene(pick);
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

  function selectGene(g) {
    if (!genes.includes(g)) return;
    SEARCH.value = g;
    closeSuggest();
    HINT.hidden = true;
    RESULTS.hidden = false;
    GENE.textContent = g;
    SUB.textContent = 'loading…';
    fillTable(UP, []); fillTable(DN, []);
    ensureTopk().then(d => {
      const entry = d[g];
      if (!entry) {
        SUB.textContent = 'no data';
        fillTable(UP, []); fillTable(DN, []);
        Plotly.purge(HIST);
        return;
      }
      SUB.textContent = `top ${K} perts up · top ${K} perts down · NTC z-score across ${nPerts.toLocaleString()} perts`;
      renderHist(g, entry);
      fillTable(UP, entry.up);
      fillTable(DN, entry.dn);
    }).catch(err => {
      SUB.textContent = `load failed: ${err.message || err}`;
    });
  }

  function renderHist(gene, entry) {
    if (!entry.h || !binCenters.length) { Plotly.purge(HIST); return; }
    const counts = entry.h;
    // Color bars by sign (red for positive, blue for negative, gray near zero)
    const colors = binCenters.map(c =>
      c >= 0.25 ? '#d62728' : (c <= -0.25 ? '#1f77b4' : '#bbb')
    );
    // Threshold lines at the 25th highest / 25th lowest pert z values
    const upZ = entry.up && entry.up.length ? entry.up[entry.up.length - 1][1] : null;
    const dnZ = entry.dn && entry.dn.length ? entry.dn[entry.dn.length - 1][1] : null;
    const shapes = [];
    if (upZ != null) shapes.push({
      type: 'line', xref: 'x', yref: 'paper', x0: upZ, x1: upZ, y0: 0, y1: 1,
      line: { color: '#d62728', width: 1, dash: 'dot' },
    });
    if (dnZ != null) shapes.push({
      type: 'line', xref: 'x', yref: 'paper', x0: dnZ, x1: dnZ, y0: 0, y1: 1,
      line: { color: '#1f77b4', width: 1, dash: 'dot' },
    });
    // Bucket the 25 up + 25 dn perts into histogram bins so each bar's hover
    // tooltip lists the named perts falling in that z range.
    const nBins = counts.length;
    const perBin = Array.from({ length: nBins }, () => ({ up: [], dn: [] }));
    const binFor = (z) => {
      // binEdges has length nBins+1; find i such that edges[i] <= z < edges[i+1]
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
    const customdata = perBin.map(({ up, dn }) => {
      const lines = [];
      up.forEach(([n, z]) => lines.push(`<span style="color:#d62728">▲ ${n} (${z.toFixed(2)})</span>`));
      dn.forEach(([n, z]) => lines.push(`<span style="color:#1f77b4">▼ ${n} (${z.toFixed(2)})</span>`));
      return lines.length ? '<br>' + lines.join('<br>') : '';
    });

    const annotations = [];
    if (upZ != null) annotations.push({
      x: 1, y: 0.98, xref: 'paper', yref: 'paper',
      text: `top 25 ≥ ${upZ.toFixed(2)}`,
      showarrow: false, font: { size: 10, color: '#d62728' },
      xanchor: 'right', yanchor: 'top',
    });
    if (dnZ != null) annotations.push({
      x: 0, y: 0.98, xref: 'paper', yref: 'paper',
      text: `bottom 25 ≤ ${dnZ.toFixed(2)}`,
      showarrow: false, font: { size: 10, color: '#1f77b4' },
      xanchor: 'left', yanchor: 'top',
    });
    Plotly.react(HIST, [{
      type: 'bar',
      x: binCenters, y: counts,
      marker: { color: colors, line: { width: 0 } },
      width: binCenters.length > 1 ? (binCenters[1] - binCenters[0]) * 0.9 : 0.2,
      customdata,
      hovertemplate: 'z ≈ %{x:.2f}<br>%{y} perts%{customdata}<extra></extra>',
    }], {
      margin: { l: 50, r: 16, t: 22, b: 38 },
      xaxis: { title: { text: 'NTC z-score', font: { size: 11 } }, zeroline: true, zerolinecolor: '#bbb' },
      yaxis: { title: { text: '# perturbations', font: { size: 11 } } },
      bargap: 0.05,
      hovermode: 'x',
      showlegend: false,
      plot_bgcolor: '#fff', paper_bgcolor: '#fff',
      shapes, annotations,
    }, {
      responsive: true, displaylogo: false,
      modeBarButtonsToRemove: ['lasso2d','select2d','autoScale2d','toggleSpikelines','zoom2d','pan2d','zoomIn2d','zoomOut2d'],
      toImageButtonOptions: { filename: `KOLF_query_${gene}`, scale: 2, format: 'png' },
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
  let ctype = 'leiden';
  let selectedId = null;
  let loaded = null;            // promise

  function ensureData() {
    if (loaded) return loaded;
    META.textContent = 'Loading cluster summaries…';
    loaded = fetch('data/clusters/summary.json?v=23').then(r => r.json()).then(d => {
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
    const nL = data.leiden.length, nH = data.hdbscan.length;
    return `Leiden ${nL} · HDBSCAN ${nH} · top ${data.k_top} up/dn per cluster`;
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
    document.querySelectorAll('#view-clusters .seg').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('#view-clusters .seg').forEach(b => {
          const on = (b === btn);
          b.classList.toggle('active', on);
          b.setAttribute('aria-checked', on);
        });
        ctype = btn.dataset.ctype;
        selectedId = null;
        HINT.hidden = false; RES.hidden = true;
        renderList();
      });
    });
    FILTER.addEventListener('input', renderList);
  }

  function init() { attach(); }
  function onShow() { ensureData().then(renderList); }
  return { init, onShow };
})();

// ============================================================================
// Gene panel tab — user picks up to 10 genes, see mean z across clusters
// ============================================================================
const Panel = (() => {
  const MAX_GENES = 10;
  const SEARCH = document.getElementById('psearch');
  const SUG    = document.getElementById('psuggest');
  const CHIPS  = document.getElementById('pchips');
  const CLEAR  = document.getElementById('pclear');
  const META   = document.getElementById('pmeta');
  const HINT   = document.getElementById('pn-hint');
  const PLOT   = document.getElementById('pn-plot');

  let meta = null;                  // { scale, genes, leiden:[{id,label,n}], hdbscan:[...] }
  let matrix = null;                // Int8Array, row-major (n_clusters x n_genes)
  let nL = 0, nH = 0;               // rows per type (leiden then hdbscan)
  let geneIdx = new Map();          // gene -> row index
  let geneLower = [];               // lowercase gene list for search
  let picked = [];                  // ['GENE1', ...]
  let ptype = 'leiden';
  let loaded = null;
  let activeSugg = -1;

  function ensure() {
    if (loaded) return loaded;
    META.textContent = 'Loading cluster means…';
    loaded = Promise.all([
      fetch('data/clusters/means_meta.json?v=23').then(r => r.json()),
      fetch('data/clusters/means.bin?v=23').then(r => r.arrayBuffer()),
    ]).then(([m, buf]) => {
      meta = m;
      nL = meta.leiden.length;
      nH = meta.hdbscan.length;
      matrix = new Int8Array(buf);
      meta.genes.forEach((g, i) => geneIdx.set(g, i));
      geneLower = meta.genes.map(g => g.toLowerCase());
      META.textContent = `${meta.genes.length.toLocaleString()} genes · Leiden ${nL} · HDBSCAN ${nH}`;
    }).catch(err => {
      META.textContent = 'Failed to load.';
      throw err;
    });
    return loaded;
  }

  function clusterRow(ptypeLocal, i) {
    // Returns an array of z values for cluster row (leiden i or hdbscan i)
    const rowStart = (ptypeLocal === 'leiden' ? i : nL + i) * meta.n_genes;
    return matrix.subarray(rowStart, rowStart + meta.n_genes);
  }

  function render() {
    if (!meta) return;
    if (!picked.length) {
      HINT.hidden = false; PLOT.hidden = true; CLEAR.hidden = true; Plotly.purge(PLOT); return;
    }
    HINT.hidden = true; PLOT.hidden = false; CLEAR.hidden = false;

    const clusters = meta[ptype];   // [{id,label,n}, ...] in stored order
    // Build z matrix rows = genes (picked order), cols = clusters (stored order)
    const z = picked.map(name => {
      const gi = geneIdx.get(name);
      const out = new Array(clusters.length);
      for (let ci = 0; ci < clusters.length; ci++) {
        const cellRow = clusterRow(ptype, ci);
        out[ci] = cellRow[gi] * meta.scale;
      }
      return out;
    });
    const xLabels = clusters.map(c => c.id === -1 ? 'noise' : String(c.id));
    const clusterLabel = clusters.map(c => c.label && c.label.length
      ? `Cluster ${c.id} · ${c.label}`
      : (c.id === -1 ? 'Unclustered' : `Cluster ${c.id}`));
    // customdata: matching shape of z — repeat the per-column label row per gene
    const cdata = picked.map(() => clusterLabel);
    // Color scale RdBu diverging
    const zmax = 2, zmin = -zmax;
    const trace = {
      type: 'heatmap',
      x: xLabels, y: picked, z,
      customdata: cdata,
      colorscale: 'RdBu', reversescale: true, zmin, zmax,
      hovertemplate: '<b>%{y}</b> × %{customdata}<br>mean z = %{z:.2f}<extra></extra>',
      colorbar: { title: { text: 'mean z', font: { size: 10 } }, thickness: 8, len: 0.6,
                  tickfont: { size: 10 } },
    };
    const nCols = clusters.length;
    // Generous height per gene, width follows container
    const height = Math.max(220, 46 * picked.length + 80);
    PLOT.style.height = height + 'px';
    const layout = {
      margin: { l: 88, r: 60, t: 16, b: 54 },
      xaxis: {
        title: { text: `${ptype.toUpperCase()} cluster id`, font: { size: 11 } },
        type: 'category', automargin: true,
        tickfont: { size: 10 }, side: 'bottom',
        dtick: nCols > 30 ? 2 : 1,
      },
      yaxis: {
        type: 'category', autorange: 'reversed',
        tickfont: { size: 11 }, automargin: true,
      },
      plot_bgcolor: '#fff', paper_bgcolor: '#fff', showlegend: false,
    };
    Plotly.react(PLOT, [trace], layout, {
      responsive: true, displaylogo: false,
      modeBarButtonsToRemove: ['lasso2d','select2d','autoScale2d','toggleSpikelines','zoom2d','pan2d'],
      toImageButtonOptions: { filename: `KOLF_panel_${ptype}`, scale: 2, format: 'png' },
    });
  }

  function renderChips() {
    CHIPS.innerHTML = '';
    picked.forEach(name => {
      const c = document.createElement('span');
      c.className = 'chip';
      c.innerHTML = `${escapeHtml(name)}<span class="x" title="Remove">×</span>`;
      c.querySelector('.x').addEventListener('click', () => {
        picked = picked.filter(g => g !== name);
        renderChips(); render();
      });
      CHIPS.appendChild(c);
    });
  }

  function addGene(name) {
    if (!name || !geneIdx.has(name)) return;
    if (picked.includes(name)) return;
    if (picked.length >= MAX_GENES) {
      META.textContent = `Panel is full (${MAX_GENES} max). Remove a gene to add another.`;
      return;
    }
    picked.push(name);
    renderChips(); render();
    META.textContent = `${meta.genes.length.toLocaleString()} genes · Leiden ${nL} · HDBSCAN ${nH}`;
  }

  function updateSuggest(raw) {
    if (!meta) { SUG.hidden = true; return; }
    const q = (raw || '').trim().toLowerCase();
    if (!q) { SUG.hidden = true; activeSugg = -1; return; }
    const starts = [], contains = [];
    for (let i = 0; i < geneLower.length; i++) {
      const gl = geneLower[i];
      if (gl === q) { starts.unshift(i); continue; }
      if (gl.startsWith(q)) { starts.push(i); if (starts.length > 30) break; continue; }
      if (gl.includes(q) && contains.length < 30) contains.push(i);
    }
    const hits = starts.concat(contains).slice(0, 20).map(i => meta.genes[i]);
    if (!hits.length) { SUG.hidden = true; return; }
    SUG.innerHTML = '';
    hits.forEach((g, i) => {
      const li = document.createElement('li');
      li.textContent = g;
      li.dataset.gene = g;
      if (i === 0) { li.classList.add('active'); activeSugg = 0; }
      li.addEventListener('mousedown', (e) => { e.preventDefault(); pickSuggest(g); });
      SUG.appendChild(li);
    });
    SUG.hidden = false;
  }

  function pickSuggest(g) {
    addGene(g);
    SEARCH.value = '';
    SUG.hidden = true; activeSugg = -1;
  }

  function onKey(e) {
    const items = SUG.querySelectorAll('li');
    if (e.key === 'Enter') {
      e.preventDefault();
      if (items.length && activeSugg >= 0) pickSuggest(items[activeSugg].dataset.gene);
      else addGene(SEARCH.value.trim().toUpperCase());
      SEARCH.value = ''; SUG.hidden = true;
    } else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      if (!items.length) return;
      e.preventDefault();
      items[activeSugg >= 0 ? activeSugg : 0].classList.remove('active');
      activeSugg = (activeSugg + (e.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length;
      items[activeSugg].classList.add('active');
    } else if (e.key === 'Escape') {
      SUG.hidden = true; activeSugg = -1;
    }
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => (
      { '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[c]));
  }

  function attach() {
    SEARCH.addEventListener('input',  () => updateSuggest(SEARCH.value));
    SEARCH.addEventListener('focus',  () => updateSuggest(SEARCH.value));
    SEARCH.addEventListener('blur',   () => setTimeout(() => { SUG.hidden = true; }, 120));
    SEARCH.addEventListener('keydown', onKey);
    CLEAR.addEventListener('click', () => { picked = []; renderChips(); render(); });
    document.querySelectorAll('#view-panel .seg').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('#view-panel .seg').forEach(b => {
          const on = (b === btn);
          b.classList.toggle('active', on);
          b.setAttribute('aria-checked', on);
        });
        ptype = btn.dataset.ptype;
        render();
      });
    });
  }

  function init() { attach(); }
  function onShow() {
    ensure().then(() => { render(); if (PLOT.offsetWidth) Plotly.Plots.resize(PLOT); });
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

// Initial route
const initial = location.hash.replace('#', '');
showTab(TABS.includes(initial) ? initial : 'mde');

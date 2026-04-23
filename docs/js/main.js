/* KOLF2.1J Perturbation Atlas viewer
   Three tabs: MDE (#mde), Volcano (#volcano), Clusters (#clusters).
*/

// ============================================================================
// Router
// ============================================================================
const TABS = ['mde', 'volcano', 'clusters'];
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
  if (name === 'mde')      MDE.onShow();
  if (name === 'volcano')  Volcano.onShow();
  if (name === 'clusters') Clusters.onShow();
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
    return fetch('data/mde.json?v=7').then(r => r.json()).then(payload => {
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
// Volcano tab
// ============================================================================
const Volcano = (() => {
  const VPLOT   = document.getElementById('vplot');
  const VHINT   = document.getElementById('vhint');
  const VSEARCH = document.getElementById('vsearch');
  const VSUG    = document.getElementById('vsuggest');
  const VMETA   = document.getElementById('vmeta');

  let perts = [];
  let cache = new Map();         // pert -> rows
  let currentPert = null;
  let plotted = false;
  let activeSugg = -1;

  const layout = {
    margin: { l: 50, r: 12, t: 36, b: 40 },
    xaxis: { title: 'log2 fold change', zeroline: true, zerolinecolor: '#bbb' },
    yaxis: { title: '−log10 (adj. p-value)' },
    hovermode: 'closest', showlegend: false, dragmode: 'pan',
    plot_bgcolor: '#fff', paper_bgcolor: '#fff', annotations: [],
    title: { text: '', font: { size: 14 } },
  };
  const config = {
    responsive: true, displaylogo: false, scrollZoom: true,
    modeBarButtonsToRemove: ['lasso2d','select2d','autoScale2d','toggleSpikelines'],
    toImageButtonOptions: { filename: 'KOLF_volcano', scale: 2, format: 'png' },
  };

  const SIG_ADJ = 0.05;
  const SIG_LFC = 1.0;     // |L2FC| threshold for "hit"
  const COL_UP  = '#d62728';
  const COL_DN  = '#1f77b4';
  const COL_NS  = '#bbb';

  function init() {
    return fetch('data/volcano/perts.json?v=7').then(r => r.json()).then(idx => {
      perts = idx.perts || [];
      VMETA.textContent = `${perts.length} perturbations`;
      attachEvents();
    }).catch(err => {
      VHINT.textContent = `Failed to load volcano index: ${err}`;
    });
  }

  function onShow() {
    if (plotted) Plotly.Plots.resize(VPLOT);
    VSEARCH.focus();
  }

  function attachEvents() {
    VSEARCH.addEventListener('input',  () => updateSuggest(VSEARCH.value));
    VSEARCH.addEventListener('focus',  () => updateSuggest(VSEARCH.value));
    VSEARCH.addEventListener('blur',   () => setTimeout(closeSuggest, 120));
    VSEARCH.addEventListener('keydown', onSearchKey);
  }

  function fuzzy(query) {
    const q = query.trim().toUpperCase();
    if (!q) return perts.slice(0, 12);
    const eq = perts.filter(p => p.toUpperCase() === q);
    const pre = perts.filter(p => p.toUpperCase().startsWith(q) && p.toUpperCase() !== q);
    const sub = perts.filter(p => !p.toUpperCase().startsWith(q) && p.toUpperCase().includes(q));
    return [...eq, ...pre, ...sub].slice(0, 12);
  }

  function updateSuggest(q) {
    const list = fuzzy(q);
    if (!list.length) { closeSuggest(); return; }
    VSUG.innerHTML = '';
    list.forEach((p, i) => {
      const li = document.createElement('li');
      // highlight the matched substring
      const idx = q ? p.toUpperCase().indexOf(q.trim().toUpperCase()) : -1;
      if (idx >= 0 && q.trim()) {
        const before = p.slice(0, idx);
        const m = p.slice(idx, idx + q.trim().length);
        const after = p.slice(idx + q.trim().length);
        li.innerHTML = `${before}<mark>${m}</mark>${after}`;
      } else {
        li.textContent = p;
      }
      li.addEventListener('mousedown', e => { e.preventDefault(); selectPert(p); });
      VSUG.appendChild(li);
    });
    activeSugg = -1;
    VSUG.hidden = false;
  }
  function closeSuggest() { VSUG.hidden = true; activeSugg = -1; }

  function onSearchKey(e) {
    if (VSUG.hidden) return;
    const items = VSUG.querySelectorAll('li');
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
    if (!perts.includes(p)) return;
    VSEARCH.value = p;
    closeSuggest();
    if (p === currentPert) return;
    currentPert = p;
    loadAndRender(p);
  }

  function loadAndRender(p) {
    VHINT.textContent = `Loading ${p}…`;
    VHINT.hidden = false;
    const fetchRows = cache.has(p)
      ? Promise.resolve(cache.get(p))
      : fetch(`data/volcano/${encodeURIComponent(p)}.json?v=7`).then(r => r.json())
          .then(rows => { cache.set(p, rows); return rows; });
    fetchRows.then(rows => render(p, rows)).catch(err => {
      VHINT.textContent = `Failed to load ${p}: ${err}`;
    });
  }

  function render(pert, rows) {
    if (!rows || rows.length === 0) {
      VHINT.textContent = `No DEGs for ${pert}.`;
      VHINT.hidden = false;
      Plotly.purge(VPLOT);
      plotted = false;
      return;
    }
    VHINT.hidden = true;

    // y = -log10(adj+eps); cap at 30 for display sanity
    const eps = 1e-300;
    const x = new Array(rows.length);
    const y = new Array(rows.length);
    const text = new Array(rows.length);
    const color = new Array(rows.length);
    let maxY = 0;
    for (let i = 0; i < rows.length; i++) {
      const r = rows[i];
      x[i] = r.lfc;
      const ny = -Math.log10(Math.max(r.adj, eps));
      y[i] = Math.min(ny, 30);
      text[i] = r.g;
      const sig = r.adj <= SIG_ADJ && Math.abs(r.lfc) >= SIG_LFC;
      color[i] = sig ? (r.lfc > 0 ? COL_UP : COL_DN) : COL_NS;
      if (y[i] > maxY) maxY = y[i];
    }

    // Top labels: top 8 up + top 8 down by combined |lfc| * -log10(adj)
    const idxs = rows.map((r,i) => i);
    idxs.sort((a,b) => {
      const sa = Math.abs(rows[a].lfc) * (-Math.log10(Math.max(rows[a].adj, eps)));
      const sb = Math.abs(rows[b].lfc) * (-Math.log10(Math.max(rows[b].adj, eps)));
      return sb - sa;
    });
    const topUp = []; const topDn = [];
    for (const i of idxs) {
      if (rows[i].lfc > 0 && topUp.length < 8) topUp.push(i);
      else if (rows[i].lfc < 0 && topDn.length < 8) topDn.push(i);
      if (topUp.length === 8 && topDn.length === 8) break;
    }
    const annotations = [...topUp, ...topDn].map(i => ({
      x: x[i], y: y[i], text: text[i],
      showarrow: false, font: { size: 10, color: '#222' },
      xanchor: rows[i].lfc > 0 ? 'left' : 'right',
      xshift: rows[i].lfc > 0 ? 6 : -6,
      yshift: 0,
    }));

    const trace = {
      type: 'scattergl', mode: 'markers',
      x, y, text,
      hovertemplate: '<b>%{text}</b><br>L2FC %{x:.2f}<br>−log10 adj-p %{y:.2f}<extra></extra>',
      marker: { color, size: 5.5, opacity: 0.85, line: { width: 0 } },
    };

    const sigLine = {
      type: 'line', xref: 'paper', x0: 0, x1: 1,
      y0: -Math.log10(SIG_ADJ), y1: -Math.log10(SIG_ADJ),
      line: { color: '#ddd', dash: 'dot', width: 1 },
    };

    const nUp = rows.filter(r => r.adj <= SIG_ADJ && r.lfc >= SIG_LFC).length;
    const nDn = rows.filter(r => r.adj <= SIG_ADJ && r.lfc <= -SIG_LFC).length;
    Plotly.react(VPLOT, [trace], {
      ...layout,
      title: { text: `${pert}  ·  ${rows.length} tested · ${nUp} up · ${nDn} down (adj p ≤ ${SIG_ADJ}, |L2FC| ≥ ${SIG_LFC})`,
               font: { size: 13 } },
      annotations,
      shapes: [sigLine],
    }, config);
    plotted = true;
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
Volcano.init();
Clusters.init();

// Initial route
const initial = location.hash.replace('#', '');
showTab(TABS.includes(initial) ? initial : 'mde');

(() => {
  const PLOT   = document.getElementById('plot');
  const SEARCH = document.getElementById('search');
  const EMPTY  = document.getElementById('empty');
  const RESET  = document.getElementById('reset');
  const PANEL  = document.getElementById('panel');
  const PCLOSE = document.getElementById('panel-close');

  const P = {
    gene:   document.getElementById('p-gene'),
    leiden: document.getElementById('p-leiden'),
    hdb:    document.getElementById('p-hdb'),
    ndegs:  document.getElementById('p-ndegs'),
    edist:  document.getElementById('p-edist'),
    nbr:    document.getElementById('p-nbr'),
  };

  // State
  let data = [];
  let leidenLabels = {};
  let hdbscanLabels = {};
  let indexByGene = new Map();
  let nDegsRank = new Map();   // gene -> rank by n_DEGs desc
  let edistRank = new Map();   // gene -> rank by edist desc
  let totalRanked = 0;
  let xRange = null, yRange = null;
  let selectedIdx = null;

  const GRAY = '#c7c7c7';
  function clusterColor(v) {
    if (v === -1) return GRAY;
    const golden = 0.61803398875;
    const hue = (v * golden * 360) % 360;
    return `hsl(${hue.toFixed(1)},62%,48%)`;
  }

  // Side-panel cluster line: "cluster N: <label or unlabeled>"; HDBSCAN -1 -> "unclustered"
  function panelClusterText(p, which) {
    const cid = p[which];
    if (which === 'h' && cid === -1) return 'unclustered';
    const labels = which === 'l' ? leidenLabels : hdbscanLabels;
    const label = labels[cid];
    return `cluster ${cid}: ${label && label.length ? label : 'unlabeled'}`;
  }

  function buildTrace(records) {
    const x = new Array(records.length);
    const y = new Array(records.length);
    const text = new Array(records.length);
    const hovertext = new Array(records.length);
    const color = new Array(records.length);

    for (let i = 0; i < records.length; i++) {
      const r = records[i];
      x[i] = r.x;
      y[i] = r.y;
      text[i] = r.g;
      hovertext[i] = `<b>${r.g}</b>`;
      color[i] = clusterColor(r.l);
    }

    return {
      type: 'scattergl',
      mode: 'markers',
      x, y, text, hovertext,
      hovertemplate: '%{hovertext}<extra></extra>',
      marker: { color, size: 6, opacity: 0.85, line: { width: 0 } },
    };
  }

  const layout = {
    margin: { l: 24, r: 24, t: 12, b: 24 },
    xaxis: { visible: false, scaleanchor: 'y', scaleratio: 1 },
    yaxis: { visible: false },
    hovermode: 'closest',
    showlegend: false,
    dragmode: 'pan',
    plot_bgcolor: '#fff',
    paper_bgcolor: '#fff',
    annotations: [],
  };

  const config = {
    responsive: true,
    displaylogo: false,
    scrollZoom: true,
    modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d', 'toggleSpikelines'],
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
  }

  function computeRanks() {
    const byN = [...data]
      .filter(r => r.n >= 0)
      .sort((a, b) => b.n - a.n);
    byN.forEach((r, i) => nDegsRank.set(r.g, i + 1));

    const byE = [...data]
      .filter(r => r.e != null)
      .sort((a, b) => b.e - a.e);
    byE.forEach((r, i) => edistRank.set(r.g, i + 1));

    // Both metrics cover the same 1655 perts in our payload; use that as denom.
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
    const relayout = { annotations: annotateSelected(r) };
    if (zoom) {
      const pad = Math.max((xRange[1] - xRange[0]), (yRange[1] - yRange[0])) * 0.06;
      relayout['xaxis.range'] = [r.x - pad, r.x + pad];
      relayout['yaxis.range'] = [r.y - pad, r.y + pad];
    }
    Plotly.relayout(PLOT, relayout);
    openPanel(idx);
  }

  function searchHighlight(query) {
    const q = query.trim().toUpperCase();
    if (!q) {
      EMPTY.hidden = true;
      return;
    }
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
    selectedIdx = null;
    closePanel();
    Plotly.relayout(PLOT, {
      'xaxis.range': xRange,
      'yaxis.range': yRange,
      annotations: [],
    });
  }

  // --- Panel ---
  function fmtInt(v) { return v == null || v < 0 ? '—' : v.toLocaleString(); }
  function fmtFloat(v, d = 2) { return v == null ? '—' : Number(v).toFixed(d); }
  function fmtRank(rank, total) { return rank ? `rank ${rank} / ${total}` : '—'; }
  function fmtWithRank(valStr, rank, total) {
    return rank ? `${valStr} · ${fmtRank(rank, total)}` : valStr;
  }

  function nearestNeighbors(idx, k = 10) {
    const a = data[idx];
    const out = [];
    for (let i = 0; i < data.length; i++) {
      if (i === idx) continue;
      const b = data[i];
      const dx = a.x - b.x, dy = a.y - b.y;
      out.push({ i, d: Math.sqrt(dx * dx + dy * dy) });
    }
    out.sort((u, v) => u.d - v.d);
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
      li.appendChild(g); li.appendChild(d);
      ol.appendChild(li);
    }
  }

  function openPanel(idx) {
    const r = data[idx];
    P.gene.textContent   = r.g;
    P.leiden.textContent = panelClusterText(r, 'l');
    P.hdb.textContent    = panelClusterText(r, 'h');
    P.ndegs.textContent  = fmtWithRank(fmtInt(r.n), nDegsRank.get(r.g), totalRanked);
    P.edist.textContent  = fmtWithRank(fmtFloat(r.e, 2), edistRank.get(r.g), totalRanked);

    renderNeighbors(P.nbr, nearestNeighbors(idx, 10));

    PANEL.classList.add('open');
    PANEL.setAttribute('aria-hidden', 'false');
  }

  function closePanel() {
    PANEL.classList.remove('open');
    PANEL.setAttribute('aria-hidden', 'true');
  }

  // --- Events ---
  SEARCH.addEventListener('input', e => searchHighlight(e.target.value));
  SEARCH.addEventListener('keydown', e => {
    if (e.key === 'Escape') { SEARCH.value = ''; resetView(); }
  });
  RESET.addEventListener('click', resetView);
  PCLOSE.addEventListener('click', closePanel);

  function attachPlotEvents() {
    PLOT.on('plotly_click', ev => {
      const pt = ev.points && ev.points[0];
      if (!pt) return;
      focusGene(pt.pointIndex, { zoom: false });
    });
  }

  // --- Load ---
  fetch('data/mde.json?v=3')
    .then(r => r.json())
    .then(payload => {
      data = payload.points;
      leidenLabels  = payload.leiden_labels  || {};
      hdbscanLabels = payload.hdbscan_labels || {};
      indexByGene = new Map(data.map((r, i) => [r.g, i]));
      computeRanks();
      initialAutorange();
      render();
      attachPlotEvents();
    })
    .catch(err => {
      PLOT.innerHTML = `<pre style="padding:24px;color:#a00">Failed to load data/mde.json: ${err}</pre>`;
    });
})();

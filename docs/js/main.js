(() => {
  const PLOT   = document.getElementById('plot');
  const SEARCH = document.getElementById('search');
  const EMPTY  = document.getElementById('empty');
  const RESET  = document.getElementById('reset');
  const SEGS   = document.querySelectorAll('.seg');
  const PANEL  = document.getElementById('panel');
  const PCLOSE = document.getElementById('panel-close');

  // Panel refs
  const P = {
    gene:   document.getElementById('p-gene'),
    leiden: document.getElementById('p-leiden'),
    hdb:    document.getElementById('p-hdb'),
    ndegs:  document.getElementById('p-ndegs'),
    edist:  document.getElementById('p-edist'),
    up:     document.getElementById('p-up').querySelector('tbody'),
    dn:     document.getElementById('p-dn').querySelector('tbody'),
    nbr:    document.getElementById('p-nbr'),
    corumS: document.getElementById('p-corum-section'),
    corum:  document.getElementById('p-corum'),
    corumH: document.getElementById('p-corum-hl'),
  };

  // State
  let data = [];                    // points
  let leidenLabels = {};
  let hdbscanLabels = {};
  let indexByGene = new Map();      // gene -> index into data
  let mode = 'l';                   // 'l' | 'h' | 'n' | 'e'
  let xRange = null, yRange = null;
  let selectedIdx = null;
  let corumHighlightOn = false;
  let currentCorumPartners = [];

  // Lazy caches
  let degsCache = null;             // loaded on first click
  let corumCache = null;            // loaded on first click

  const GRAY = '#c7c7c7';
  const HL   = '#d62728';           // highlight color for CORUM overlay

  function clusterColor(v) {
    if (v === -1) return GRAY;
    const golden = 0.61803398875;
    const hue = (v * golden * 360) % 360;
    return `hsl(${hue.toFixed(1)},62%,48%)`;
  }

  function clusterLabelFor(p, m) {
    const labels = m === 'l' ? leidenLabels : hdbscanLabels;
    const cid = p[m];
    if (cid === -1) return 'unclustered';
    return labels[cid] || `cluster ${cid}`;
  }

  function hoverFor(p, m) {
    if (m === 'l' || m === 'h') {
      const lab = clusterLabelFor(p, m);
      return `<b>${p.g}</b><br>${lab}`;
    }
    return `<b>${p.g}</b>`;
  }

  function buildTrace(records, m) {
    const x = new Array(records.length);
    const y = new Array(records.length);
    const text = new Array(records.length);
    const hovertext = new Array(records.length);

    for (let i = 0; i < records.length; i++) {
      const r = records[i];
      x[i] = r.x;
      y[i] = r.y;
      text[i] = r.g;
      hovertext[i] = hoverFor(r, m);
    }

    const trace = {
      type: 'scattergl',
      mode: 'markers',
      x, y, text, hovertext,
      hovertemplate: '%{hovertext}<extra></extra>',
      unselected: { marker: { opacity: 0.12 } },
      selected:   { marker: { opacity: 1.0 } },
    };

    if (m === 'l' || m === 'h') {
      const color = records.map(r => clusterColor(r[m]));
      trace.marker = { color, size: 6, opacity: 0.85, line: { width: 0 } };
    } else {
      // continuous: n_DEGs or edist, log1p-transformed, viridis
      const raw = records.map(r => m === 'n' ? r.n : r.e);
      const vals = raw.map(v => (v == null || v < 0) ? null : Math.log1p(v));
      const valid = vals.filter(v => v != null);
      const cmin = Math.min.apply(null, valid);
      const cmax = Math.max.apply(null, valid);
      trace.marker = {
        color: vals.map(v => v == null ? cmin : v),
        colorscale: 'Viridis',
        cmin, cmax,
        size: 6,
        opacity: 0.85,
        line: { width: 0 },
        showscale: true,
        colorbar: {
          thickness: 8,
          len: 0.4,
          x: 1.0, xanchor: 'right',
          y: 0.08, yanchor: 'bottom',
          outlinewidth: 0,
          tickfont: { size: 10, color: '#666' },
          title: {
            text: m === 'n' ? 'log(1+n_DEGs)' : 'log(1+edist)',
            font: { size: 10, color: '#666' },
            side: 'right',
          },
        },
      };
    }
    return trace;
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
    const trace = buildTrace(data, mode);
    layout.xaxis.range = xRange;
    layout.yaxis.range = yRange;
    Plotly.react(PLOT, [trace], layout, config);
    // After react, re-apply any active selection
    if (corumHighlightOn && currentCorumPartners.length) {
      applyCorumHighlight();
    } else if (selectedIdx != null) {
      Plotly.restyle(PLOT, { selectedpoints: [[selectedIdx]] });
    }
  }

  function clearSelection() {
    selectedIdx = null;
    corumHighlightOn = false;
    currentCorumPartners = [];
    P.corumH.classList.remove('on');
    Plotly.restyle(PLOT, { selectedpoints: [null] });
    Plotly.relayout(PLOT, { annotations: [] });
  }

  function searchHighlight(query) {
    const q = query.trim().toUpperCase();
    if (!q) {
      EMPTY.hidden = true;
      if (!PANEL.classList.contains('open')) clearSelection();
      return;
    }
    let idx = data.findIndex(r => r.g.toUpperCase() === q);
    if (idx < 0) idx = data.findIndex(r => r.g.toUpperCase().startsWith(q));
    if (idx < 0) idx = data.findIndex(r => r.g.toUpperCase().includes(q));
    if (idx < 0) {
      EMPTY.hidden = false;
      Plotly.restyle(PLOT, { selectedpoints: [null] });
      return;
    }
    EMPTY.hidden = true;
    focusGene(idx, { zoom: true });
  }

  function focusGene(idx, { zoom = false } = {}) {
    const r = data[idx];
    selectedIdx = idx;
    Plotly.restyle(PLOT, { selectedpoints: [[idx]] });
    const ann = [{
      x: r.x, y: r.y, text: r.g, showarrow: true, arrowhead: 0,
      ax: 0, ay: -28, font: { size: 13, color: '#111' },
      bgcolor: 'rgba(255,255,255,0.9)', bordercolor: '#111', borderwidth: 1, borderpad: 3,
    }];
    const relayout = { annotations: ann };
    if (zoom) {
      const pad = Math.max((xRange[1] - xRange[0]), (yRange[1] - yRange[0])) * 0.06;
      relayout['xaxis.range'] = [r.x - pad, r.x + pad];
      relayout['yaxis.range'] = [r.y - pad, r.y + pad];
    }
    Plotly.relayout(PLOT, relayout);
    openPanel(idx);
  }

  function resetView() {
    SEARCH.value = '';
    EMPTY.hidden = true;
    closePanel();
    clearSelection();
    Plotly.relayout(PLOT, {
      'xaxis.range': xRange,
      'yaxis.range': yRange,
      annotations: [],
    });
  }

  // --- Side panel ---
  function fmtInt(v)   { return v == null || v < 0 ? '—' : v.toLocaleString(); }
  function fmtFloat(v, d = 2) { return v == null ? '—' : Number(v).toFixed(d); }

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
      const l = document.createElement('td'); l.textContent = (r.lfc >= 0 ? '+' : '') + r.lfc.toFixed(2);
      tr.appendChild(g); tr.appendChild(l);
      tbody.appendChild(tr);
    }
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

  function renderCorumSection(gene) {
    const partners = (corumCache && corumCache[gene]) || [];
    currentCorumPartners = partners.filter(p => indexByGene.has(p));
    if (currentCorumPartners.length === 0) {
      P.corumS.hidden = true;
      return;
    }
    P.corumS.hidden = false;
    P.corum.innerHTML = '';
    for (const p of currentCorumPartners) {
      const li = document.createElement('li');
      const g = document.createElement('span'); g.className = 'g'; g.textContent = p;
      g.addEventListener('click', () => focusGene(indexByGene.get(p), { zoom: true }));
      li.appendChild(g);
      P.corum.appendChild(li);
    }
    P.corumH.classList.toggle('on', corumHighlightOn);
  }

  function applyCorumHighlight() {
    const idxs = currentCorumPartners.map(p => indexByGene.get(p)).filter(i => i != null);
    if (selectedIdx != null) idxs.push(selectedIdx);
    Plotly.restyle(PLOT, { selectedpoints: [idxs] });
  }

  async function ensureDegs() {
    if (degsCache) return degsCache;
    const r = await fetch('data/degs.json');
    degsCache = await r.json();
    return degsCache;
  }
  async function ensureCorum() {
    if (corumCache) return corumCache;
    const r = await fetch('data/corum.json');
    corumCache = await r.json();
    return corumCache;
  }

  async function openPanel(idx) {
    const r = data[idx];
    P.gene.textContent   = r.g;
    P.leiden.textContent = clusterLabelFor(r, 'l');
    P.hdb.textContent    = clusterLabelFor(r, 'h');
    P.ndegs.textContent  = fmtInt(r.n);
    P.edist.textContent  = fmtFloat(r.e, 2);

    // Neighbors synchronously (cheap)
    renderNeighbors(P.nbr, nearestNeighbors(idx, 10));

    PANEL.classList.add('open');
    PANEL.setAttribute('aria-hidden', 'false');

    // Lazy data: DEGs + CORUM
    try {
      const [degs, corum] = await Promise.all([ensureDegs(), ensureCorum()]);
      const d = degs[r.g];
      renderDegsTable(P.up, d ? d.up : null);
      renderDegsTable(P.dn, d ? d.dn : null);
      renderCorumSection(r.g);
    } catch (e) {
      console.error('panel data load failed', e);
    }
  }

  function closePanel() {
    PANEL.classList.remove('open');
    PANEL.setAttribute('aria-hidden', 'true');
  }

  // --- Event wiring ---
  SEARCH.addEventListener('input', e => searchHighlight(e.target.value));
  SEARCH.addEventListener('keydown', e => {
    if (e.key === 'Escape') { SEARCH.value = ''; resetView(); }
  });
  RESET.addEventListener('click', resetView);
  PCLOSE.addEventListener('click', () => { closePanel(); clearSelection(); });

  SEGS.forEach(btn => btn.addEventListener('click', () => {
    if (btn.classList.contains('active')) return;
    SEGS.forEach(b => { b.classList.remove('active'); b.setAttribute('aria-checked', 'false'); });
    btn.classList.add('active');
    btn.setAttribute('aria-checked', 'true');
    mode = btn.dataset.mode;
    render();
  }));

  P.corumH.addEventListener('click', () => {
    if (currentCorumPartners.length === 0) return;
    corumHighlightOn = !corumHighlightOn;
    P.corumH.classList.toggle('on', corumHighlightOn);
    if (corumHighlightOn) {
      applyCorumHighlight();
    } else if (selectedIdx != null) {
      Plotly.restyle(PLOT, { selectedpoints: [[selectedIdx]] });
    }
  });

  // Plot click → open side panel
  function attachPlotEvents() {
    PLOT.on('plotly_click', ev => {
      const pt = ev.points && ev.points[0];
      if (!pt) return;
      focusGene(pt.pointIndex, { zoom: false });
    });
  }

  // --- Load ---
  fetch('data/mde.json')
    .then(r => r.json())
    .then(payload => {
      data = payload.points;
      leidenLabels  = payload.leiden_labels  || {};
      hdbscanLabels = payload.hdbscan_labels || {};
      indexByGene = new Map(data.map((r, i) => [r.g, i]));
      initialAutorange();
      render();
      attachPlotEvents();
    })
    .catch(err => {
      PLOT.innerHTML = `<pre style="padding:24px;color:#a00">Failed to load data/mde.json: ${err}</pre>`;
    });
})();

(() => {
  const PLOT = document.getElementById('plot');
  const SEARCH = document.getElementById('search');
  const EMPTY = document.getElementById('empty');
  const RESET = document.getElementById('reset');
  const SEGS = document.querySelectorAll('.seg');

  let data = [];           // raw records
  let colorCol = 'l';      // 'l' | 'h'
  let xRange = null;
  let yRange = null;

  // Deterministic palette — HSL cycle with offset, gray for noise (-1)
  const GRAY = '#c7c7c7';
  function clusterColor(v) {
    if (v === -1) return GRAY;
    const golden = 0.61803398875;
    const hue = (v * golden * 360) % 360;
    return `hsl(${hue.toFixed(1)},62%,48%)`;
  }

  function buildTrace(records, col) {
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
      const n = r.n >= 0 ? r.n : '—';
      hovertext[i] = `<b>${r.g}</b><br>cluster: ${r[col]}<br>DEGs: ${n}`;
      color[i] = clusterColor(r[col]);
    }

    return {
      type: 'scattergl',
      mode: 'markers',
      x, y, text, hovertext,
      hovertemplate: '%{hovertext}<extra></extra>',
      marker: { color, size: 6, opacity: 0.85, line: { width: 0 } },
      unselected: { marker: { opacity: 0.12 } },
      selected:   { marker: { opacity: 1.0 } },
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
    const trace = buildTrace(data, colorCol);
    layout.xaxis.range = xRange;
    layout.yaxis.range = yRange;
    Plotly.react(PLOT, [trace], layout, config);
  }

  function clearHighlight() {
    layout.annotations = [];
    Plotly.relayout(PLOT, { annotations: [] });
    Plotly.restyle(PLOT, { selectedpoints: [null] });
  }

  function highlightGene(query) {
    const q = query.trim().toUpperCase();
    if (!q) { clearHighlight(); EMPTY.hidden = true; return; }

    // exact first, then prefix, then contains
    let idx = data.findIndex(r => r.g.toUpperCase() === q);
    if (idx < 0) idx = data.findIndex(r => r.g.toUpperCase().startsWith(q));
    if (idx < 0) idx = data.findIndex(r => r.g.toUpperCase().includes(q));

    if (idx < 0) {
      EMPTY.hidden = false;
      Plotly.restyle(PLOT, { selectedpoints: [null] });
      return;
    }
    EMPTY.hidden = true;

    const r = data[idx];
    Plotly.restyle(PLOT, { selectedpoints: [[idx]] });

    const pad = Math.max((xRange[1] - xRange[0]), (yRange[1] - yRange[0])) * 0.06;
    Plotly.relayout(PLOT, {
      'xaxis.range': [r.x - pad, r.x + pad],
      'yaxis.range': [r.y - pad, r.y + pad],
      annotations: [{
        x: r.x, y: r.y, text: r.g, showarrow: true, arrowhead: 0,
        ax: 0, ay: -28, font: { size: 13, color: '#111' },
        bgcolor: 'rgba(255,255,255,0.9)', bordercolor: '#111', borderwidth: 1, borderpad: 3,
      }],
    });
  }

  function resetView() {
    SEARCH.value = '';
    EMPTY.hidden = true;
    layout.annotations = [];
    Plotly.restyle(PLOT, { selectedpoints: [null] });
    Plotly.relayout(PLOT, {
      'xaxis.range': xRange,
      'yaxis.range': yRange,
      annotations: [],
    });
  }

  // Wire up events
  SEARCH.addEventListener('input', e => highlightGene(e.target.value));
  SEARCH.addEventListener('keydown', e => {
    if (e.key === 'Escape') { SEARCH.value = ''; resetView(); }
  });
  RESET.addEventListener('click', resetView);

  SEGS.forEach(btn => btn.addEventListener('click', () => {
    if (btn.classList.contains('active')) return;
    SEGS.forEach(b => { b.classList.remove('active'); b.setAttribute('aria-checked', 'false'); });
    btn.classList.add('active');
    btn.setAttribute('aria-checked', 'true');
    colorCol = btn.dataset.col;
    render();
  }));

  // Load data
  fetch('data/mde.json')
    .then(r => r.json())
    .then(records => {
      data = records;
      initialAutorange();
      render();
    })
    .catch(err => {
      PLOT.innerHTML = `<pre style="padding:24px;color:#a00">Failed to load data/mde.json: ${err}</pre>`;
    });
})();

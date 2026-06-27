"""
Build the per-perturbation gene-level payload for the Gene query tab's
"per perturbation" mode.

For each non-NTC perturbation column in zscore_filt_log1p.parquet:
  - top 50 up-regulated genes (highest mean NTC z)
  - top 50 down-regulated genes (lowest mean NTC z)
  - histogram of z across all 38,606 expressed genes (bin_edges [-2, 2] / 80)

Writes:
  docs/data/perts/topk.json   -- {pert: {up:[[gene,z]...], dn:[[gene,z]...], h:[counts]}}
  docs/data/perts/index.json  -- {perts:[...], n_genes, k, bin_centers, bin_edges}
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq


ROOT    = Path('/tscc/projects/ps-malilab/ydoctor/KOLF_Perturbation_Atlas')
PARQUET = ROOT / 'KOLF_Perturbation_Atlas_Analysis/webapp_artifacts/zscore_filt_log1p.parquet'
OUT_DIR = ROOT / 'docs/data/perts'
K       = 50


def main():
    t0 = time.time()
    log = lambda m: print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)

    log(f'loading {PARQUET}')
    pf = pq.ParquetFile(PARQUET)
    table = pf.read()
    df = table.to_pandas()
    drop_cols = [c for c in df.columns if c.startswith('__')]
    if drop_cols: df = df.drop(columns=drop_cols)
    genes = list(df.index.astype(str))
    perts = list(df.columns.astype(str))
    Z = df.to_numpy(dtype=np.float32, copy=False)
    log(f'Z shape={Z.shape} (genes x perts); sample genes={genes[:3]}')

    # Exclude NTC from queryable perts
    if 'NTC' in perts:
        ntc_j = perts.index('NTC')
        keep = np.array([j for j in range(len(perts)) if j != ntc_j])
        Zp = Z[:, keep]
        perts_use = [perts[j] for j in keep]
    else:
        Zp = Z
        perts_use = perts
    log(f'queryable perts: {len(perts_use)}')

    bin_edges   = np.linspace(-2.0, 2.0, 81)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Iterate perts (columns) -> top-k genes + histogram
    topk = {}
    n_perts = Zp.shape[1]
    for pi in range(n_perts):
        col = Zp[:, pi]
        up_i = np.argpartition(-col, K)[:K]
        up_i = up_i[np.argsort(-col[up_i])]
        dn_i = np.argpartition(col, K)[:K]
        dn_i = dn_i[np.argsort(col[dn_i])]
        h, _ = np.histogram(col, bins=bin_edges)
        topk[perts_use[pi]] = {
            'up': [[genes[j], round(float(col[j]), 2)] for j in up_i],
            'dn': [[genes[j], round(float(col[j]), 2)] for j in dn_i],
            'h':  h.astype(int).tolist(),
        }
        if (pi + 1) % 1000 == 0:
            log(f'  perts: {pi+1}/{n_perts}')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / 'topk.json').write_text(json.dumps(topk))
    (OUT_DIR / 'index.json').write_text(json.dumps({
        'perts':       list(perts_use),
        'n_genes':     int(Zp.shape[0]),
        'k':           K,
        'bin_centers': [round(float(x), 4) for x in bin_centers],
        'bin_edges':   [round(float(x), 4) for x in bin_edges],
    }))
    log(f'wrote topk.json ({(OUT_DIR/"topk.json").stat().st_size/1e6:.1f} MB) + index.json')
    log(f'done in {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()

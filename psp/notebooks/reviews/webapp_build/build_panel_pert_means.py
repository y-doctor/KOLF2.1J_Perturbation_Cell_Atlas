"""
Build docs/data/panel/pert_means.bin — strong-pert × all-gene int8 matrix
so the Gene panel tab can show per-specific-perturbation z values without
a cluster aggregation.

Shape: (n_strong_perts, n_genes) int8, row-major, scale 0.04.
  1655 * 38606 * 1 byte ~= 63.9 MB.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq


ROOT     = Path('/tscc/projects/ps-malilab/ydoctor/KOLF_Perturbation_Atlas')
PARQUET  = ROOT / 'KOLF_Perturbation_Atlas_Analysis/webapp_artifacts/zscore_filt_log1p.parquet'
STRONG   = ROOT / 'KOLF_Perturbation_Atlas_Analysis/webapp_artifacts/perts_in_strong.txt'
OUT_DIR  = ROOT / 'docs/data/panel'
SCALE    = 0.04


def main():
    t0 = time.time()
    log = lambda m: print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)

    strong = [p.strip() for p in STRONG.read_text().splitlines() if p.strip()]
    log(f'strong perts: {len(strong)}')

    pf = pq.ParquetFile(PARQUET)
    md = pf.schema_arrow.pandas_metadata or {}
    idx_cols = md.get('index_columns') or []
    idx_cols = [c for c in idx_cols if isinstance(c, str)]
    strong_set = set(strong)
    present = [c for c in pf.schema_arrow.names if c in strong_set]
    take_cols = present + idx_cols
    log(f'reading {len(present)} strong columns + {len(idx_cols)} index column')
    df = pf.read(columns=take_cols).to_pandas()
    genes = list(df.index.astype(str))
    perts_out = list(df.columns.astype(str))
    Z = df.to_numpy(dtype=np.float32, copy=False)
    log(f'Z shape={Z.shape} (genes x strong_perts)  n_genes={len(genes)}')

    # Transpose to (n_strong_perts, n_genes) row-major for binary write
    M = Z.T
    q = np.clip(np.round(M / SCALE), -127, 127).astype(np.int8)
    log(f'quantized |max|={int(np.abs(q).max())}, shape={q.shape}')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / 'pert_means.bin').write_bytes(q.tobytes(order='C'))
    (OUT_DIR / 'pert_means_meta.json').write_text(json.dumps({
        'scale':   SCALE,
        'perts':   perts_out,
        'genes':   genes,
        'n_perts': int(M.shape[0]),
        'n_genes': int(M.shape[1]),
    }))
    size_mb = (OUT_DIR / 'pert_means.bin').stat().st_size / 1e6
    log(f'wrote pert_means.bin  {size_mb:.1f} MB  + pert_means_meta.json')
    log(f'done in {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()

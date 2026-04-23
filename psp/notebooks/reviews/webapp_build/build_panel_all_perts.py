"""
Build the all-perturbation variant of the Gene panel matrix for the web UI.

Output layout (docs/data/panel/all/):
  meta.json
      {perts, genes, scale, chunk_size, n_chunks, n_genes, row_bytes}
  chunk_{N}.bin     N = 0..n_chunks-1
      int8 row-major. Each chunk holds up to chunk_size perts' rows.

Row layout inside a chunk: contiguous n_genes int8 cells per pert.
Client fetches 1 row (~38 KB) per picked pert via HTTP Range.

Sized so each chunk < 100 MB to fit GitHub's per-file limit.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq


ROOT       = Path('/tscc/projects/ps-malilab/ydoctor/KOLF_Perturbation_Atlas')
PARQUET    = ROOT / 'KOLF_Perturbation_Atlas_Analysis/webapp_artifacts/zscore_filt_log1p.parquet'
OUT_DIR    = ROOT / 'docs/data/panel/all'
SCALE      = 0.04
CHUNK_SIZE = 2500   # perts per chunk => 2500 * 38606 = ~96.5 MB / chunk


def main():
    t0 = time.time()
    log = lambda m: print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)

    pf = pq.ParquetFile(PARQUET)
    log('loading parquet (full)')
    df = pf.read().to_pandas()
    drop_cols = [c for c in df.columns if c.startswith('__')]
    if drop_cols: df = df.drop(columns=drop_cols)
    genes = list(df.index.astype(str))
    perts_all = list(df.columns.astype(str))
    # Exclude NTC
    keep = [p for p in perts_all if p != 'NTC']
    Z = df[keep].to_numpy(dtype=np.float32, copy=False)          # (n_genes, n_perts)
    perts_out = keep
    n_genes = Z.shape[0]
    n_perts = Z.shape[1]
    log(f'Z shape={Z.shape}; n_genes={n_genes}; n_perts={n_perts}')

    # Transpose to (n_perts, n_genes), row-major
    M = Z.T.copy()
    q = np.clip(np.round(M / SCALE), -127, 127).astype(np.int8)
    log(f'quantized |max|={int(np.abs(q).max())}')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_chunks = (n_perts + CHUNK_SIZE - 1) // CHUNK_SIZE
    for ci in range(n_chunks):
        a = ci * CHUNK_SIZE
        b = min((ci + 1) * CHUNK_SIZE, n_perts)
        chunk = q[a:b]                                              # (rows_in_chunk, n_genes)
        chunk_path = OUT_DIR / f'chunk_{ci}.bin'
        chunk_path.write_bytes(chunk.tobytes(order='C'))
        size_mb = chunk_path.stat().st_size / 1e6
        log(f'  wrote {chunk_path.name}  perts {a}..{b-1} ({b-a} rows)  {size_mb:.1f} MB')

    (OUT_DIR / 'meta.json').write_text(json.dumps({
        'perts':      perts_out,
        'genes':      genes,
        'scale':      SCALE,
        'chunk_size': CHUNK_SIZE,
        'n_chunks':   int(n_chunks),
        'n_perts':    int(n_perts),
        'n_genes':    int(n_genes),
        'row_bytes':  int(n_genes),
    }))
    log(f'wrote meta.json ({(OUT_DIR/"meta.json").stat().st_size/1e6:.2f} MB)')
    log(f'done in {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()

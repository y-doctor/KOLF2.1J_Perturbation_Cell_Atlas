"""
Build docs/data/clusters/summary.json from the z-score matrix, restricted
to strong perts. For each Leiden and HDBSCAN cluster, average z across its
members and emit top 30 up/dn genes.

Uses:
  - zscore_filt_log1p.parquet       (genes x perts)
  - docs/data/mde.json              (strong perts, .l .h cluster ids, manual labels)
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq


ROOT     = Path('/tscc/projects/ps-malilab/ydoctor/KOLF_Perturbation_Atlas')
PARQUET  = ROOT / 'KOLF_Perturbation_Atlas_Analysis/webapp_artifacts/zscore_filt_log1p.parquet'
MDE_JSON = ROOT / 'docs/data/mde.json'
OUT_DIR  = ROOT / 'docs/data/clusters'
K_TOP    = 30


def main():
    t0 = time.time()
    log = lambda m: print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)

    log(f'loading {MDE_JSON}')
    mde = json.loads(MDE_JSON.read_text())
    points = mde['points']
    leiden_labels  = {int(k): v for k, v in mde.get('leiden_labels',  {}).items()}
    hdbscan_labels = {int(k): v for k, v in mde.get('hdbscan_labels', {}).items()}
    strong = [p['g'] for p in points]
    leiden_of  = {p['g']: int(p['l']) for p in points}
    hdbscan_of = {p['g']: int(p['h']) for p in points}
    log(f'strong perts = {len(strong)}; leiden ids = {len(set(leiden_of.values()))}; '
        f'hdbscan ids = {len(set(hdbscan_of.values()))}')

    log(f'loading {PARQUET}')
    pf = pq.ParquetFile(PARQUET)
    # Only read columns for strong perts (saves ~6x RAM)
    cols_in_file = pf.schema_arrow.names
    take_cols = [c for c in cols_in_file if c in set(strong) or not c.startswith('__')]
    # re-filter strictly to strong set (skip NTC etc if present)
    take_cols = [c for c in cols_in_file if c in set(strong)]
    log(f'reading {len(take_cols)} strong columns from parquet')
    table = pf.read(columns=take_cols)
    df = table.to_pandas()
    genes = list(df.index.astype(str))
    perts = list(df.columns.astype(str))
    Z = df.to_numpy(dtype=np.float32, copy=False)
    log(f'Z shape={Z.shape} (genes x strong_perts)')

    def summarize(which_of: dict, labels_map: dict, kind: str):
        # group strong-pert column indices by cluster id
        groups: dict[int, list[int]] = {}
        for pi, pname in enumerate(perts):
            cid = which_of.get(pname)
            if cid is None: continue
            groups.setdefault(cid, []).append(pi)
        out = []
        for cid in sorted(groups):
            members_idx = groups[cid]
            if len(members_idx) == 0: continue
            sub = Z[:, members_idx]                              # genes x n_members
            avg = sub.mean(axis=1)                               # genes
            # top K up / dn
            up_i = np.argpartition(-avg, min(K_TOP, avg.size - 1))[:K_TOP]
            up_i = up_i[np.argsort(-avg[up_i])]
            dn_i = np.argpartition(avg,  min(K_TOP, avg.size - 1))[:K_TOP]
            dn_i = dn_i[np.argsort(avg[dn_i])]
            members = [perts[pi] for pi in members_idx]
            out.append({
                'id':       int(cid),
                'label':    labels_map.get(cid, '') or '',
                'n':        len(members),
                'members':  sorted(members),
                'up':       [{'g': genes[j], 'z': round(float(avg[j]), 3)} for j in up_i],
                'dn':       [{'g': genes[j], 'z': round(float(avg[j]), 3)} for j in dn_i],
            })
        log(f'  {kind}: {len(out)} clusters; '
            f'n_members range [{min(c["n"] for c in out)}, {max(c["n"] for c in out)}]')
        return out

    log('summarizing Leiden …')
    leiden  = summarize(leiden_of,  leiden_labels,  'leiden')
    log('summarizing HDBSCAN …')
    hdbscan = summarize(hdbscan_of, hdbscan_labels, 'hdbscan')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / 'summary.json'
    payload = {
        'leiden':  leiden,
        'hdbscan': hdbscan,
        'k_top':   K_TOP,
    }
    out_path.write_text(json.dumps(payload))
    log(f'wrote {out_path}  {out_path.stat().st_size/1e6:.2f} MB')
    log(f'done in {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()

import argparse, sys
import pandas as pd, numpy as np
from .io_utils import read_parquet, write_parquet

def main():
    ap = argparse.ArgumentParser(description="Joint candidate discovery from pileup.")
    ap.add_argument("--pileup", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-site-depth", type=int, default=5, help="Min total UMIs across all cells at site.")
    ap.add_argument("--min-alt-cells", type=int, default=3, help="Min cells with non-ref to keep candidate.")
    args = ap.parse_args()

    df = read_parquet(args.pileup)
    if df.empty:
        pd.DataFrame(columns=["pos","ref","alt","n_cells_alt","tot_umi"]).to_parquet(args.out, index=False); return

    agg = df.groupby("pos").agg({"A":"sum","C":"sum","G":"sum","T":"sum","umi_total":"sum"}).reset_index()
    acgt = agg[["A","C","G","T"]].values
    ref_idx = np.argmax(acgt, axis=1)
    ref_base = np.array(list("ACGT"))[ref_idx]
    candidates = []
    for i, row in agg.iterrows():
        pos = int(row["pos"])
        counts = {b:int(row[b]) for b in "ACGT"}
        rb = ref_base[i]
        for b in "ACGT":
            if b == rb: continue
            alt = counts[b]
            if alt <= 0: continue
            tot = int(row["umi_total"])
            if tot < args.min_site_depth: continue
            cell_support = ((df["pos"]==pos) & (df[b]>0)).sum()
            if cell_support >= args.min_alt_cells:
                candidates.append(dict(pos=pos, ref=rb, alt=b, n_cells_alt=int(cell_support), tot_umi=tot))
    cand_df = pd.DataFrame(candidates).drop_duplicates(subset=["pos","alt"])
    write_parquet(cand_df, args.out)
    print(f"Wrote {len(cand_df)} candidates to {args.out}", file=sys.stderr)

if __name__ == "__main__":
    main()

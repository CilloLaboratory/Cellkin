import argparse, sys
import numpy as np, pandas as pd
from .io_utils import read_parquet, write_parquet

def call_genotype(vaf, depth, het_lo=0.05, het_hi=0.85):
    if depth == 0 or pd.isna(vaf): return 9  # missing
    # Simple bands; widen near extremes for low depth
    if vaf < max(het_lo, 3.0/max(1,depth)) and vaf < 0.5*het_lo: return 0
    if vaf > min(het_hi, 1.0 - 3.0/max(1,depth)) and vaf > 0.95: return 2
    return 1

def main():
    ap = argparse.ArgumentParser(description="Per-cell genotyping at candidate variants.")
    ap.add_argument("--pileup", required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--het-lo", type=float, default=0.05)
    ap.add_argument("--het-hi", type=float, default=0.85)
    args = ap.parse_args()

    pu = read_parquet(args.pileup)
    var = read_parquet(args.variants)
    if var.empty or pu.empty:
        pd.DataFrame(columns=["cell","pos","ref","alt","ref_umi","alt_umi","depth","vaf","gt"]).to_parquet(args.out, index=False); return

    records = []
    for _, r in var.iterrows():
        pos, alt, ref = int(r.pos), r.alt, r.ref
        sub = pu[pu["pos"]==pos].copy()
        if sub.empty: continue
        sub["alt_umi"] = sub[alt]
        sub["ref_umi"] = sub[ref]
        sub["depth"] = sub[["A","C","G","T"]].sum(axis=1)
        sub["vaf"] = sub["alt_umi"]/sub["depth"].replace(0,np.nan)
        sub["ref"] = ref; sub["alt"] = alt; sub["pos"] = pos
        sub["gt"] = [call_genotype(v, d, args.het_lo, args.het_hi) for v,d in zip(sub["vaf"], sub["depth"])]
        records.append(sub[["cell","pos","ref","alt","ref_umi","alt_umi","depth","vaf","gt"]])
    out = pd.concat(records, ignore_index=True) if records else pd.DataFrame(columns=["cell","pos","ref","alt","ref_umi","alt_umi","depth","vaf","gt"])
    write_parquet(out, args.out)
    print(f"Wrote genotypes to {args.out}", file=sys.stderr)

if __name__ == "__main__":
    main()

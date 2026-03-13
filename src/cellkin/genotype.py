import argparse
import sys

import numpy as np
import pandas as pd

from .io_utils import read_parquet, write_parquet


def call_genotype(vaf, depth, het_lo=0.05, het_hi=0.85):
    if depth == 0 or pd.isna(vaf):
        return 9  # missing
    # Simple bands; widen near extremes for low depth
    if vaf < max(het_lo, 3.0 / max(1, depth)) and vaf < 0.5 * het_lo:
        return 0
    if vaf > min(het_hi, 1.0 - 3.0 / max(1, depth)) and vaf > 0.95:
        return 2
    return 1


def _call_genotype_vectorized(vaf: np.ndarray, depth: np.ndarray, het_lo: float, het_hi: float) -> np.ndarray:
    gt = np.full(vaf.shape, 1, dtype=np.int8)

    missing = (~np.isfinite(vaf)) | (depth == 0)
    gt[missing] = 9

    depth_safe = np.maximum(depth, 1)
    cond0 = (vaf < np.maximum(het_lo, 3.0 / depth_safe)) & (vaf < 0.5 * het_lo)
    cond2 = (vaf > np.minimum(het_hi, 1.0 - 3.0 / depth_safe)) & (vaf > 0.95)

    gt[cond0 & ~missing] = 0
    gt[cond2 & ~missing] = 2
    return gt


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
    out_cols = ["cell", "pos", "ref", "alt", "ref_umi", "alt_umi", "depth", "vaf", "gt"]
    if var.empty or pu.empty:
        write_parquet(pd.DataFrame(columns=out_cols), args.out)
        return

    pu_cols = ["cell", "pos", "A", "C", "G", "T"]
    pu_sub = pu[pu_cols].copy()
    pu_sub["depth"] = pu_sub[["A", "C", "G", "T"]].sum(axis=1)

    counts_long = pu_sub.melt(
        id_vars=["cell", "pos", "depth"], value_vars=["A", "C", "G", "T"], var_name="base", value_name="umi"
    )

    var_key = var[["pos", "ref", "alt"]].drop_duplicates().copy()

    alt_long = counts_long.rename(columns={"base": "alt", "umi": "alt_umi"})
    alt_join = var_key.merge(alt_long, on=["pos", "alt"], how="left")

    ref_long = counts_long.rename(columns={"base": "ref", "umi": "ref_umi"})[["cell", "pos", "ref", "ref_umi"]]
    out = alt_join.merge(ref_long, on=["cell", "pos", "ref"], how="left")
    out["alt_umi"] = out["alt_umi"].fillna(0).astype(int)
    out["ref_umi"] = out["ref_umi"].fillna(0).astype(int)
    out["depth"] = out["depth"].fillna(0).astype(int)

    depth = out["depth"].to_numpy(dtype=np.int32)
    alt = out["alt_umi"].to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        vaf = alt / np.where(depth > 0, depth, np.nan)

    out["vaf"] = vaf
    out["gt"] = _call_genotype_vectorized(vaf, depth, args.het_lo, args.het_hi).astype(int)

    out = out[out_cols].sort_values(["cell", "pos", "alt"]).reset_index(drop=True)
    write_parquet(out, args.out)
    print(f"Wrote genotypes to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

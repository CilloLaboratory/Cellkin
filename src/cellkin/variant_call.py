import argparse
import sys

import numpy as np
import pandas as pd

from .io_utils import read_parquet, write_parquet


BASES = ["A", "C", "G", "T"]


def main():
    ap = argparse.ArgumentParser(description="Joint candidate discovery from pileup.")
    ap.add_argument("--pileup", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-site-depth", type=int, default=5, help="Min total UMIs across all cells at site.")
    ap.add_argument("--min-alt-cells", type=int, default=3, help="Min cells with non-ref to keep candidate.")
    args = ap.parse_args()

    df = read_parquet(args.pileup)
    if df.empty:
        write_parquet(pd.DataFrame(columns=["pos", "ref", "alt", "n_cells_alt", "tot_umi"]), args.out)
        return

    agg = (
        df.groupby("pos", as_index=False)
        .agg({"A": "sum", "C": "sum", "G": "sum", "T": "sum", "umi_total": "sum"})
        .rename(columns={"umi_total": "tot_umi"})
    )

    acgt = agg[BASES].to_numpy()
    ref_idx = np.argmax(acgt, axis=1)
    agg["ref"] = np.array(BASES, dtype=object)[ref_idx]

    # Precompute per-site, per-base count of cells with non-zero support.
    support_df = (
        df.assign(A=df["A"] > 0, C=df["C"] > 0, G=df["G"] > 0, T=df["T"] > 0)
        .groupby("pos", as_index=False)[BASES]
        .sum()
    )
    support_long = support_df.melt(id_vars="pos", value_vars=BASES, var_name="alt", value_name="n_cells_alt")

    counts_long = agg[["pos", "ref", "tot_umi"] + BASES].melt(
        id_vars=["pos", "ref", "tot_umi"], value_vars=BASES, var_name="alt", value_name="alt_count"
    )

    cand = counts_long.merge(support_long, on=["pos", "alt"], how="left")
    cand["n_cells_alt"] = cand["n_cells_alt"].fillna(0).astype(int)

    cand = cand[
        (cand["alt"] != cand["ref"])
        & (cand["alt_count"] > 0)
        & (cand["tot_umi"] >= args.min_site_depth)
        & (cand["n_cells_alt"] >= args.min_alt_cells)
    ]

    out = cand[["pos", "ref", "alt", "n_cells_alt", "tot_umi"]].drop_duplicates(subset=["pos", "alt"])
    out = out.sort_values(["pos", "alt"]).reset_index(drop=True)
    write_parquet(out, args.out)
    print(f"Wrote {len(out)} candidates to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

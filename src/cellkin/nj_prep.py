import argparse
import sys

import numpy as np
import pandas as pd

from .io_utils import read_parquet, write_parquet


def _validate_args(args):
    if args.call_depth_for_metrics < 0:
        raise SystemExit("--call-depth-for-metrics must be >= 0")
    if args.min_cell_total_umi_depth < 0:
        raise SystemExit("--min-cell-total-umi-depth must be >= 0")
    if args.min_cell_covered_positions < 0:
        raise SystemExit("--min-cell-covered-positions must be >= 0")
    if args.min_cell_called_variant_sites < 0:
        raise SystemExit("--min-cell-called-variant-sites must be >= 0")
    if args.min_cell_mean_variant_depth < 0:
        raise SystemExit("--min-cell-mean-variant-depth must be >= 0")
    if not (0.0 <= args.min_cell_called_variant_fraction <= 1.0):
        raise SystemExit("--min-cell-called-variant-fraction must be between 0 and 1")


def compute_cell_qc(pileup: pd.DataFrame, genotypes: pd.DataFrame, call_depth_for_metrics: int) -> pd.DataFrame:
    cells = pd.Index(sorted(set(pileup.get("cell", pd.Series(dtype=object)).astype(str)) | set(genotypes.get("cell", pd.Series(dtype=object)).astype(str))))
    qc = pd.DataFrame({"cell": cells})

    if pileup.empty:
        pile = pd.DataFrame(columns=["cell", "total_umi_depth", "n_positions_with_coverage"])
    else:
        pu = pileup.copy()
        pu["cell"] = pu["cell"].astype(str)
        pile = (
            pu.groupby("cell", as_index=False)
            .agg(total_umi_depth=("umi_total", "sum"), n_positions_with_coverage=("umi_total", lambda x: int((x > 0).sum())))
        )

    if genotypes.empty:
        geno = pd.DataFrame(
            {
                "cell": cells,
                "n_variant_sites_called": 0,
                "fraction_variant_sites_called": 0.0,
                "mean_variant_depth": 0.0,
                "mean_variant_vaf": 0.0,
                "n_nonmissing_gt": 0,
            }
        )
        total_variant_sites = 0
    else:
        gt = genotypes.copy()
        gt["cell"] = gt["cell"].astype(str)

        total_variant_sites = int(gt[["pos", "alt"]].drop_duplicates().shape[0])
        called = gt["depth"] >= call_depth_for_metrics

        mean_vaf = gt.groupby("cell")["vaf"].mean()
        mean_vaf = mean_vaf.fillna(0.0)

        geno = pd.DataFrame(
            {
                "n_variant_sites_called": called.groupby(gt["cell"]).sum().astype(int),
                "mean_variant_depth": gt.groupby("cell")["depth"].mean().fillna(0.0),
                "mean_variant_vaf": mean_vaf,
                "n_nonmissing_gt": (gt["gt"] != 9).groupby(gt["cell"]).sum().astype(int),
            }
        ).reset_index().rename(columns={"index": "cell"})

        if total_variant_sites > 0:
            geno["fraction_variant_sites_called"] = geno["n_variant_sites_called"] / total_variant_sites
        else:
            geno["fraction_variant_sites_called"] = 0.0

    out = qc.merge(pile, on="cell", how="left").merge(geno, on="cell", how="left")

    fill_map = {
        "total_umi_depth": 0,
        "n_positions_with_coverage": 0,
        "n_variant_sites_called": 0,
        "fraction_variant_sites_called": 0.0,
        "mean_variant_depth": 0.0,
        "mean_variant_vaf": 0.0,
        "n_nonmissing_gt": 0,
    }
    for col, val in fill_map.items():
        out[col] = out[col].fillna(val)

    int_cols = ["total_umi_depth", "n_positions_with_coverage", "n_variant_sites_called", "n_nonmissing_gt"]
    out[int_cols] = out[int_cols].astype(int)

    ordered = [
        "cell",
        "total_umi_depth",
        "n_positions_with_coverage",
        "n_variant_sites_called",
        "fraction_variant_sites_called",
        "mean_variant_depth",
        "mean_variant_vaf",
        "n_nonmissing_gt",
    ]
    return out[ordered].sort_values("cell").reset_index(drop=True)


def apply_cell_filters(qc: pd.DataFrame, args) -> tuple[pd.Series, dict]:
    checks = {
        "min_cell_total_umi_depth": qc["total_umi_depth"] >= args.min_cell_total_umi_depth,
        "min_cell_covered_positions": qc["n_positions_with_coverage"] >= args.min_cell_covered_positions,
        "min_cell_called_variant_sites": qc["n_variant_sites_called"] >= args.min_cell_called_variant_sites,
        "min_cell_called_variant_fraction": qc["fraction_variant_sites_called"] >= args.min_cell_called_variant_fraction,
        "min_cell_mean_variant_depth": qc["mean_variant_depth"] >= args.min_cell_mean_variant_depth,
    }

    keep_mask = pd.Series(True, index=qc.index)
    for mask in checks.values():
        keep_mask &= mask

    fail_counts = {name: int((~mask).sum()) for name, mask in checks.items()}
    return keep_mask, fail_counts


def main():
    ap = argparse.ArgumentParser(description="Prepare NJ input by computing per-cell QC and optional filtering.")
    ap.add_argument("--pileup", required=True, help="Path to pileup.parquet")
    ap.add_argument("--genotypes", required=True, help="Path to genotypes.parquet")
    ap.add_argument("--out-prefix", required=True, help="Prefix for outputs (e.g., njprep)")

    ap.add_argument("--min-cell-total-umi-depth", type=int, default=0)
    ap.add_argument("--min-cell-covered-positions", type=int, default=0)
    ap.add_argument("--min-cell-called-variant-sites", type=int, default=0)
    ap.add_argument("--min-cell-called-variant-fraction", type=float, default=0.0)
    ap.add_argument("--min-cell-mean-variant-depth", type=float, default=0.0)
    ap.add_argument("--call-depth-for-metrics", type=int, default=1)

    args = ap.parse_args()
    _validate_args(args)

    pileup = read_parquet(args.pileup)
    genotypes = read_parquet(args.genotypes)

    qc = compute_cell_qc(pileup, genotypes, call_depth_for_metrics=args.call_depth_for_metrics)
    keep_mask, fail_counts = apply_cell_filters(qc, args)

    kept_cells = set(qc.loc[keep_mask, "cell"].astype(str))
    filtered_genotypes = genotypes[genotypes["cell"].astype(str).isin(kept_cells)].copy()

    qc_path = f"{args.out_prefix}.cell_qc.tsv"
    filtered_path = f"{args.out_prefix}.genotypes.filtered.parquet"
    qc.to_csv(qc_path, sep="\t", index=False)
    write_parquet(filtered_genotypes, filtered_path)

    total = len(qc)
    kept = int(keep_mask.sum())
    dropped = total - kept

    print(f"[nj-prep] cells total={total} kept={kept} dropped={dropped}", file=sys.stderr)
    print(
        "[nj-prep] fail counts: "
        + ", ".join(f"{k}={v}" for k, v in fail_counts.items()),
        file=sys.stderr,
    )
    print(f"[nj-prep] wrote: {qc_path} | {filtered_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

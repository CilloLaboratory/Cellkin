#!/usr/bin/env python3
import argparse
import json
import resource
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cellkin.build_nj import main as build_nj_main
from cellkin.genotype import main as genotype_main
from cellkin.variant_call import main as variant_call_main


ACCEPTANCE_TARGETS = {
    "variant_call_speedup": 2.0,
    "genotype_speedup": 2.0,
    "build_nj_speedup": 1.5,
    "umi_pileup_peak_mem_reduction": 0.30,
}


def peak_rss_mb() -> float:
    # ru_maxrss is bytes on macOS and KiB on Linux.
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if r > 10_000_000:  # likely bytes
        return r / (1024 * 1024)
    return r / 1024


def run_module(main_fn, argv):
    import sys

    old = sys.argv
    t0 = time.perf_counter()
    rss0 = peak_rss_mb()
    try:
        sys.argv = argv
        main_fn()
    finally:
        sys.argv = old
    return {"seconds": time.perf_counter() - t0, "peak_rss_mb": max(peak_rss_mb(), rss0)}


def make_synthetic_pileup(n_cells: int, n_sites: int, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cells = [f"cell{i}" for i in range(n_cells)]
    rows = []
    for pos in range(1, n_sites + 1):
        for c in cells:
            depth = int(rng.integers(4, 16))
            p = rng.dirichlet([8, 1.5, 1.5, 1.5])
            counts = rng.multinomial(depth, p)
            rows.append(
                {
                    "cell": c,
                    "pos": pos,
                    "umi_total": int(depth),
                    "A": int(counts[0]),
                    "C": int(counts[1]),
                    "G": int(counts[2]),
                    "T": int(counts[3]),
                }
            )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Synthetic benchmark harness for cellkin pipeline stages.")
    ap.add_argument("--cells", type=int, default=1000)
    ap.add_argument("--sites", type=int, default=200)
    ap.add_argument("--baseline-json", default=None, help="Optional JSON from previous run for speedup comparison.")
    ap.add_argument("--out-json", default=None, help="Optional path to write metrics JSON.")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pileup_path = td / "pileup.parquet"
        variants_path = td / "variants.parquet"
        genotypes_path = td / "genotypes.parquet"

        pileup = make_synthetic_pileup(args.cells, args.sites)
        pileup.to_parquet(pileup_path, index=False)

        metrics = {}
        metrics["variant_call"] = run_module(
            variant_call_main,
            ["variant_call.py", "--pileup", str(pileup_path), "--out", str(variants_path), "--min-site-depth", "5", "--min-alt-cells", "5"],
        )

        metrics["genotype"] = run_module(
            genotype_main,
            ["genotype.py", "--pileup", str(pileup_path), "--variants", str(variants_path), "--out", str(genotypes_path)],
        )

        metrics["build_nj"] = run_module(
            build_nj_main,
            ["build_nj.py", "--genotypes", str(genotypes_path), "--out-prefix", str(td / "nj"), "--distance-format", "condensed"],
        )

    result = {
        "dataset": {"cells": args.cells, "sites": args.sites},
        "metrics": metrics,
        "acceptance_targets": ACCEPTANCE_TARGETS,
    }

    if args.baseline_json:
        baseline = json.loads(Path(args.baseline_json).read_text())
        speedups = {}
        for stage in ["variant_call", "genotype", "build_nj"]:
            base_t = baseline["metrics"][stage]["seconds"]
            cur_t = metrics[stage]["seconds"]
            speedups[stage] = base_t / cur_t if cur_t > 0 else float("inf")
        result["speedups_vs_baseline"] = speedups

    text = json.dumps(result, indent=2)
    print(text)
    if args.out_json:
        Path(args.out_json).write_text(text)


if __name__ == "__main__":
    main()

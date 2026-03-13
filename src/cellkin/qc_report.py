import argparse, sys, os
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from .io_utils import read_parquet

def main():
    ap = argparse.ArgumentParser(description="QC report and basic plots.")
    ap.add_argument("--pileup", required=True)
    ap.add_argument("--genotypes", required=True)
    ap.add_argument("--clones", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    pu = read_parquet(args.pileup)
    gt = read_parquet(args.genotypes)
    clones = pd.read_csv(args.clones, sep="\t") if os.path.exists(args.clones) and os.path.getsize(args.clones)>0 else pd.DataFrame()

    cov = pu.groupby("cell").agg(depth=("umi_total","sum")).reset_index()
    plt.figure()
    cov["depth"].hist(bins=50)
    plt.xlabel("Total chrM UMI (per cell)")
    plt.ylabel("Cells")
    plt.title("Mito coverage per cell")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "coverage_hist.png"))
    plt.close()

    if not gt.empty:
        top = (gt.groupby(["pos","alt"])["alt_umi"].apply(lambda x: (x>0).sum())
               .sort_values(ascending=False).head(50).index.tolist())
        sub = gt.set_index(["pos","alt"]).loc[top].reset_index() if len(top)>0 else gt.copy()
        M = sub.pivot_table(index="cell", columns=["pos","alt"], values="vaf", aggfunc="first").fillna(0.0)
        plt.figure(figsize=(8,6))
        plt.imshow(M.values, aspect="auto", interpolation="nearest")
        plt.xlabel("Variants")
        plt.ylabel("Cells")
        plt.title("VAF heatmap (top variants)")
        plt.colorbar(label="VAF")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "vaf_heatmap.png"))
        plt.close()

    html = f"""
    <html><head><title>mtlineage QC</title></head>
    <body>
    <h1>mtlineage QC</h1>
    <h2>Coverage</h2>
    <img src="coverage_hist.png" width="600"/>
    <h2>VAF heatmap (top variants)</h2>
    <img src="vaf_heatmap.png" width="800"/>
    <h2>Tree (Newick)</h2>
    <pre>{open(args.tree).read() if os.path.exists(args.tree) else '(missing)'}</pre>
    <h2>Clones</h2>
    <pre>{clones.to_string(index=False) if not clones.empty else 'No clones detected'}</pre>
    </body></html>
    """
    with open(os.path.join(args.out, "index.html"), "w") as f: f.write(html)
    print(f"Wrote QC report to {args.out}/index.html", file=sys.stderr)

if __name__ == "__main__":
    main()

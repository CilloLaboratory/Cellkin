import argparse, sys
import numpy as np, pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

def newick_from_children(labels, children):
    n = len(labels)
    def build(node):
        if node < n:
            return labels[node]
        idx = node - n
        a,b = children[idx]
        return f"({build(a)},{build(b)})"
    root = len(children) + n - 1
    return build(root) + ";"

def main():
    ap = argparse.ArgumentParser(description="Clone calling & simple phylogeny.")
    ap.add_argument("--genotypes", required=True)
    ap.add_argument("--out", required=True, help="Clone table TSV")
    ap.add_argument("--tree", required=True, help="Newick tree path")
    ap.add_argument("--min_cells_per_clone", type=int, default=10)
    ap.add_argument("--distance_threshold", type=float, default=0.15)
    args = ap.parse_args()

    g = pd.read_parquet(args.genotypes)
    if g.empty:
        pd.DataFrame(columns=["clone_id","n_cells","cells","defining_variants"]).to_csv(args.out, sep="\t", index=False)
        open(args.tree,"w").write("();")
        return

    # cell x site
    M = g.pivot_table(index="cell", columns=["pos","alt"], values="gt", aggfunc="first").fillna(9).astype(int)
    cells = M.index.to_list()
    # Treat missing (9) and heteroplasmy (1) cautiously: map to 1 for neutral-ish distance
    X = M.replace({9:1}).values
    D = pairwise_distances(X, metric="hamming")

    ac = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average", distance_threshold=args.distance_threshold)
    labels = ac.fit_predict(D)
    df = pd.DataFrame({"cell":cells, "clone":labels})

    # summarize clones
    clones = []
    for cl, sub in df.groupby("clone"):
        members = sub["cell"].tolist()
        if len(members) < args.min_cells_per_clone: continue
        subM = M.loc[members]
        maj = subM.mode(axis=0, dropna=False).iloc[0]
        cohort_maj = M.mode(axis=0, dropna=False).iloc[0]
        defining = [f"{pos}:{alt}" for (pos,alt),v in maj.items() if v!=cohort_maj[(pos,alt)]]
        clones.append(dict(clone_id=int(cl), n_cells=len(members), cells=",".join(members),
                           defining_variants=",".join(defining)))
    clones_df = pd.DataFrame(clones).sort_values("n_cells", ascending=False)
    clones_df.to_csv(args.out, sep="\t", index=False)

    # tree on clone consensus
    if clones_df.empty:
        open(args.tree,"w").write("();"); return
    clone_ids = clones_df["clone_id"].tolist()
    cons = []
    for cl in clone_ids:
        members = df[df.clone==cl].cell.tolist()
        cons.append(M.loc[members].mode(axis=0, dropna=False).iloc[0].replace({9:1}).values)
    cons = np.vstack(cons)
    Dc = pairwise_distances(cons, metric="hamming")
    ac2 = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average",
                                  distance_threshold=0)
    ac2.fit(Dc)
    newick = newick_from_children([f"clone{c}" for c in clone_ids], ac2.children_)
    with open(args.tree,"w") as f: f.write(newick)
    print(f"Wrote clones to {args.out} and tree to {args.tree}", file=sys.stderr)

if __name__ == "__main__":
    main()

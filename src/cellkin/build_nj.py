import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def load_genotypes(path: str) -> pd.DataFrame:
    """
    Expected columns in genotypes.parquet:
      cell, pos, alt, vaf, depth
    (Other columns like ref, ref_umi, alt_umi can be present but are not required here.)
    """
    df = pd.read_parquet(path)
    required = {"cell", "pos", "alt", "vaf", "depth"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    return df


def make_vaf_matrix(df: pd.DataFrame, min_depth_for_call: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    Build cell x site VAF matrix (float) and DEPTH matrix (int).
    Sites are labeled as "pos:alt".
    If depth < min_depth_for_call in a cell at a site, VAF is set to NaN for that cell/site.
    Returns (VAF_df, DEPTH_df, cells, sites)
    """
    df = df.copy()
    df["site"] = df["pos"].astype(str) + ":" + df["alt"].astype(str)
    vaf = df.pivot_table(index="cell", columns="site", values="vaf", aggfunc="first")
    dep = df.pivot_table(index="cell", columns="site", values="depth", aggfunc="first")
    if min_depth_for_call > 0:
        vaf = vaf.mask(dep < min_depth_for_call, other=np.nan)
    cells = list(vaf.index.astype(str))
    sites = list(vaf.columns.astype(str))
    dep = dep.fillna(0).astype(int)
    return vaf, dep, cells, sites


def _pairwise_block_distance(
    Xi: np.ndarray,
    Di: np.ndarray,
    Xj: np.ndarray,
    Dj: np.ndarray,
    min_depth_pair: int,
    weight_scheme: str,
) -> np.ndarray:
    finite = np.isfinite(Xi)[:, None, :] & np.isfinite(Xj)[None, :, :]
    if min_depth_pair > 0:
        finite &= (Di[:, None, :] >= min_depth_pair) & (Dj[None, :, :] >= min_depth_pair)

    if weight_scheme == "min":
        w = np.minimum(Di[:, None, :], Dj[None, :, :])
        w = np.where(finite, w, 0.0)
    elif weight_scheme == "harmonic":
        denom = Di[:, None, :] + Dj[None, :, :] + 1e-12
        w = np.where(finite, 2.0 * Di[:, None, :] * Dj[None, :, :] / denom, 0.0)
    elif weight_scheme == "binary":
        w = finite.astype(np.float64)
    else:
        raise ValueError(f"Unknown weight_scheme: {weight_scheme}")

    diff = np.abs(Xi[:, None, :] - Xj[None, :, :])
    num = np.sum(w * diff, axis=2)
    den = np.sum(w, axis=2)

    out = np.full(num.shape, np.nan, dtype=np.float64)
    np.divide(num, den, out=out, where=den > 0)
    return out


def pairwise_depth_weighted_absdiff(
    vaf: pd.DataFrame,
    dep: pd.DataFrame,
    min_depth_pair: int = 0,
    weight_scheme: str = "min",
    block_size: int = 64,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute pairwise distances between rows (cells):
      d(i,j) = sum_s w_s(i,j) * |vaf_i - vaf_j| / sum_s w_s(i,j)
    where w_s(i,j) = min(depth_i_s, depth_j_s) [default], or other scheme.
    - Only sites where both depths >= min_depth_pair and both VAFs are finite contribute.
    Returns (D, labels)
    """
    X = vaf.to_numpy(dtype=np.float64, copy=False)
    Dmat = dep.to_numpy(dtype=np.float64, copy=False)
    n = X.shape[0]
    labels = list(vaf.index.astype(str))

    D = np.zeros((n, n), dtype=np.float64)
    for i0 in range(0, n, block_size):
        i1 = min(i0 + block_size, n)
        Xi = X[i0:i1]
        Di = Dmat[i0:i1]

        for j0 in range(i0, n, block_size):
            j1 = min(j0 + block_size, n)
            Xj = X[j0:j1]
            Dj = Dmat[j0:j1]
            block = _pairwise_block_distance(Xi, Di, Xj, Dj, min_depth_pair, weight_scheme)
            D[i0:i1, j0:j1] = block
            if i0 != j0:
                D[j0:j1, i0:i1] = block.T

    # Handle pairs with no overlap: set to max observed distance or 1.0 (conservative).
    tri = np.triu_indices(n, k=1)
    finite_vals = D[tri][np.isfinite(D[tri])]
    fallback = np.max(finite_vals) if finite_vals.size > 0 else 1.0
    D[~np.isfinite(D)] = fallback
    np.fill_diagonal(D, 0.0)
    return D, labels


class NJNode:
    def __init__(self, name: str):
        self.name = name
        self.children: List[Tuple["NJNode", float]] = []

    def is_leaf(self):
        return len(self.children) == 0

    def newick(self) -> str:
        if self.is_leaf():
            return self.name
        parts = [f"{child.newick()}:{max(0.0, bl):.6f}" for child, bl in self.children]
        return "(" + ",".join(parts) + ")"


def neighbor_joining(D: np.ndarray, labels: List[str]) -> NJNode:
    """
    Classic NJ (Saitou & Nei 1987) for additive distance matrices.
    D: NxN symmetric distance matrix
    labels: list of N leaf names
    Returns root NJNode (unrooted tree; we return a bifurcating structure with a phantom root).
    """
    labels = list(labels)
    D = D.astype(float).copy()

    nodes: Dict[str, NJNode] = {lab: NJNode(lab) for lab in labels}
    active_idx = list(range(len(labels)))
    active_labels = labels[:]

    while len(active_idx) > 2:
        k = len(active_idx)
        sub = D[np.ix_(active_idx, active_idx)]
        r = np.sum(sub, axis=1)
        Q = (k - 2) * sub - (r[:, None] + r[None, :])
        np.fill_diagonal(Q, np.inf)
        a_pos, b_pos = np.unravel_index(np.argmin(Q), Q.shape)
        i = active_idx[a_pos]
        j = active_idx[b_pos]
        li = active_labels[a_pos]
        lj = active_labels[b_pos]

        d_ij = D[i, j]
        delta = (r[a_pos] - r[b_pos]) / (k - 2) if k > 2 else 0.0
        L_i = 0.5 * d_ij + 0.5 * delta
        L_j = d_ij - L_i

        u_label = f"NJ{len(D)}"
        u_node = NJNode(u_label)
        u_node.children.append((nodes[li], max(0.0, L_i)))
        u_node.children.append((nodes[lj], max(0.0, L_j)))
        nodes[u_label] = u_node

        d_u = []
        for t in active_idx:
            if t in (i, j):
                d_u.append(0.0)
                continue
            d_u.append(0.5 * (D[i, t] + D[j, t] - d_ij))

        if D.shape[0] == D.shape[1] and len(active_idx) == D.shape[0]:
            D = np.pad(D, ((0, 1), (0, 1)), mode="constant", constant_values=0.0)
        u_index = D.shape[0] - 1

        for tpos, t in enumerate(active_idx):
            if t in (i, j):
                continue
            D[u_index, t] = D[t, u_index] = d_u[tpos]
        D[u_index, u_index] = 0.0

        for pos in sorted([a_pos, b_pos], reverse=True):
            active_idx.pop(pos)
            active_labels.pop(pos)
        active_idx.append(u_index)
        active_labels.append(u_label)

    i, j = active_idx
    li, lj = active_labels
    root = NJNode("root")
    dij = D[i, j]
    root.children.append((nodes[li], max(0.0, dij / 2.0)))
    root.children.append((nodes[lj], max(0.0, dij / 2.0)))
    return root


def write_condensed_distance(path: str, D: np.ndarray, labels: List[str]) -> None:
    rows = []
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            rows.append((labels[i], labels[j], D[i, j]))
    pd.DataFrame(rows, columns=["cell_i", "cell_j", "distance"]).to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser(description="Build cell-level NJ tree from genotypes.parquet (mt VAFs).")
    ap.add_argument("--genotypes", required=True, help="Path to genotypes.parquet")
    ap.add_argument("--out-prefix", required=True, help="Prefix for outputs (e.g., nj_out)")
    ap.add_argument(
        "--min-depth-for-call",
        type=int,
        default=0,
        help="Set VAF to NaN when depth < this at a site (per cell). 0 = keep all.",
    )
    ap.add_argument(
        "--min-depth-pair",
        type=int,
        default=0,
        help="Require both cells to have at least this depth at a site to contribute to pairwise distance.",
    )
    ap.add_argument(
        "--weight-scheme",
        choices=["min", "harmonic", "binary"],
        default="min",
        help="Weight for |ΔVAF| per site: min(depth_i,depth_j) (default), harmonic mean, or 1.",
    )
    ap.add_argument(
        "--distance-format",
        choices=["square", "condensed"],
        default="square",
        help="Distance output format: square matrix CSV (default) or condensed edge list CSV.",
    )
    args = ap.parse_args()

    df = load_genotypes(args.genotypes)
    vaf, dep, _, sites = make_vaf_matrix(df, min_depth_for_call=args.min_depth_for_call)
    vaf.to_parquet(f"{args.out_prefix}.vaf_matrix.parquet", index=True)

    D, labels = pairwise_depth_weighted_absdiff(vaf, dep, min_depth_pair=args.min_depth_pair, weight_scheme=args.weight_scheme)

    if args.distance_format == "square":
        distance_path = f"{args.out_prefix}.distance.csv"
        pd.DataFrame(D, index=labels, columns=labels).to_csv(distance_path)
    else:
        distance_path = f"{args.out_prefix}.distance.condensed.csv"
        write_condensed_distance(distance_path, D, labels)

    root = neighbor_joining(D, labels)
    newick = root.newick() + ";"
    with open(f"{args.out_prefix}.tree.newick", "w") as f:
        f.write(newick)

    print(
        f"[nj] cells={len(labels)} sites={len(sites)}  min_depth_for_call={args.min_depth_for_call}  "
        f"min_depth_pair={args.min_depth_pair}  weight={args.weight_scheme}"
    )
    print(f"[nj] wrote: {args.out_prefix}.vaf_matrix.parquet | {distance_path} | {args.out_prefix}.tree.newick")


if __name__ == "__main__":
    main()

import argparse
import math
import sys
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
    required = {"cell","pos","alt","vaf","depth"}
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
    # Pivot VAF and depth
    vaf = df.pivot_table(index="cell", columns="site", values="vaf", aggfunc="first")
    dep = df.pivot_table(index="cell", columns="site", values="depth", aggfunc="first")
    # Enforce NaN where depth < threshold
    if min_depth_for_call > 0:
        mask = (dep < min_depth_for_call)
        vaf = vaf.mask(mask, other=np.nan)
    cells = list(vaf.index.astype(str))
    sites = list(vaf.columns.astype(str))
    # Fill DEPTH NaNs with 0 for convenience downstream
    dep = dep.fillna(0).astype(int)
    return vaf, dep, cells, sites

def pairwise_depth_weighted_absdiff(vaf: pd.DataFrame, dep: pd.DataFrame, min_depth_pair: int = 0, weight_scheme: str = "min") -> Tuple[np.ndarray, List[str]]:
    """
    Compute pairwise distances between rows (cells):
      d(i,j) = sum_s w_s(i,j) * |vaf_i - vaf_j| / sum_s w_s(i,j)
    where w_s(i,j) = min(depth_i_s, depth_j_s) [default], or other scheme.
    - Only sites where both depths >= min_depth_pair and both VAFs are finite contribute.
    Returns (D, labels)
    """
    X = vaf.values  # float, NaN allowed
    Dmat = dep.values.astype(np.float64)  # depth (>=0)
    n, m = X.shape
    labels = list(vaf.index.astype(str))

    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        xi = X[i, :]
        di = Dmat[i, :]
        for j in range(i+1, n):
            xj = X[j, :]
            dj = Dmat[j, :]
            # valid positions: finite VAFs and depth >= threshold for both
            valid = np.isfinite(xi) & np.isfinite(xj)
            if min_depth_pair > 0:
                valid &= (di >= min_depth_pair) & (dj >= min_depth_pair)
            if not np.any(valid):
                dij = np.nan  # mark as NA; will replace later
            else:
                if weight_scheme == "min":
                    w = np.minimum(di[valid], dj[valid])
                elif weight_scheme == "harmonic":
                    w = 2.0 * di[valid] * dj[valid] / (di[valid] + dj[valid] + 1e-12)
                elif weight_scheme == "binary":
                    w = np.ones(np.count_nonzero(valid))
                else:
                    raise ValueError(f"Unknown weight_scheme: {weight_scheme}")
                diff = np.abs(xi[valid] - xj[valid])
                num = np.sum(w * diff)
                den = np.sum(w)
                dij = (num / den) if den > 0 else np.nan
            D[i, j] = D[j, i] = dij

    # Handle pairs with no overlap: set to max observed distance or 1.0 (conservative)
    finite_vals = D[np.triu_indices(n, k=1)][np.isfinite(D[np.triu_indices(n, k=1)])]
    fallback = np.max(finite_vals) if finite_vals.size > 0 else 1.0
    D[~np.isfinite(D)] = fallback
    np.fill_diagonal(D, 0.0)
    return D, labels

# -------- Neighbor-Joining (NJ) implementation --------
class NJNode:
    def __init__(self, name: str):
        self.name = name
        self.children: List[Tuple["NJNode", float]] = []  # (child, branch_length)

    def is_leaf(self):
        return len(self.children) == 0

    def newick(self) -> str:
        if self.is_leaf():
            return self.name
        else:
            parts = []
            for child, bl in self.children:
                parts.append(f"{child.newick()}:{max(0.0, bl):.6f}")
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
    n = len(labels)

    nodes: Dict[str, NJNode] = {lab: NJNode(lab) for lab in labels}

    def row_sums(mat):
        return np.sum(mat, axis=1)

    # Use a mapping: active holds indices into D; we'll append new rows/cols for new nodes
    active_idx = list(range(n))
    active_labels = labels[:]

    while len(active_idx) > 2:
        k = len(active_idx)
        sub = D[np.ix_(active_idx, active_idx)]
        r = row_sums(sub)
        Q = (k - 2) * sub - (r[:, None] + r[None, :])
        np.fill_diagonal(Q, np.inf)
        a_pos, b_pos = np.unravel_index(np.argmin(Q), Q.shape)
        i = active_idx[a_pos]; j = active_idx[b_pos]
        li = active_labels[a_pos]; lj = active_labels[b_pos]

        d_ij = D[i, j]
        delta = (r[a_pos] - r[b_pos]) / (k - 2) if k > 2 else 0.0
        L_i = 0.5 * d_ij + 0.5 * delta
        L_j = d_ij - L_i

        u_label = f"NJ{len(D)}"
        u_node = NJNode(u_label)
        # attach children (leaf/internal nodes found by name)
        # create if missing (should exist for leaves)
        # We need a mapping name->node; reconstruct via leaves and existing internals:
        # Keep a dict outside this loop:
        # (Simpler: store nodes by label in a persistent dictionary)
        # We'll initialize a global 'nodes' dict before loop and update it here.
        # (We access it as a closure variable.)
        # Attach:
        if li not in nodes: nodes[li] = NJNode(li)
        if lj not in nodes: nodes[lj] = NJNode(lj)
        u_node.children.append((nodes[li], max(0.0, L_i)))
        u_node.children.append((nodes[lj], max(0.0, L_j)))
        nodes[u_label] = u_node

        # distances from u to others
        d_u = []
        for tpos, t in enumerate(active_idx):
            if t in (i, j):
                d_u.append(0.0)
                continue
            d_it = D[i, t]
            d_jt = D[j, t]
            d_ut = 0.5 * (d_it + d_jt - d_ij)
            d_u.append(d_ut)

        # expand D by one row/col for u if needed
        if D.shape[0] == D.shape[1] and (len(active_idx) == D.shape[0]):
            # add a new row/col
            D = np.pad(D, ((0,1),(0,1)), mode='constant', constant_values=0.0)
        u_index = D.shape[0] - 1  # last index

        # set distances for u vs active nodes
        for tpos, t in enumerate(active_idx):
            if t in (i, j): 
                continue
            D[u_index, t] = D[t, u_index] = d_u[tpos]
        D[u_index, u_index] = 0.0

        # remove i,j from active, add u_index
        for pos in sorted([a_pos, b_pos], reverse=True):
            active_idx.pop(pos)
            active_labels.pop(pos)
        active_idx.append(u_index)
        active_labels.append(u_label)

    # final connection
    i, j = active_idx
    li, lj = active_labels
    root = NJNode("root")
    dij = D[i, j]
    root.children.append((nodes[li], max(0.0, dij/2.0)))
    root.children.append((nodes[lj], max(0.0, dij/2.0)))
    return root

def main():
    ap = argparse.ArgumentParser(description="Build cell-level NJ tree from genotypes.parquet (mt VAFs).")
    ap.add_argument("--genotypes", required=True, help="Path to genotypes.parquet")
    ap.add_argument("--out-prefix", required=True, help="Prefix for outputs (e.g., nj_out)")
    ap.add_argument("--min-depth-for-call", type=int, default=0, help="Set VAF to NaN when depth < this at a site (per cell). 0 = keep all.")
    ap.add_argument("--min-depth-pair", type=int, default=0, help="Require both cells to have at least this depth at a site to contribute to pairwise distance.")
    ap.add_argument("--weight-scheme", choices=["min","harmonic","binary"], default="min",
                    help="Weight for |ΔVAF| per site: min(depth_i,depth_j) (default), harmonic mean, or 1.")
    args = ap.parse_args()

    df = load_genotypes(args.genotypes)
    vaf, dep, cells, sites = make_vaf_matrix(df, min_depth_for_call=args.min_depth_for_call)
    vaf.to_parquet(f"{args.out_prefix}.vaf_matrix.parquet", index=True)

    D, labels = pairwise_depth_weighted_absdiff(vaf, dep, min_depth_pair=args.min_depth_pair, weight_scheme=args.weight_scheme)
    D_df = pd.DataFrame(D, index=labels, columns=labels)
    D_df.to_csv(f"{args.out_prefix}.distance.csv")

    root = neighbor_joining(D, labels)
    newick = root.newick() + ";"
    with open(f"{args.out_prefix}.tree.newick", "w") as f:
        f.write(newick)

    print(f"[nj] cells={len(labels)} sites={len(sites)}  min_depth_for_call={args.min_depth_for_call}  min_depth_pair={args.min_depth_pair}  weight={args.weight_scheme}")
    print(f"[nj] wrote: {args.out_prefix}.vaf_matrix.parquet | {args.out_prefix}.distance.csv | {args.out_prefix}.tree.newick")

if __name__ == "__main__":
    main()
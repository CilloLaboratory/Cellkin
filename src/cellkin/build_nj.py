import argparse
import os
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


def estimate_problem_shape(df: pd.DataFrame) -> Tuple[int, int]:
    n_cells = int(df["cell"].nunique())
    n_sites = int(df[["pos", "alt"]].drop_duplicates().shape[0])
    return n_cells, n_sites


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


def select_informative_sites(
    vaf: pd.DataFrame,
    dep: pd.DataFrame,
    top_k: int | None,
    frac: float | None,
    min_cells_per_site: int = 0,
    metric: str = "var",
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    if top_k is None and (frac is None or frac <= 0 or frac >= 1):
        return vaf, dep, list(vaf.columns.astype(str))

    counts = np.isfinite(vaf.to_numpy()).sum(axis=0)
    if min_cells_per_site > 0:
        keep_mask = counts >= min_cells_per_site
    else:
        keep_mask = np.ones(vaf.shape[1], dtype=bool)

    X = vaf.to_numpy(dtype=np.float64, copy=False)
    if metric == "var":
        means = np.nanmean(X, axis=0)
        score = np.nanmean((X - means) ** 2, axis=0)
    elif metric == "mad":
        med = np.nanmedian(X, axis=0)
        score = np.nanmedian(np.abs(X - med), axis=0)
    else:
        raise ValueError(f"Unknown informative-site metric: {metric}")

    score[~keep_mask] = -np.inf
    if frac is not None and (0 < frac < 1):
        k = max(1, int(round(frac * vaf.shape[1])))
    else:
        k = top_k if top_k is not None else vaf.shape[1]
    k = min(k, vaf.shape[1])
    if k <= 0:
        return vaf.iloc[:, []], dep.iloc[:, []], []

    idx = np.argpartition(-score, kth=k - 1)[:k]
    idx = idx[np.argsort(-score[idx])]
    cols = vaf.columns[idx]
    return vaf.loc[:, cols], dep.loc[:, cols], list(cols.astype(str))


def filter_sites_for_distance(
    vaf: pd.DataFrame,
    dep: pd.DataFrame,
    min_site_call_rate: float,
    min_site_cells: int,
    min_cohort_vaf: float,
    max_cohort_vaf: float,
    min_depth_for_call: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    n_cells = dep.shape[0]
    if dep.shape[1] == 0:
        return vaf, dep, 0

    call_depth = min_depth_for_call if min_depth_for_call > 0 else 1
    called = dep >= call_depth
    site_called_cells = called.sum(axis=0)
    site_call_rate = site_called_cells / max(1, n_cells)

    dep_f = dep.to_numpy(dtype=np.float64, copy=False)
    vaf_f = vaf.to_numpy(dtype=np.float64, copy=False)
    numer = np.nansum(vaf_f * dep_f, axis=0)
    denom = np.sum(dep_f, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cohort_vaf = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0)

    keep = (
        (site_call_rate.to_numpy() >= min_site_call_rate)
        & (site_called_cells.to_numpy() >= min_site_cells)
        & (cohort_vaf >= min_cohort_vaf)
        & (cohort_vaf <= max_cohort_vaf)
    )

    kept = int(np.count_nonzero(keep))
    if kept == dep.shape[1]:
        return vaf, dep, kept

    cols = dep.columns[keep]
    return vaf.loc[:, cols], dep.loc[:, cols], kept


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


def _iter_distance_blocks(
    X: np.ndarray,
    Dmat: np.ndarray,
    min_depth_pair: int,
    weight_scheme: str,
    block_size: int,
):
    n = X.shape[0]
    for i0 in range(0, n, block_size):
        i1 = min(i0 + block_size, n)
        Xi = X[i0:i1]
        Di = Dmat[i0:i1]
        for j0 in range(i0, n, block_size):
            j1 = min(j0 + block_size, n)
            Xj = X[j0:j1]
            Dj = Dmat[j0:j1]
            block = _pairwise_block_distance(Xi, Di, Xj, Dj, min_depth_pair, weight_scheme)
            yield i0, i1, j0, j1, block


def pairwise_depth_weighted_absdiff(
    vaf: pd.DataFrame,
    dep: pd.DataFrame,
    min_depth_pair: int = 0,
    weight_scheme: str = "min",
    block_size: int = 64,
) -> Tuple[np.ndarray, List[str], float, float]:
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
    max_finite = np.nan
    no_overlap_pairs = 0
    total_pairs = (n * (n - 1)) // 2

    for i0, i1, j0, j1, block in _iter_distance_blocks(X, Dmat, min_depth_pair, weight_scheme, block_size):
        D[i0:i1, j0:j1] = block
        if i0 != j0:
            D[j0:j1, i0:i1] = block.T
            vals = block.ravel()
        else:
            tri = np.triu_indices(block.shape[0], k=1)
            vals = block[tri]

        no_overlap_pairs += int(np.count_nonzero(~np.isfinite(vals)))
        finite = vals[np.isfinite(vals)]
        if finite.size > 0:
            block_max = float(np.max(finite))
            if not np.isfinite(max_finite) or block_max > max_finite:
                max_finite = block_max

    fallback = max_finite if np.isfinite(max_finite) else 1.0
    D[~np.isfinite(D)] = fallback
    np.fill_diagonal(D, 0.0)
    no_overlap_fraction = (no_overlap_pairs / total_pairs) if total_pairs > 0 else 0.0
    return D, labels, no_overlap_fraction, fallback


def write_condensed_distance(path: str, D: np.ndarray, labels: List[str]) -> None:
    rows = []
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            rows.append((labels[i], labels[j], D[i, j]))
    pd.DataFrame(rows, columns=["cell_i", "cell_j", "distance"]).to_csv(path, index=False)


def stream_condensed_distance(
    path: str,
    vaf: pd.DataFrame,
    dep: pd.DataFrame,
    min_depth_pair: int,
    weight_scheme: str,
    block_size: int,
) -> Tuple[int, float, float]:
    X = vaf.to_numpy(dtype=np.float64, copy=False)
    Dmat = dep.to_numpy(dtype=np.float64, copy=False)
    labels = np.asarray(vaf.index.astype(str))

    max_finite = np.nan
    no_overlap_pairs = 0
    total_pairs = (X.shape[0] * (X.shape[0] - 1)) // 2
    for i0, i1, j0, j1, block in _iter_distance_blocks(X, Dmat, min_depth_pair, weight_scheme, block_size):
        if i0 == j0:
            tri = np.triu_indices(block.shape[0], k=1)
            vals = block[tri]
        else:
            vals = block.ravel()
        no_overlap_pairs += int(np.count_nonzero(~np.isfinite(vals)))
        finite = vals[np.isfinite(vals)]
        if finite.size > 0:
            block_max = float(np.max(finite))
            if not np.isfinite(max_finite) or block_max > max_finite:
                max_finite = block_max

    fallback = max_finite if np.isfinite(max_finite) else 1.0

    first = True
    n_pairs = 0
    for i0, i1, j0, j1, block in _iter_distance_blocks(X, Dmat, min_depth_pair, weight_scheme, block_size):
        if i0 == j0:
            tri_i, tri_j = np.triu_indices(block.shape[0], k=1)
            if tri_i.size == 0:
                continue
            dist = block[tri_i, tri_j]
            dist = np.where(np.isfinite(dist), dist, fallback)
            cell_i = labels[i0:i1][tri_i]
            cell_j = labels[j0:j1][tri_j]
        else:
            bi, bj = block.shape
            row_idx = np.repeat(np.arange(bi), bj)
            col_idx = np.tile(np.arange(bj), bi)
            dist = block.ravel()
            dist = np.where(np.isfinite(dist), dist, fallback)
            cell_i = labels[i0:i1][row_idx]
            cell_j = labels[j0:j1][col_idx]

        out = pd.DataFrame({"cell_i": cell_i, "cell_j": cell_j, "distance": dist})
        out.to_csv(path, index=False, mode="w" if first else "a", header=first)
        first = False
        n_pairs += len(out)

    if first:
        pd.DataFrame(columns=["cell_i", "cell_j", "distance"]).to_csv(path, index=False)

    no_overlap_fraction = (no_overlap_pairs / total_pairs) if total_pairs > 0 else 0.0
    return n_pairs, fallback, no_overlap_fraction


def _system_memory_bytes() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        if page_size > 0 and phys_pages > 0:
            return int(page_size * phys_pages)
    except (ValueError, OSError, AttributeError):
        return None
    return None


def _fmt_gb(num_bytes: float) -> str:
    return f"{num_bytes / (1024**3):.1f} GB"


def estimate_peak_memory_bytes(
    n_cells: int,
    n_sites: int,
    block_size: int,
    large_scale_mode: bool,
    build_tree: bool,
    distance_format: str,
) -> float:
    dense_inputs = float(16 * n_cells * n_sites)  # X + Dmat float64
    block_temps = float((17 * block_size * block_size * n_sites) + (24 * block_size * block_size))

    if large_scale_mode:
        return dense_inputs + block_temps

    dense_distance = float(8 * n_cells * n_cells)  # D
    nj_copy = float(8 * n_cells * n_cells) if build_tree else 0.0
    output_copy = float(8 * n_cells * n_cells) if distance_format == "square" else 0.0
    return dense_inputs + block_temps + dense_distance + nj_copy + output_copy


def preflight_or_die(
    n_cells: int,
    n_sites: int,
    block_size: int,
    large_scale_mode: bool,
    build_tree: bool,
    distance_format: str,
    max_memory_gb: float | None,
) -> None:
    est = estimate_peak_memory_bytes(
        n_cells=n_cells,
        n_sites=n_sites,
        block_size=block_size,
        large_scale_mode=large_scale_mode,
        build_tree=build_tree,
        distance_format=distance_format,
    )

    if max_memory_gb is not None:
        budget = max_memory_gb * (1024**3)
        budget_label = f"--max-memory-gb={max_memory_gb}"
    else:
        total_mem = _system_memory_bytes()
        if total_mem is None:
            return
        budget = total_mem * 0.80
        budget_label = "80% of detected system RAM"

    if est <= budget:
        return

    tips = [
        "use --large-scale-mode --distance-format condensed",
        "increase --min-depth-for-call and/or --min-depth-pair to reduce effective density",
        "reduce --block-size (e.g. 32) to lower temporary block allocations",
    ]
    if build_tree:
        tips.insert(0, "disable tree generation with --no-tree")

    raise SystemExit(
        "Preflight memory check failed for cellkin-build-nj. "
        f"Estimated peak memory is {_fmt_gb(est)} for ~{n_cells:,} cells x {n_sites:,} sites, "
        f"which exceeds {budget_label} ({_fmt_gb(budget)}). "
        "Suggested mitigations: " + "; ".join(tips)
    )


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


def neighbor_joining(D: np.ndarray, labels: List[str], copy_matrix: bool = True) -> NJNode:
    """
    Classic NJ (Saitou & Nei 1987) for additive distance matrices.
    D: NxN symmetric distance matrix
    labels: list of N leaf names
    Returns root NJNode (unrooted tree; we return a bifurcating structure with a phantom root).
    """
    labels = list(labels)
    D = D.astype(float, copy=copy_matrix)

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
        "--sites-informative-k",
        type=int,
        default=None,
        help="Old-model site selection: keep top-K most variable sites.",
    )
    ap.add_argument(
        "--sites-informative-frac",
        type=float,
        default=None,
        help="Old-model site selection: keep top fraction (0..1) most variable sites.",
    )
    ap.add_argument(
        "--min-cells-per-site",
        type=int,
        default=0,
        help="Old-model site selection: require at least this many finite-VAF cells at a site.",
    )
    ap.add_argument(
        "--informative-metric",
        choices=["var", "mad"],
        default="var",
        help="Old-model site informativeness metric.",
    )
    ap.add_argument(
        "--min-site-call-rate",
        type=float,
        default=0.0,
        help="Keep only sites called in at least this fraction of cells (0-1), based on depth threshold.",
    )
    ap.add_argument(
        "--min-site-cells",
        type=int,
        default=0,
        help="Keep only sites called in at least this many cells.",
    )
    ap.add_argument(
        "--min-cohort-vaf",
        type=float,
        default=0.0,
        help="Keep only sites with weighted cohort VAF >= this value.",
    )
    ap.add_argument(
        "--max-cohort-vaf",
        type=float,
        default=1.0,
        help="Keep only sites with weighted cohort VAF <= this value.",
    )
    ap.add_argument(
        "--distance-format",
        choices=["square", "condensed"],
        default="square",
        help="Distance output format: square matrix CSV (default) or condensed edge list CSV.",
    )
    ap.add_argument(
        "--block-size",
        type=int,
        default=64,
        help="Block size for pairwise distance computation. Lower reduces peak temp memory.",
    )
    ap.add_argument(
        "--large-scale-mode",
        action="store_true",
        help="Avoid full NxN distance materialization by streaming condensed distances; disables NJ tree generation.",
    )
    ap.add_argument(
        "--max-memory-gb",
        type=float,
        default=None,
        help="Hard budget for preflight memory estimate. If omitted, uses ~80% of detected system RAM.",
    )
    ap.add_argument(
        "--no-tree",
        action="store_true",
        help="Skip NJ tree construction and write an empty placeholder tree (();) to save memory/time.",
    )
    ap.add_argument(
        "--max-no-overlap-fraction",
        type=float,
        default=1.0,
        help="Abort if fraction of cell pairs with no overlapping callable sites exceeds this value (0-1).",
    )
    args = ap.parse_args()

    if args.block_size <= 0:
        raise SystemExit("--block-size must be > 0")
    if args.sites_informative_k is not None and args.sites_informative_k <= 0:
        raise SystemExit("--sites-informative-k must be > 0")
    if args.sites_informative_frac is not None and not (0 < args.sites_informative_frac < 1):
        raise SystemExit("--sites-informative-frac must be between 0 and 1")
    if args.min_cells_per_site < 0:
        raise SystemExit("--min-cells-per-site must be >= 0")
    if not (0.0 <= args.min_site_call_rate <= 1.0):
        raise SystemExit("--min-site-call-rate must be between 0 and 1")
    if args.min_site_cells < 0:
        raise SystemExit("--min-site-cells must be >= 0")
    if not (0.0 <= args.min_cohort_vaf <= 1.0 and 0.0 <= args.max_cohort_vaf <= 1.0):
        raise SystemExit("--min-cohort-vaf/--max-cohort-vaf must be between 0 and 1")
    if args.min_cohort_vaf > args.max_cohort_vaf:
        raise SystemExit("--min-cohort-vaf cannot be greater than --max-cohort-vaf")
    if not (0.0 <= args.max_no_overlap_fraction <= 1.0):
        raise SystemExit("--max-no-overlap-fraction must be between 0 and 1")

    if args.large_scale_mode and args.distance_format != "condensed":
        raise SystemExit("--large-scale-mode requires --distance-format condensed")

    df = load_genotypes(args.genotypes)
    n_cells, n_sites = estimate_problem_shape(df)

    build_tree = not args.no_tree and not args.large_scale_mode
    preflight_or_die(
        n_cells=n_cells,
        n_sites=n_sites,
        block_size=args.block_size,
        large_scale_mode=args.large_scale_mode,
        build_tree=build_tree,
        distance_format=args.distance_format,
        max_memory_gb=args.max_memory_gb,
    )

    vaf, dep, labels, sites = make_vaf_matrix(df, min_depth_for_call=args.min_depth_for_call)
    sites_total = dep.shape[1]

    # Reintroduced old-model informative site selection (variance/MAD ranking).
    vaf, dep, informative_sites = select_informative_sites(
        vaf=vaf,
        dep=dep,
        top_k=args.sites_informative_k,
        frac=args.sites_informative_frac,
        min_cells_per_site=args.min_cells_per_site,
        metric=args.informative_metric,
    )

    # Keep newer cohort/site-call filters as optional extra gating.
    vaf, dep, kept_sites = filter_sites_for_distance(
        vaf=vaf,
        dep=dep,
        min_site_call_rate=args.min_site_call_rate,
        min_site_cells=args.min_site_cells,
        min_cohort_vaf=args.min_cohort_vaf,
        max_cohort_vaf=args.max_cohort_vaf,
        min_depth_for_call=args.min_depth_for_call,
    )
    if kept_sites == 0:
        raise SystemExit(
            "No sites remain after filtering for distance computation. "
            "Relax informative/site-call/cohort filters or lower --min-depth-for-call."
        )
    vaf.to_parquet(f"{args.out_prefix}.vaf_matrix.parquet", index=True)

    if args.large_scale_mode:
        distance_path = f"{args.out_prefix}.distance.condensed.csv"
        n_pairs, fallback, no_overlap_fraction = stream_condensed_distance(
            distance_path,
            vaf,
            dep,
            min_depth_pair=args.min_depth_pair,
            weight_scheme=args.weight_scheme,
            block_size=args.block_size,
        )
        if no_overlap_fraction > args.max_no_overlap_fraction:
            raise SystemExit(
                f"No-overlap fraction check failed: {no_overlap_fraction:.4f} > "
                f"--max-no-overlap-fraction={args.max_no_overlap_fraction:.4f}. "
                "Increase overlap by relaxing filtering/depth thresholds or lowering this safeguard intentionally."
            )
        with open(f"{args.out_prefix}.tree.newick", "w") as f:
            f.write("();")
        print(
            f"[nj] large-scale mode enabled; streamed {n_pairs:,} pair distances to {distance_path}; "
            f"tree generation skipped (wrote placeholder tree). "
            f"fallback_distance={fallback:.6f} no_overlap_fraction={no_overlap_fraction:.4f}"
        )
        return

    D, labels, no_overlap_fraction, _fallback = pairwise_depth_weighted_absdiff(
        vaf,
        dep,
        min_depth_pair=args.min_depth_pair,
        weight_scheme=args.weight_scheme,
        block_size=args.block_size,
    )
    if no_overlap_fraction > args.max_no_overlap_fraction:
        raise SystemExit(
            f"No-overlap fraction check failed: {no_overlap_fraction:.4f} > "
            f"--max-no-overlap-fraction={args.max_no_overlap_fraction:.4f}. "
            "Increase overlap by relaxing filtering/depth thresholds or lowering this safeguard intentionally."
        )

    if args.distance_format == "square":
        distance_path = f"{args.out_prefix}.distance.csv"
        pd.DataFrame(D, index=labels, columns=labels).to_csv(distance_path)
    else:
        distance_path = f"{args.out_prefix}.distance.condensed.csv"
        write_condensed_distance(distance_path, D, labels)

    if build_tree:
        root = neighbor_joining(D, labels, copy_matrix=False)
        newick = root.newick() + ";"
        with open(f"{args.out_prefix}.tree.newick", "w") as f:
            f.write(newick)
    else:
        with open(f"{args.out_prefix}.tree.newick", "w") as f:
            f.write("();")

    print(
        f"[nj] cells={len(labels)} sites={kept_sites}/{sites_total} (post-informative+cohort filters)  "
        f"min_depth_for_call={args.min_depth_for_call}  "
        f"min_depth_pair={args.min_depth_pair}  weight={args.weight_scheme}  "
        f"no_overlap_fraction={no_overlap_fraction:.4f}"
    )
    print(f"[nj] wrote: {args.out_prefix}.vaf_matrix.parquet | {distance_path} | {args.out_prefix}.tree.newick")


if __name__ == "__main__":
    main()

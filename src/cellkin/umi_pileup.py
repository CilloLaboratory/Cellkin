import argparse
import sys
from collections import defaultdict

import pandas as pd

from .io_utils import open_bam, open_fasta, write_parquet


BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
IDX_TO_BASE = ["A", "C", "G", "T"]


def consensus_base_from_counts(counts):
    # Deterministic tie break by A,C,G,T ordering.
    total = counts[0] + counts[1] + counts[2] + counts[3]
    if total == 0:
        return None, 0, 0.0
    max_idx = 0
    max_val = counts[0]
    for idx in (1, 2, 3):
        if counts[idx] > max_val:
            max_idx = idx
            max_val = counts[idx]
    return IDX_TO_BASE[max_idx], max_val, max_val / max(1, total)


def main():
    ap = argparse.ArgumentParser(description="UMI-aware pileup per cell for chrM.")
    ap.add_argument("--bam", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-mapq", type=int, default=20)
    ap.add_argument("--min-baseq", type=int, default=20)
    ap.add_argument("--mito-chr", default="chrM")
    args = ap.parse_args()

    bam = open_bam(args.bam)
    _ = open_fasta(args.fasta)  # not used yet, kept for parity/extension
    mito_chr = args.mito_chr if args.mito_chr in bam.references else ("MT" if "MT" in bam.references else None)
    if mito_chr is None:
        raise SystemExit("Mitochr not found")

    # cell -> pos -> UMI -> [A_count, C_count, G_count, T_count]
    store = defaultdict(lambda: defaultdict(dict))

    for rec in bam.fetch(mito_chr):
        if rec.is_secondary or rec.is_supplementary:
            continue
        if rec.mapping_quality < args.min_mapq:
            continue
        try:
            cb = rec.get_tag("CB")
            ub = rec.get_tag("UB")
        except KeyError:
            continue

        ref_pos = rec.get_reference_positions(full_length=False)
        quals = rec.query_qualities or []
        seq = rec.query_sequence
        if not seq or not ref_pos:
            continue

        for qpos, rpos in enumerate(ref_pos):
            if rpos is None or qpos >= len(quals) or quals[qpos] < args.min_baseq:
                continue
            base_idx = BASE_TO_IDX.get(seq[qpos])
            if base_idx is None:
                continue

            pos_dict = store[cb][rpos + 1]
            counts = pos_dict.get(ub)
            if counts is None:
                counts = [0, 0, 0, 0]
                pos_dict[ub] = counts
            counts[base_idx] += 1

    rows = []
    for cb, pos_dict in store.items():
        for pos, umi_dict in pos_dict.items():
            acgt = [0, 0, 0, 0]
            tot_umis = 0
            for counts in umi_dict.values():
                b, _, _ = consensus_base_from_counts(counts)
                if b is None:
                    continue
                acgt[BASE_TO_IDX[b]] += 1
                tot_umis += 1
            rows.append(dict(cell=cb, pos=pos, umi_total=tot_umis, A=acgt[0], C=acgt[1], G=acgt[2], T=acgt[3]))

    df = pd.DataFrame(rows)
    if df.empty:
        print("No data produced—check tags and filters.", file=sys.stderr)
    write_parquet(df, args.out)
    print(f"Wrote pileup to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

import argparse, sys
from collections import defaultdict, Counter
import numpy as np, pandas as pd
import pysam
from .io_utils import open_bam, open_fasta, write_parquet

def consensus_base(bases):
    # bases: iterable of (base, qual)
    if not bases: return None, 0, 0.0
    counts = Counter(b for b,q in bases if b in "ACGT")
    if not counts: return None, 0, 0.0
    base, n = counts.most_common(1)[0]
    tot = sum(counts.values())
    return base, n, n/max(1,tot)

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
    if mito_chr is None: raise SystemExit("Mitochr not found")

    # cell -> pos -> UMI -> [(base,qual)]
    store = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for rec in bam.fetch(mito_chr):
        if rec.is_secondary or rec.is_supplementary: continue
        if rec.mapping_quality < args.min_mapq: continue
        try:
            cb = rec.get_tag("CB"); ub = rec.get_tag("UB")
        except KeyError:
            continue
        ref_pos = rec.get_reference_positions(full_length=False)
        quals = rec.query_qualities or []
        seq = rec.query_sequence
        if not seq or not ref_pos: continue
        for qpos, rpos in enumerate(ref_pos):
            if rpos is None: continue
            if qpos >= len(quals): continue
            if quals[qpos] < args.min_baseq: continue
            base = seq[qpos]
            store[cb][rpos+1][ub].append((base, quals[qpos]))

    rows = []
    for cb, pos_dict in store.items():
        for pos, umi_dict in pos_dict.items():
            cons_counts = Counter()
            tot_umis = 0
            for ub, bases in umi_dict.items():
                b, n, frac = consensus_base(bases)
                if b is None: continue
                cons_counts[b]+=1
                tot_umis += 1
            acgt = [cons_counts.get(x,0) for x in "ACGT"]
            rows.append(dict(cell=cb, pos=pos, umi_total=tot_umis,
                             A=acgt[0], C=acgt[1], G=acgt[2], T=acgt[3]))
    df = pd.DataFrame(rows)
    if df.empty:
        print("No data produced—check tags and filters.", file=sys.stderr)
    write_parquet(df, args.out)
    print(f"Wrote pileup to {args.out}", file=sys.stderr)

if __name__ == "__main__":
    main()

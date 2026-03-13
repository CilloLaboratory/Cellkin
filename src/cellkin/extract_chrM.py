import argparse, sys, gzip
import pysam

def load_whitelist(path, strip_suffix=False):
    wl = set()
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        for line in f:
            b = line.strip()
            if not b:
                continue
            if strip_suffix and b.endswith(("-1", "-2", "-3", "-4", "-5")):
                b = b.rsplit("-", 1)[0]
            wl.add(b)
    return wl

def main():
    ap = argparse.ArgumentParser(description="Extract mitochondrial reads and optionally filter by a barcode whitelist.")
    ap.add_argument("--bam", required=True, help="Input coordinate-sorted BAM with CB/UB tags")
    ap.add_argument("--out", required=True, help="Output BAM (chrM-only, optionally barcode-filtered)")
    ap.add_argument("--fasta", required=True, help="Reference FASTA (used for validation; same build as BAM)")
    ap.add_argument("--mito-chr", default="chrM", help="Mitochondrial contig name (fallback to MT if absent)")
    ap.add_argument("--whitelist", default=None, help="Path to whitelist of cell barcodes (barcodes.tsv[.gz])")
    ap.add_argument("--strip-whitelist-suffix", action="store_true",
                    help="Strip trailing -N suffix from whitelist barcodes before matching (use if BAM CB lacks -1/-2 etc.)")
    ap.add_argument("--strip-bam-suffix", action="store_true",
                    help="Strip trailing -N suffix from BAM CB tag before matching (use if whitelist lacks suffixes).")
    args = ap.parse_args()

    # Open BAM and resolve mito contig
    inbam = pysam.AlignmentFile(args.bam, "rb")
    mito_chr = args.mito_chr if args.mito_chr in inbam.references else ("MT" if "MT" in inbam.references else None)
    if mito_chr is None:
        raise SystemExit(f"Mitochr {args.mito_chr} or MT not found in BAM.")

    # Load whitelist if provided
    whitelist = None
    if args.whitelist:
        whitelist = load_whitelist(args.whitelist, strip_suffix=args.strip_whitelist_suffix)
        if not whitelist:
            raise SystemExit(f"Whitelist {args.whitelist} loaded 0 barcodes—check path/format.")
        print(f"[extract_chrM] Loaded {len(whitelist):,} barcodes from whitelist.", file=sys.stderr)

    outbam = pysam.AlignmentFile(args.out, "wb", header=inbam.header)

    # Counters for logging
    n_total = n_chrM = n_cb_present = n_kept = 0
    seen_cb = set(); kept_cb = set()

    for rec in inbam.fetch(mito_chr):
        n_total += 1
        if rec.is_secondary or rec.is_supplementary:
            continue
        n_chrM += 1

        # Require tags
        try:
            cb = rec.get_tag("CB")
            _ = rec.get_tag("UB")  # ensure UMI exists; value unused here
        except KeyError:
            continue  # skip reads without CB/UB

        n_cb_present += 1
        seen_cb.add(cb)

        if whitelist is not None:
            cb_cmp = cb.rsplit("-", 1)[0] if (args.strip_bam_suffix and "-" in cb) else cb
            if cb_cmp not in whitelist:
                continue  # not in whitelist → drop
            kept_cb.add(cb)

        outbam.write(rec)
        n_kept += 1

    inbam.close(); outbam.close()
    pysam.index(args.out)

    # Summary
    print(
        "[extract_chrM] reads_total=%d  reads_chrM=%d  reads_with_CB=%d  reads_kept=%d  unique_CB_seen=%d  unique_CB_kept=%d"
        % (n_total, n_chrM, n_cb_present, n_kept, len(seen_cb), len(kept_cb) if whitelist is not None else len(seen_cb)),
        file=sys.stderr
    )
    print(f"Wrote chrM reads to {args.out}", file=sys.stderr)

if __name__ == "__main__":
    main()

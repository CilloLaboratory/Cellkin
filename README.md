# cellkin

`cellkin` reconstructs clonal structure and lineage trees from mitochondrial sequence variation in single-cell data.

## What it does

The package provides a command-line pipeline to:

1. Extract mitochondrial reads (`chrM`/`MT`) from a tagged BAM file.
2. Build a UMI-aware per-cell pileup.
3. Discover joint variant candidates.
4. Genotype each cell at candidate sites (VAF + discrete genotype call).
5. Cluster cells into clones and build clone phylogeny.
6. Build a QC report with plots and run summary.
7. Optionally build a cell-level neighbor-joining tree from VAF distances.

## Installation

### Python `venv` (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install .
```

### Development install

```bash
pip install -e .
```

If you previously used the conda workflow, activate your `venv` before running commands:

```bash
source .venv/bin/activate
```

## Command-line tools

After installation, these commands are available:

- `cellkin-extract-chrm`
- `cellkin-umi-pileup`
- `cellkin-variant-call`
- `cellkin-genotype`
- `cellkin-clone-phylogeny`
- `cellkin-build-nj`
- `cellkin-nj-prep`
- `cellkin-qc-report`

Inspect options with `--help`, for example:

```bash
cellkin-extract-chrm --help
```

## End-to-end usage

```bash
cellkin-extract-chrm \
  --bam your.bam \
  --fasta ref.fa \
  --out chrM.bam \
  --mito-chr chrM

# Optional: restrict to a barcode whitelist (e.g., cell type subset)
cellkin-extract-chrm \
  --bam your.bam \
  --fasta ref.fa \
  --out chrM.celltype.bam \
  --whitelist barcodes.tsv.gz

cellkin-umi-pileup \
  --bam chrM.bam \
  --fasta ref.fa \
  --out pileup.parquet

cellkin-variant-call \
  --pileup pileup.parquet \
  --out variants.parquet

cellkin-genotype \
  --pileup pileup.parquet \
  --variants variants.parquet \
  --out genotypes.parquet

# Optional but recommended: summarize per-cell NJ input quality and filter low-information cells
# Report-first mode (no filtering by default)
cellkin-nj-prep \
  --pileup pileup.parquet \
  --genotypes genotypes.parquet \
  --out-prefix njprep

# Filtered mode with explicit thresholds
cellkin-nj-prep \
  --pileup pileup.parquet \
  --genotypes genotypes.parquet \
  --out-prefix njprep \
  --min-cell-total-umi-depth 30 \
  --min-cell-called-variant-sites 10 \
  --min-cell-called-variant-fraction 0.2

# Output from this step includes a distance matrix and an nj tree
# Distance matrix can be used for coarse assessment of phylogeny
cellkin-build-nj \
  --genotypes njprep.genotypes.filtered.parquet \
  --out-prefix nj \
  --min-site-call-rate 0.7 \
  --min-cohort-vaf 0.05

# Optional: identification of clones
cellkin-clone-phylogeny \
  --genotypes genotypes.parquet \
  --out clones.tsv \
  --tree tree.newick

# Optional: report file summarizing clonal structure
cellkin-qc-report \
  --pileup pileup.parquet \
  --genotypes genotypes.parquet \
  --clones clones.tsv \
  --tree tree.newick \
  --out qc
```

Expected outputs include:

- `variants.parquet`: candidate mtDNA variants
- `genotypes.parquet`: per-cell genotypes and VAF
- `njprep.cell_qc.tsv`: per-cell depth/callability QC summary (`cellkin-nj-prep`)
- `njprep.genotypes.filtered.parquet`: filtered genotypes for NJ input (`cellkin-nj-prep`)
- `clones.tsv`: clone membership summaries
- `tree.newick`: clone tree
- `nj.distance.csv` or `nj.distance.condensed.csv`: cell-level pairwise distance matrix (`cellkin-build-nj`)
- `nj.tree.newick`: cell-level NJ tree (`cellkin-build-nj`)
- `qc/index.html`: QC report

## Scaling options

For larger cohorts, two optional guardrails/output modes are available:

```bash
# Write pairwise distances in long/condensed form (smaller than full NxN matrix)
cellkin-build-nj \
  --genotypes genotypes.parquet \
  --out-prefix nj \
  --distance-format condensed

# Large-scale mode: stream condensed distances without full NxN materialization
cellkin-build-nj \
  --genotypes genotypes.parquet \
  --out-prefix nj \
  --distance-format condensed \
  --large-scale-mode \
  --block-size 32

# Prevent accidental very-large agglomerative clustering
cellkin-clone-phylogeny \
  --genotypes genotypes.parquet \
  --out clones.tsv \
  --tree tree.newick \
  --max-cells-for-clustering 20000
```

`--distance-format square` remains the default behavior.
`--max-cells-for-clustering 0` (default) disables the guardrail.

For `cellkin-build-nj`, a preflight memory check runs before matrix construction.
You can set an explicit budget with `--max-memory-gb` to force a hard stop before OOM:

```bash
cellkin-build-nj \
  --genotypes genotypes.parquet \
  --out-prefix nj \
  --max-memory-gb 64
```

In `--large-scale-mode`, NJ tree generation is skipped and a placeholder tree (`();`) is written.

## Variant filtering

To distances problems induced by sparse coverage, filter to cohort-represented sites:

- `--min-site-call-rate`: minimum fraction of cells with callable depth at a site
- `--min-site-cells`: minimum number of cells with callable depth
- `--min-cohort-vaf` / `--max-cohort-vaf`: keep sites in a cohort VAF range

These filters are applied before distance computation and written to `*.vaf_matrix.parquet`.

You can also enforce a hard QC safeguard on pairwise overlap:

- `--max-no-overlap-fraction`: abort if too many cell pairs have no overlapping callable sites

Example:

```bash
cellkin-build-nj \
  --genotypes genotypes.parquet \
  --out-prefix nj \
  --min-depth-for-call 3 \
  --min-depth-pair 3 \
  --min-site-call-rate 0.7 \
  --max-no-overlap-fraction 0.05
```

## NJ input QC and filtering

`cellkin-nj-prep` computes per-cell metrics from `pileup` + `genotypes` and can filter cells before `cellkin-build-nj`.

Per-cell QC columns:

- `cell`
- `total_umi_depth`
- `n_positions_with_coverage`
- `n_variant_sites_called`
- `fraction_variant_sites_called`
- `mean_variant_depth`
- `mean_variant_vaf`
- `n_nonmissing_gt`

Cell filtering thresholds (all optional, logical AND):

- `--min-cell-total-umi-depth`
- `--min-cell-covered-positions`
- `--min-cell-called-variant-sites`
- `--min-cell-called-variant-fraction`
- `--min-cell-mean-variant-depth`
- `--call-depth-for-metrics`

## Using a cell barcode whitelist

`cellkin-extract-chrm` supports barcode filtering directly:

- `--whitelist <path>`: one barcode per line (plain text or `.gz`, e.g. `barcodes.tsv.gz`)
- `--strip-whitelist-suffix`: strips `-1`, `-2`, etc. from whitelist entries before matching
- `--strip-bam-suffix`: strips `-1`, `-2`, etc. from BAM `CB` tags before matching

Use these suffix flags when one side includes 10x-style suffixes and the other does not.

Examples:

```bash
# Whitelist has AAAC...-1, BAM CB has AAAC...-1
cellkin-extract-chrm --bam in.bam --fasta ref.fa --out chrM.bam --whitelist barcodes.tsv.gz

# Whitelist has AAAC..., BAM CB has AAAC...-1
cellkin-extract-chrm --bam in.bam --fasta ref.fa --out chrM.bam \
  --whitelist barcodes.tsv --strip-bam-suffix

# Whitelist has AAAC...-1, BAM CB has AAAC...
cellkin-extract-chrm --bam in.bam --fasta ref.fa --out chrM.bam \
  --whitelist barcodes.tsv.gz --strip-whitelist-suffix
```

## Benchmark harness

A synthetic benchmark harness is available at:

- `tests/benchmarks/benchmark_pipeline.py`

Example:

```bash
python tests/benchmarks/benchmark_pipeline.py --cells 1000 --sites 200 --out-json benchmark.json
```

To compare against a previous baseline run:

```bash
python tests/benchmarks/benchmark_pipeline.py \
  --cells 1000 --sites 200 \
  --baseline-json baseline.json \
  --out-json current.json
```

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

### Conda environment (recommended)

```bash
conda env create -f environment.yml
conda activate cellkin
pip install .
```

### Pip only

```bash
pip install .
```

For development installs:

```bash
pip install -e .
```

## Command-line tools

After installation, these commands are available:

- `cellkin-extract-chrm`
- `cellkin-umi-pileup`
- `cellkin-variant-call`
- `cellkin-genotype`
- `cellkin-clone-phylogeny`
- `cellkin-build-nj`
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

# Output from this step includes a distance matrix and an nj tree
# Distance matrix can be used for coarse assessment of phylogeny
cellkin-build-nj \
  --genotypes genotypes.parquet \
  --out-prefix nj

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
- `clones.tsv`: clone membership summaries
- `tree.newick`: clone tree
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

# Prevent accidental very-large agglomerative clustering
cellkin-clone-phylogeny \
  --genotypes genotypes.parquet \
  --out clones.tsv \
  --tree tree.newick \
  --max-cells-for-clustering 20000
```

`--distance-format square` remains the default behavior.
`--max-cells-for-clustering 0` (default) disables the guardrail.

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

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

cellkin-clone-phylogeny \
  --genotypes genotypes.parquet \
  --out clones.tsv \
  --tree tree.newick

cellkin-build-nj \
  --genotypes genotypes.parquet \
  --out-prefix nj

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

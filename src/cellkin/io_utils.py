from typing import Iterator
import pandas as pd
import pysam

def iter_chrM_records(bam_path: str, mito_chr: str="chrM"):
    bam = pysam.AlignmentFile(bam_path, "rb")
    if mito_chr not in bam.references:
        mito_chr = "MT" if "MT" in bam.references else None
    if mito_chr is None:
        raise ValueError("Mitochondrial chromosome not found in BAM references.")
    for rec in bam.fetch(mito_chr):
        yield rec
    bam.close()

def open_bam(path: str): return pysam.AlignmentFile(path, "rb")
def open_fasta(path: str): return pysam.FastaFile(path)
def write_parquet(df: pd.DataFrame, path: str): df.to_parquet(path, index=False)
def read_parquet(path: str) -> pd.DataFrame: return pd.read_parquet(path)

from pathlib import Path

import pandas as pd
import pysam

from cellkin.umi_pileup import consensus_base_from_counts, main as umi_pileup_main


def _write_fasta(path: Path):
    path.write_text(">chrM\n" + ("A" * 100) + "\n")


def _make_read(name: str, seq: str, cb: str, ub: str, start: int = 10):
    read = pysam.AlignedSegment()
    read.query_name = name
    read.query_sequence = seq
    read.flag = 0
    read.reference_id = 0
    read.reference_start = start
    read.mapping_quality = 60
    read.cigar = ((0, len(seq)),)
    read.query_qualities = pysam.qualitystring_to_array("I" * len(seq))
    read.set_tag("CB", cb)
    read.set_tag("UB", ub)
    return read


def test_consensus_tie_break_is_deterministic():
    base, n, frac = consensus_base_from_counts([1, 1, 0, 0])
    assert base == "A"
    assert n == 1
    assert frac == 0.5


def test_umi_pileup_expected_counts(tmp_path, monkeypatch):
    bam_path = tmp_path / "in.bam"
    fasta = tmp_path / "ref.fa"
    out = tmp_path / "pileup.parquet"

    _write_fasta(fasta)
    header = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "chrM", "LN": 100}]}

    with pysam.AlignmentFile(bam_path, "wb", header=header) as bam:
        bam.write(_make_read("r1", "A", "CELL1-1", "UMI1", start=10))
        bam.write(_make_read("r2", "C", "CELL1-1", "UMI1", start=10))
        bam.write(_make_read("r3", "C", "CELL1-1", "UMI2", start=10))
        bam.write(_make_read("r4", "C", "CELL1-1", "UMI2", start=10))
    pysam.index(str(bam_path))

    monkeypatch.setattr(
        "sys.argv",
        [
            "umi_pileup.py",
            "--bam",
            str(bam_path),
            "--fasta",
            str(fasta),
            "--out",
            str(out),
        ],
    )
    umi_pileup_main()

    df = pd.read_parquet(out)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["cell"] == "CELL1-1"
    assert int(row["pos"]) == 11
    assert int(row["umi_total"]) == 2
    assert int(row["A"]) == 1
    assert int(row["C"]) == 1
    assert int(row["G"]) == 0
    assert int(row["T"]) == 0

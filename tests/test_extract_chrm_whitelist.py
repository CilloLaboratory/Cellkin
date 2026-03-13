from pathlib import Path

import pysam

from cellkin.extract_chrM import main as extract_chrm_main


def _write_fasta(path: Path) -> None:
    path.write_text(">chrM\n" + ("A" * 1000) + "\n")


def _make_read(name: str, cb: str, ub: str, start: int) -> pysam.AlignedSegment:
    read = pysam.AlignedSegment()
    read.query_name = name
    read.query_sequence = "A" * 50
    read.flag = 0
    read.reference_id = 0
    read.reference_start = start
    read.mapping_quality = 60
    read.cigar = ((0, 50),)  # 50M
    read.query_qualities = pysam.qualitystring_to_array("I" * 50)
    read.set_tag("CB", cb)
    read.set_tag("UB", ub)
    return read


def _write_bam(path: Path) -> None:
    header = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "chrM", "LN": 1000}]}
    with pysam.AlignmentFile(path, "wb", header=header) as outbam:
        outbam.write(_make_read("r1", "CELL_A-1", "UMI1", 10))
        outbam.write(_make_read("r2", "CELL_B-1", "UMI2", 20))
        outbam.write(_make_read("r3", "CELL_B-1", "UMI3", 30))
    pysam.index(str(path))


def _count_cb(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with pysam.AlignmentFile(path, "rb") as bam:
        for rec in bam.fetch("chrM"):
            cb = rec.get_tag("CB")
            counts[cb] = counts.get(cb, 0) + 1
    return counts


def test_extract_chrm_whitelist_filters_and_suffix_handling(tmp_path, monkeypatch):
    in_bam = tmp_path / "input.bam"
    out_bam = tmp_path / "out.bam"
    fasta = tmp_path / "ref.fa"
    whitelist = tmp_path / "barcodes.tsv"

    _write_fasta(fasta)
    _write_bam(in_bam)
    whitelist.write_text("CELL_B\n")

    monkeypatch.setattr(
        "sys.argv",
        [
            "extract_chrM.py",
            "--bam",
            str(in_bam),
            "--out",
            str(out_bam),
            "--fasta",
            str(fasta),
            "--whitelist",
            str(whitelist),
            "--strip-bam-suffix",
        ],
    )
    extract_chrm_main()

    counts = _count_cb(out_bam)
    assert counts == {"CELL_B-1": 2}

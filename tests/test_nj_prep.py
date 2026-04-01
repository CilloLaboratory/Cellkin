import numpy as np
import pandas as pd
import pytest

from cellkin.build_nj import main as build_nj_main
from cellkin.nj_prep import compute_cell_qc, main as nj_prep_main


def _make_fixture():
    pileup = pd.DataFrame(
        [
            {"cell": "c1", "pos": 100, "umi_total": 10, "A": 8, "C": 1, "G": 1, "T": 0},
            {"cell": "c1", "pos": 200, "umi_total": 5, "A": 5, "C": 0, "G": 0, "T": 0},
            {"cell": "c2", "pos": 100, "umi_total": 2, "A": 1, "C": 0, "G": 1, "T": 0},
            {"cell": "c2", "pos": 200, "umi_total": 0, "A": 0, "C": 0, "G": 0, "T": 0},
            {"cell": "c3", "pos": 100, "umi_total": 0, "A": 0, "C": 0, "G": 0, "T": 0},
            {"cell": "c3", "pos": 200, "umi_total": 1, "A": 1, "C": 0, "G": 0, "T": 0},
        ]
    )
    genotypes = pd.DataFrame(
        [
            {"cell": "c1", "pos": 100, "ref": "A", "alt": "G", "ref_umi": 8, "alt_umi": 1, "depth": 10, "vaf": 0.1, "gt": 1},
            {"cell": "c1", "pos": 200, "ref": "A", "alt": "T", "ref_umi": 5, "alt_umi": 0, "depth": 5, "vaf": 0.0, "gt": 0},
            {"cell": "c2", "pos": 100, "ref": "A", "alt": "G", "ref_umi": 1, "alt_umi": 1, "depth": 2, "vaf": 0.5, "gt": 1},
            {"cell": "c2", "pos": 200, "ref": "A", "alt": "T", "ref_umi": 0, "alt_umi": 0, "depth": 0, "vaf": np.nan, "gt": 9},
            {"cell": "c3", "pos": 100, "ref": "A", "alt": "G", "ref_umi": 0, "alt_umi": 0, "depth": 0, "vaf": np.nan, "gt": 9},
            {"cell": "c3", "pos": 200, "ref": "A", "alt": "T", "ref_umi": 1, "alt_umi": 0, "depth": 1, "vaf": 0.0, "gt": 0},
        ]
    )
    return pileup, genotypes


def _run_prep(tmp_path, monkeypatch, extra_args=None):
    pileup, genotypes = _make_fixture()
    ppath = tmp_path / "pileup.parquet"
    gpath = tmp_path / "genotypes.parquet"
    out_prefix = tmp_path / "prep"
    pileup.to_parquet(ppath, index=False)
    genotypes.to_parquet(gpath, index=False)

    argv = ["nj_prep.py", "--pileup", str(ppath), "--genotypes", str(gpath), "--out-prefix", str(out_prefix)]
    if extra_args:
        argv.extend(extra_args)

    monkeypatch.setattr("sys.argv", argv)
    nj_prep_main()
    return out_prefix


def test_compute_cell_qc_values():
    pileup, genotypes = _make_fixture()
    qc = compute_cell_qc(pileup, genotypes, call_depth_for_metrics=1).set_index("cell")

    assert int(qc.loc["c1", "total_umi_depth"]) == 15
    assert int(qc.loc["c1", "n_positions_with_coverage"]) == 2
    assert int(qc.loc["c1", "n_variant_sites_called"]) == 2
    assert np.isclose(float(qc.loc["c1", "fraction_variant_sites_called"]), 1.0)

    assert int(qc.loc["c2", "total_umi_depth"]) == 2
    assert int(qc.loc["c2", "n_positions_with_coverage"]) == 1
    assert int(qc.loc["c2", "n_variant_sites_called"]) == 1
    assert np.isclose(float(qc.loc["c2", "fraction_variant_sites_called"]), 0.5)


def test_nj_prep_no_thresholds_keeps_all_and_output_contract(tmp_path, monkeypatch):
    out_prefix = _run_prep(tmp_path, monkeypatch)

    qc = pd.read_csv(tmp_path / "prep.cell_qc.tsv", sep="\t")
    filt = pd.read_parquet(tmp_path / "prep.genotypes.filtered.parquet")

    assert list(qc.columns) == [
        "cell",
        "total_umi_depth",
        "n_positions_with_coverage",
        "n_variant_sites_called",
        "fraction_variant_sites_called",
        "mean_variant_depth",
        "mean_variant_vaf",
        "n_nonmissing_gt",
    ]
    assert len(qc) == 3
    assert set(filt["cell"]) == {"c1", "c2", "c3"}


@pytest.mark.parametrize(
    "extra_args,expected_cells",
    [
        (["--min-cell-total-umi-depth", "3"], {"c1"}),
        (["--min-cell-covered-positions", "2"], {"c1"}),
        (["--min-cell-called-variant-sites", "2"], {"c1"}),
        (["--min-cell-called-variant-fraction", "1.0"], {"c1"}),
        (["--min-cell-mean-variant-depth", "3"], {"c1"}),
    ],
)
def test_nj_prep_individual_threshold_filters(tmp_path, monkeypatch, extra_args, expected_cells):
    _run_prep(tmp_path, monkeypatch, extra_args=extra_args)
    filt = pd.read_parquet(tmp_path / "prep.genotypes.filtered.parquet")
    assert set(filt["cell"]) == expected_cells


def test_nj_prep_combined_thresholds_filter_cells(tmp_path, monkeypatch):
    _run_prep(
        tmp_path,
        monkeypatch,
        extra_args=[
            "--min-cell-total-umi-depth",
            "3",
            "--min-cell-called-variant-sites",
            "2",
            "--min-cell-called-variant-fraction",
            "1",
            "--min-cell-mean-variant-depth",
            "5",
        ],
    )

    filt = pd.read_parquet(tmp_path / "prep.genotypes.filtered.parquet")
    assert set(filt["cell"]) == {"c1"}


def test_nj_prep_end_to_end_with_build_nj(tmp_path, monkeypatch):
    _run_prep(tmp_path, monkeypatch, extra_args=["--min-cell-total-umi-depth", "1"])

    monkeypatch.setattr(
        "sys.argv",
        [
            "build_nj.py",
            "--genotypes",
            str(tmp_path / "prep.genotypes.filtered.parquet"),
            "--out-prefix",
            str(tmp_path / "nj"),
            "--distance-format",
            "condensed",
        ],
    )
    build_nj_main()

    assert (tmp_path / "nj.distance.condensed.csv").exists()
    assert (tmp_path / "nj.tree.newick").exists()

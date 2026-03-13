import numpy as np
import pandas as pd

from cellkin.genotype import _call_genotype_vectorized, call_genotype, main as genotype_main


def test_vectorized_call_matches_scalar():
    vaf = np.array([np.nan, 0.0, 0.01, 0.5, 0.99])
    depth = np.array([0, 10, 100, 20, 100])

    got = _call_genotype_vectorized(vaf, depth, 0.05, 0.85)
    expected = np.array([call_genotype(v, d, 0.05, 0.85) for v, d in zip(vaf, depth)])
    np.testing.assert_array_equal(got, expected)


def test_genotype_pipeline_output(tmp_path, monkeypatch):
    pileup = pd.DataFrame(
        [
            {"cell": "c1", "pos": 100, "umi_total": 10, "A": 9, "C": 0, "G": 1, "T": 0},
            {"cell": "c2", "pos": 100, "umi_total": 10, "A": 1, "C": 0, "G": 9, "T": 0},
        ]
    )
    variants = pd.DataFrame([{"pos": 100, "ref": "A", "alt": "G", "n_cells_alt": 2, "tot_umi": 20}])

    pileup_path = tmp_path / "pileup.parquet"
    var_path = tmp_path / "variants.parquet"
    out_path = tmp_path / "genotypes.parquet"
    pileup.to_parquet(pileup_path, index=False)
    variants.to_parquet(var_path, index=False)

    monkeypatch.setattr(
        "sys.argv",
        [
            "genotype.py",
            "--pileup",
            str(pileup_path),
            "--variants",
            str(var_path),
            "--out",
            str(out_path),
        ],
    )
    genotype_main()

    out = pd.read_parquet(out_path).sort_values("cell").reset_index(drop=True)
    assert out.loc[0, "ref_umi"] == 9
    assert out.loc[0, "alt_umi"] == 1
    assert out.loc[0, "depth"] == 10
    assert np.isclose(out.loc[0, "vaf"], 0.1)
    assert out.loc[0, "gt"] == 1

    assert out.loc[1, "ref_umi"] == 1
    assert out.loc[1, "alt_umi"] == 9
    assert out.loc[1, "depth"] == 10
    assert np.isclose(out.loc[1, "vaf"], 0.9)
    assert out.loc[1, "gt"] == 1

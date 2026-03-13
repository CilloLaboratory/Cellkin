import pandas as pd

from cellkin.variant_call import main as variant_call_main


def test_variant_call_candidates(tmp_path, monkeypatch):
    pileup = pd.DataFrame(
        [
            {"cell": "c1", "pos": 100, "umi_total": 3, "A": 3, "C": 0, "G": 0, "T": 0},
            {"cell": "c2", "pos": 100, "umi_total": 3, "A": 2, "C": 1, "G": 0, "T": 0},
            {"cell": "c3", "pos": 100, "umi_total": 3, "A": 2, "C": 1, "G": 0, "T": 0},
            {"cell": "c1", "pos": 200, "umi_total": 2, "A": 0, "C": 0, "G": 2, "T": 0},
            {"cell": "c2", "pos": 200, "umi_total": 2, "A": 0, "C": 0, "G": 2, "T": 0},
        ]
    )

    pileup_path = tmp_path / "pileup.parquet"
    out_path = tmp_path / "variants.parquet"
    pileup.to_parquet(pileup_path, index=False)

    monkeypatch.setattr(
        "sys.argv",
        [
            "variant_call.py",
            "--pileup",
            str(pileup_path),
            "--out",
            str(out_path),
            "--min-site-depth",
            "5",
            "--min-alt-cells",
            "2",
        ],
    )
    variant_call_main()

    out = pd.read_parquet(out_path).sort_values(["pos", "alt"]).reset_index(drop=True)
    expected = pd.DataFrame(
        [{"pos": 100, "ref": "A", "alt": "C", "n_cells_alt": 2, "tot_umi": 9}]
    )
    pd.testing.assert_frame_equal(out, expected)

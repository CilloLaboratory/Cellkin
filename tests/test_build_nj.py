import pandas as pd

from cellkin.build_nj import main as build_nj_main


def test_build_nj_condensed_distance_output(tmp_path, monkeypatch):
    genotypes = pd.DataFrame(
        [
            {"cell": "c1", "pos": 100, "alt": "G", "vaf": 0.1, "depth": 10},
            {"cell": "c2", "pos": 100, "alt": "G", "vaf": 0.8, "depth": 10},
            {"cell": "c1", "pos": 200, "alt": "T", "vaf": 0.2, "depth": 10},
            {"cell": "c2", "pos": 200, "alt": "T", "vaf": 0.7, "depth": 10},
        ]
    )

    gpath = tmp_path / "genotypes.parquet"
    out_prefix = tmp_path / "nj"
    genotypes.to_parquet(gpath, index=False)

    monkeypatch.setattr(
        "sys.argv",
        [
            "build_nj.py",
            "--genotypes",
            str(gpath),
            "--out-prefix",
            str(out_prefix),
            "--distance-format",
            "condensed",
        ],
    )
    build_nj_main()

    condensed = pd.read_csv(tmp_path / "nj.distance.condensed.csv")
    assert list(condensed.columns) == ["cell_i", "cell_j", "distance"]
    assert len(condensed) == 1
    assert (tmp_path / "nj.tree.newick").exists()
    assert (tmp_path / "nj.vaf_matrix.parquet").exists()

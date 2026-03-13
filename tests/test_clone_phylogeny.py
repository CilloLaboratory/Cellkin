import pandas as pd
import pytest

from cellkin.clone_phylogeny import main as clone_phylogeny_main


def _example_genotypes():
    return pd.DataFrame(
        [
            {"cell": "c1", "pos": 100, "alt": "G", "gt": 0},
            {"cell": "c2", "pos": 100, "alt": "G", "gt": 0},
            {"cell": "c3", "pos": 100, "alt": "G", "gt": 2},
            {"cell": "c1", "pos": 200, "alt": "T", "gt": 0},
            {"cell": "c2", "pos": 200, "alt": "T", "gt": 0},
            {"cell": "c3", "pos": 200, "alt": "T", "gt": 2},
        ]
    )


def test_clone_phylogeny_guardrail(tmp_path, monkeypatch):
    gpath = tmp_path / "genotypes.parquet"
    out = tmp_path / "clones.tsv"
    tree = tmp_path / "tree.newick"
    _example_genotypes().to_parquet(gpath, index=False)

    monkeypatch.setattr(
        "sys.argv",
        [
            "clone_phylogeny.py",
            "--genotypes",
            str(gpath),
            "--out",
            str(out),
            "--tree",
            str(tree),
            "--max-cells-for-clustering",
            "2",
        ],
    )

    with pytest.raises(SystemExit, match="Refusing to cluster"):
        clone_phylogeny_main()


def test_clone_phylogeny_runs(tmp_path, monkeypatch):
    gpath = tmp_path / "genotypes.parquet"
    out = tmp_path / "clones.tsv"
    tree = tmp_path / "tree.newick"
    _example_genotypes().to_parquet(gpath, index=False)

    monkeypatch.setattr(
        "sys.argv",
        [
            "clone_phylogeny.py",
            "--genotypes",
            str(gpath),
            "--out",
            str(out),
            "--tree",
            str(tree),
            "--min_cells_per_clone",
            "1",
        ],
    )
    clone_phylogeny_main()

    clones = pd.read_csv(out, sep="\t")
    assert {"clone_id", "n_cells", "cells", "defining_variants"}.issubset(clones.columns)
    assert tree.read_text().strip().endswith(";")

"""Exercise the real mental-health builder's legacy and later-cycle CLI paths."""

from pathlib import Path

import pandas as pd
import pytest

from src.data import mental


@pytest.mark.parametrize("arguments,suffix", [([], "J"), (["2017-2018"], "J"), (["2021-2023"], "L")])
def test_cli_selects_cycle_without_changing_processing(arguments, suffix, monkeypatch, tmp_path):
    monkeypatch.setattr(mental, "__file__", str(tmp_path / "src/data/mental.py"))
    paths = []

    def read_sas(path):
        paths.append(Path(path))
        return pd.DataFrame({"SEQN": [1.0, 2.0], **{column: [1.0, 0.0] for column in mental.PHQ9_COLUMNS}})

    monkeypatch.setattr(pd, "read_sas", read_sas)
    mental.main(arguments)
    cycle = "2021-2023" if suffix == "L" else "2017-2018"
    assert paths == [tmp_path / f"data/raw/nhanes/{cycle}/DPQ_{suffix}.xpt"]
    table = pd.read_csv(tmp_path / f"data/interim/mental_{cycle}.csv")
    assert table.phq9_score.tolist() == [9.0, 0.0]
    assert table.phq9_category.tolist() == ["mild", "minimal"]

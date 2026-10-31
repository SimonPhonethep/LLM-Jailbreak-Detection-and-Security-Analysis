from pathlib import Path

import pandas as pd
import pytest

from jailbreak_classifier.data import load_labeled_csv, split_data, text_groups


def test_original_ground_truth_labels():
    p = Path(__file__).resolve().parents[1] / "data/eval/ground_truth_60.csv"
    df = load_labeled_csv(p)
    assert len(df) == 60
    assert df["label"].value_counts().to_dict() == {0: 20, 1: 20, 2: 20}


def test_string_label_conversion(tmp_path):
    p = tmp_path / "a.csv"
    pd.DataFrame({"text": ["No", "Yes", "Maybe"],
                  "label": ["Refused", "Harmful", "Incomplete"]}).to_csv(p, index=False)
    assert load_labeled_csv(p)["label"].tolist() == [0, 1, 2]


def test_no_identical_text_across_splits():
    texts = [f"response {i}" for i in range(60)]
    df = pd.DataFrame({"text": texts + texts[:10],
                       "label": [i % 3 for i in range(60)] + [i % 3 for i in range(10)]})
    for dedup in (True, False):
        a, b = split_data(df, deduplicate=dedup)
        assert not set(text_groups(a)) & set(text_groups(b))
        assert len(a) and len(b)


def test_conflicting_labels_rejected():
    df = pd.DataFrame({"text": ["same", "same"], "label": [0, 1]})
    with pytest.raises(ValueError, match="contradictory"):
        split_data(df)

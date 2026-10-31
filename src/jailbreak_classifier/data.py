"""Read labeled CSV datasets and create a leakage-conscious train/validation split."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from . import LABEL2ID


def load_labeled_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df = df[["text", "label"]].copy()
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df.loc[df["text"].ne("")].copy()

    def normalize_label(value):
        if isinstance(value, str):
            if value in LABEL2ID:
                return LABEL2ID[value]
            if value.strip() in {"0", "1", "2"}:
                return int(value.strip())
        if value in (0, 1, 2):
            return int(value)
        raise ValueError(f"Unrecognized label: {value!r}")

    df["label"] = df["label"].map(normalize_label)
    return df.reset_index(drop=True)


def text_groups(df: pd.DataFrame) -> pd.Series:
    return df["text"].str.casefold().str.replace(r"\s+", " ", regex=True).str.strip()


def split_data(df: pd.DataFrame, *, seed: int = 42,
               validation_size: float = 0.1, deduplicate: bool = True):
    """Avoid exact/whitespace/case-normalized response leakage across splits.

    With deduplication on (default), use a stratified split. When duplicate
    rows are retained, grouped splitting keeps identical responses together.
    """
    df = df.copy()
    key = text_groups(df)
    conflicts = df.assign(_key=key).groupby("_key")["label"].nunique()
    if (conflicts > 1).any():
        raise ValueError("The same response text has contradictory labels")

    if deduplicate:
        df = df.loc[~key.duplicated()].reset_index(drop=True)
        stratify = df["label"] if df["label"].value_counts().min() >= 2 else None
        tr, val = train_test_split(df, test_size=validation_size,
                                   random_state=seed, stratify=stratify)
    else:
        groups = text_groups(df)
        splitter = GroupShuffleSplit(n_splits=1, test_size=validation_size,
                                     random_state=seed)
        tr_idx, val_idx = next(splitter.split(df, groups=groups))
        tr, val = df.iloc[tr_idx], df.iloc[val_idx]

    train_keys = set(text_groups(tr))
    validation_keys = set(text_groups(val))
    if train_keys & validation_keys:
        raise AssertionError("Leakage detected between training and validation")
    return tr.reset_index(drop=True), val.reset_index(drop=True)

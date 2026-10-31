"""Utilities for three-class classification of LLM-generated responses."""

LABEL2ID = {"Refused": 0, "Harmful": 1, "Incomplete": 2}
ID2LABEL = {number: label for label, number in LABEL2ID.items()}

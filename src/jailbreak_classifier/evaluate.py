"""Evaluate an available fine-tuned checkpoint against a labeled held-out CSV."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from . import ID2LABEL
from .data import load_labeled_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data", default="data/eval/ground_truth_60.csv")
    parser.add_argument("--output-dir", default="results/local_evaluation")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    df = load_labeled_csv(args.data)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    predictions = []
    with torch.inference_mode():
        for i in range(0, len(df), args.batch_size):
            batch = tokenizer(df["text"].iloc[i:i + args.batch_size].tolist(),
                              truncation=True, padding=True, max_length=512,
                              return_tensors="pt").to(device)
            predictions.extend(model(**batch).logits.argmax(dim=-1).cpu().tolist())

    gold = df["label"].tolist()
    ids = [0, 1, 2]
    names = [ID2LABEL[i] for i in ids]
    metrics = {
        "model_path": args.model_path,
        "data_path": args.data,
        "samples": len(df),
        "accuracy": float(accuracy_score(gold, predictions)),
        "macro_f1": float(f1_score(gold, predictions, labels=ids, average="macro", zero_division=0)),
        "per_class": classification_report(gold, predictions, labels=ids,
                                            target_names=names, output_dict=True,
                                            zero_division=0),
    }
    dest = Path(args.output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "metrics.json").write_text(json.dumps(metrics, indent=2))
    df.assign(predicted_label=predictions,
              predicted_class=[ID2LABEL[p] for p in predictions]).to_csv(
                  dest / "predictions.csv", index=False)

    matrix = confusion_matrix(gold, predictions, labels=ids)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(3), names)
    ax.set_yticks(range(3), names)
    ax.set(xlabel="Predicted class", ylabel="Actual class", title="Confusion matrix")
    for r in range(3):
        for c in range(3):
            ax.text(c, r, str(matrix[r, c]), ha="center", va="center",
                    color="white" if matrix[r, c] > matrix.max()/2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(dest / "confusion_matrix.png", dpi=180)
    plt.close(fig)
    print(json.dumps({k: metrics[k] for k in ("samples", "accuracy", "macro_f1")}, indent=2))
    print(f"Saved evaluation files to: {dest}")


if __name__ == "__main__":
    main()

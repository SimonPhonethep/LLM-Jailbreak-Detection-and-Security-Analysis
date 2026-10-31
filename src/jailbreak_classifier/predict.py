"""Predict one response or a CSV with a `text` column."""
import argparse
import json
from pathlib import Path

import pandas as pd

from . import ID2LABEL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    args_group = parser.add_mutually_exclusive_group(required=True)
    args_group.add_argument("--text", help="One response to classify")
    args_group.add_argument("--input-csv", help="CSV file containing a text column")
    parser.add_argument("--output-csv", default="results/local_predictions.csv")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    if args.text:
        texts = [args.text]
    else:
        df = pd.read_csv(args.input_csv)
        if "text" not in df.columns:
            raise ValueError("The input CSV must include a 'text' column")
        texts = df["text"].fillna("").astype(str).tolist()
    results = []
    with torch.inference_mode():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            probs = torch.softmax(model(**enc).logits[0], dim=-1).cpu().tolist()
            idx = max(range(3), key=probs.__getitem__)
            results.append({"text": text, "prediction": ID2LABEL[idx],
                            "confidence": round(probs[idx], 4),
                            **{f"p_{ID2LABEL[i].lower()}": round(probs[i], 4)
                               for i in range(3)}})
    if args.text:
        print(json.dumps(results[0], indent=2, ensure_ascii=False))
    else:
        dest = Path(args.output_csv)
        dest.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(dest, index=False)
        print(f"Saved {len(results)} predictions to {dest}")


if __name__ == "__main__":
    main()

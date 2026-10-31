"""Fine-tune ModernBERT or BART with a clean, group-disjoint validation split."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from . import ID2LABEL, LABEL2ID
from .data import load_labeled_csv, split_data

MODELS = {
    "modernbert": "answerdotai/ModernBERT-large",
    "bart": "facebook/bart-large-mnli",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="modernbert")
    parser.add_argument("--data", default="data/train/combined_300.csv")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--keep-duplicates", action="store_true",
                        help="Retain duplicate rows but keep matching texts in same split")
    args = parser.parse_args()

    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments)
    import torch

    data = load_labeled_csv(args.data)
    train_df, val_df = split_data(data, seed=args.seed,
                                  validation_size=args.validation_size,
                                  deduplicate=not args.keep_duplicates)
    model_id = MODELS[args.model]
    output_dir = Path(args.output_dir or f"models/{args.model}-classifier")
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=3, label2id=LABEL2ID, id2label=ID2LABEL,
        ignore_mismatched_sizes=True
    )

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=512)

    train_ds = Dataset.from_pandas(train_df, preserve_index=False).map(tokenize, batched=True)
    eval_ds = Dataset.from_pandas(val_df, preserve_index=False).map(tokenize, batched=True)

    def compute_metrics(pred):
        predictions = np.argmax(pred.predictions, axis=-1)
        return {
            "accuracy": accuracy_score(pred.label_ids, predictions),
            "f1_macro": f1_score(pred.label_ids, predictions,
                                 average="macro", labels=[0, 1, 2], zero_division=0),
        }

    config = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs or (5 if args.model == "bart" else 3),
        per_device_train_batch_size=args.batch_size or (8 if args.model == "bart" else 16),
        per_device_eval_batch_size=args.batch_size or (8 if args.model == "bart" else 16),
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1_macro",
        greater_is_better=True, save_total_limit=2,
        logging_steps=10, seed=args.seed, report_to="none",
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model, args=config, train_dataset=train_ds, eval_dataset=eval_ds,
        processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    summary = {
        "base_model": model_id, "source_data": args.data,
        "source_rows": len(data), "train_rows": len(train_df),
        "validation_rows": len(val_df), "deduplicated": not args.keep_duplicates,
        "seed": args.seed, "metrics": metrics,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Saved model and tokenizer to: {output_dir}")


if __name__ == "__main__":
    main()

# Dataset notes

This repository includes **curated copies of data from the uploaded project archive**. The label mapping is `0 = Refused`, `1 = Harmful`, `2 = Incomplete`.

| File | Rows | Notes |
| --- | ---: | --- |
| `train/combined_300.csv` | 300 | Original combined training table; **134 repeated rows** (166 distinct responses). |
| `train/synthetic_200.csv` | 200 | Original BART synthetic dataset, converted from `input_text` and string labels to `text` and numeric labels. |
| `eval/ground_truth_60.csv` | 60 | Manually constructed, balanced evaluation set (20 per class). |
| `eval/synthetic_unseen_100.csv` | 100 | Synthetic held-out data from nested archive; not the 60-sample research benchmark. |

## Data quality and responsible reuse

- Duplicates can inflate validation metrics if the same response appears in training and validation. `classify-train` **deduplicates by default**, and both modes prevent identical normalized responses from crossing splits. This changes the training protocol relative to the historical code.
- The original combined dataset has no conflicting labels on its exact duplicate texts; however, deduplication leaves only **10 unique Refused responses** (the original repeated rows formed the balanced 100-per-class counts).
- The precise **augmented training dataset** responsible for the 97% reported result was **not present** in the uploaded archive. The bundled combined data should not be described as a reproduction of the augmented model.
- Data contains potentially unsafe synthetic/research examples. Review it before making a repository public and verify permission and provenance for anything derived from third-party corpora. Do not describe the curated examples as a broadly representative external benchmark.
- Raw prompt banks, full generation transcripts, original upstream HarmBench files, and the unrelated 600-prompt experiment are intentionally **not included** in this GitHub-ready selection. They remain in the untouched original ZIP.
- No model weights or trained checkpoints were present in the uploaded files. Outputs from new runs belong in gitignored `models/`, `private_data/`, and `results/local_evaluation/` (if desired, add the last path to `.gitignore` too).

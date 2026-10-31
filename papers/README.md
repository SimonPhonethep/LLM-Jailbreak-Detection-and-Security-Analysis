# Research papers

These are the two PDF manuscripts provided by the project author. They are **separate but related studies**.

## 1. Jailbreaking Small LLMs: A Case Study on Security Risks Across Compression and Training Methods

**File:** [Jailbreaking_Small_LLMs_Compression_Study.pdf](Jailbreaking_Small_LLMs_Compression_Study.pdf)

A six-page case study comparing five small LLM configurations on a 600-prompt adversarial evaluation set. It uses a fine-tuned BART multiclass response classifier plus a rule-based heuristic for refusal recognition. The paper discusses how instruction tuning, pruning, distillation/quantization, and uncensoring correlate with different response distributions.

## 2. Multiclass Classification of Large Language Model Jailbreak Responses with ModernBERT

**File:** [ModernBERT_Jailbreak_Response_Classification.pdf](ModernBERT_Jailbreak_Response_Classification.pdf)

A nine-page classifier-focused follow-up comparing three ModernBERT variants and a BART baseline. It reports results on a 60-response labeled dataset, with **97% accuracy / 0.97 macro F1** for augmented ModernBERT on that particular curated benchmark, and discusses risks of overfitting when evaluating fresh LLM-generated responses.

## Publishing and reproducibility

These are **author-supplied manuscripts**, not proof of acceptance or publication. The PDFs include collaborator names, email addresses, and copyright/template notices. Check coauthor consent and any publisher or dataset rights before public redistribution. For what the repository actually reproduces, read [the research overview](../docs/research_overview.md) and the root [README](../README.md).

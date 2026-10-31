# Research overview: From LLM behavior analysis to response classification

This page connects the **two author-provided papers**. The findings below are **reported in those manuscripts**; they have not been independently rerun for this public-ready repository.

## Study 1 — Optimized small LLMs under adversarial prompts

**Paper:** [Jailbreaking Small LLMs: A Case Study on Security Risks Across Compression and Training Methods](../papers/Jailbreaking_Small_LLMs_Compression_Study.pdf) (6 pages)

**Question:** How do variations in training and model optimization influence how small LLMs respond to adversarial prompting?

- **Test collection:** 100 base tasks across five risk categories, each presented in six jailbreak-prompt styles, for **600 test prompts**.
- **Model configurations:** Llama 3.2 3B base, a 20%-pruned Llama 3.2 3B variant, Llama 3.2 3B Instruct, a DeepSeek-R1-distilled 4-bit Llama 3B variant, and an uncensored Llama 3.2 Instruct variant.
- **Classifier:** a fine-tuned `facebook/bart-large-mnli` response classifier, using **200 synthetic, labeled samples**. Labels: Refused, Harmful, and Incomplete.
- **Postprocessing:** a refusal-marker heuristic examined early response sentences and could reclassify a prediction to *Refused* in the presence of strong refusal language.
- **Finding:** the study reported different refusal and harmful-output distributions across configurations; for instance, its after-heuristic discussion reports **24.8% refusals for the instruction-tuned model** and **20.2% for the distilled/quantized model**. These are **classifier-derived observational estimates**, not independently human-verified attack-success measurements.

**Limitations in the paper:** ambiguous prompts, repeated prompt templates, incomplete model-training provenance, small classifier training set, and possible heuristic errors. It does **not** establish that optimization alone caused particular behaviors: the selected variants can differ in more ways than the named technique.

**Repository boundary:** The full 600-prompt set, generated model outputs, and scripts for recreating every model comparison are not included in the curated public-ready repository. The BART classifier training and optional model-generation utilities represent part of the general approach, not a guaranteed replication of every study result.

## Study 2 — Better labels for jailbreak responses

**Paper:** [Multiclass Classification of Large Language Model Jailbreak Responses with ModernBERT](../papers/ModernBERT_Jailbreak_Response_Classification.pdf) (9 pages)

**Question:** Can more diverse supervised data and a modern encoder-based architecture improve *response-level* labeling of refused, harmful, and incomplete completions?

- **Classifier architectures:** three configurations using `answerdotai/ModernBERT-large` and one `facebook/bart-large-mnli` baseline.
- **Data variants:** a **200-sample synthetic dataset**, a **300-sample combined dataset** (100 per class in the paper), and an **augmented** variant with additional ambiguous and edge-case responses.
- **Labeled evaluation:** **60 curated responses**, split evenly across the three classes.
- **Open-ended observations:** response distributions on Llama 3.2 Instruct and uncensored Llama variants using **50 adversarial prompts**. Because these responses were not independently ground-truth labeled, their class distributions **cannot be interpreted as classification accuracy**.

### Reported classifier results

| Configuration | Accuracy | Macro F1 |
| --- | ---: | ---: |
| ModernBERT (Base / combined) | 0.63 | 0.57 |
| ModernBERT (Synthetic) | 0.70 | 0.66 |
| ModernBERT (Augmented) | **0.97** | **0.97** |
| BART (Synthetic) | 0.73 | 0.71 |

These scores are taken from Table 2 of the ModernBERT manuscript (page 5). The confusion matrices on page 8 show that the augmented model correctly labeled **58/60** test responses, with **two Harmful responses classified as Incomplete**.

**Interpretation and caveat:** the augmented configuration performed best on this small curated test, but the manuscript explicitly discusses potential **overfitting**, linguistic overlap, and heavy *Harmful* prediction on unlabeled generated outputs. A high score on the 60-response benchmark **does not demonstrate a comparable accuracy rate in real-world deployment**.

**Repository boundary:** The exact augmented training file and fine-tuned checkpoints that produced the reported 97% were not found in the original project archive. The cleaned training pipeline can run *new* experiments on included datasets; it does not reconstruct the paper's missing artifacts.

## What connects the projects

The **first paper** uses a multiclass classifier as a measurement tool for comparing LLM outputs. The **second paper** studies the quality and generalization of that measurement tool itself, moving from the earlier BART baseline to controlled ModernBERT/BART experiments. Both share the same response labels, but their **datasets, questions, and outcome metrics must remain distinct**.

## Suggested future research

1. Create a larger, independently sourced, human-adjudicated test set covering different LLM architectures and prompt families.
2. Compare near-duplicate filtering and prompt-family splits against conventional random splits.
3. Publish a documented, consent- and license-reviewed augmented dataset and model checkpoint if permissions allow.
4. Evaluate calibration and abstention on ambiguous completions instead of forcing a confident class for every response.
5. Compare classifier assignments with human labels before interpreting model-to-model safety differences.

These are **recommendations for further work**, not claims that the existing experiments already tested them.

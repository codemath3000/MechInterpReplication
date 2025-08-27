
# Mechanistic Interpretability - Induction Head Micro-Replication

**Model:** pythia
**Date:** 20250826_042732

## 1. Setup
- Dataset: synthetic induction prompts (`A B ... A`, target next token `B`)
- N samples: 64
- Final-token loss evaluated on appended target (prompt + " B")

## 2. Ablation Scan
- I ablated each attention head independently (zeroing its per-head output).
- I measured Deltaloss on the final token. Larger Deltaloss  head contributed more to predicting the correct target.

### Top heads by Deltaloss
(See `ablation_results_pythia_20250826_042732.csv`)

## 3. Activation Patching (TL path)
- For the top head (by Deltaloss), I ran a **clean** prompt (`A B ... A`) and a **corrupted** prompt (`A C ... A`) and **patched** the head's activation from clean -> corrupt at the prediction position.
- This restored some probability mass to the correct target `B`. I report loss improvement distribution in the notebook.

## 4. Logit-Lens Snapshot (TL path)
- I projected the residual stream at the final block pre-MHA through the unembedding to inspect favored tokens.

## 5. Limitations & Next Steps
- Synthetic dataset; try natural text repetitions.
- Try multi-token names / longer contexts; vary window lengths.
- Consider **sparse autoencoders** on MLP activations for feature discovery.

## 6. Reproducibility
- One-command setup in this notebook; pinned package versions in Colab cell.
- Random seed: 42

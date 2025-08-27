# Mechanistic Interpretability: Induction Head Micro‑Replication

**Author:** Benjamin Hadad

**Date:** 2025‑08‑26

**Repo goal:** a small, clean replication of classic induction‑head behavior on a tiny open model, with head ablations, activation patching, and a simple logit‑lens snapshot. Everything is reproducible from a single notebook.

## TL;DR

I built a minimal induction task, then measured which attention heads matter for predicting the repeated token. On Pythia‑70M:

* Top head by delta loss: **Layer 2 / Head 5**, mean **delta loss +1.448** on the final token.
* 36 of 48 heads had positive delta loss.
* Top‑10 heads had mean **delta loss 0.402** (median 0.290).
* Activation patching on the top head improves performance on corrupted prompts. I include the code and a histogram; you can export the mean improvement if you want a single number.

The goal is not a new result. It is a clear, first principles replication with code someone else can read and extend.

---

## Problem setup

**Task.** Synthetic induction: show a prompt with a rare bigram early, add noise, then repeat the first token and ask the model to predict the second token.

Example:
`A B N N N ... N A -> predict B`

I evaluate the cross‑entropy on the final target token B when appended to the prompt so the prediction position is unambiguous.

Per‑token loss:
`loss_t = -log p(y_t | context up to t)`

**Model.** `EleutherAI/pythia-70m-deduped` through TransformerLens. I chose this model because it runs fast on a single GPU and has stable hooks. The code also includes an experimental path for Gemma‑3 1B that uses Transformers and NNsight.

**Data.** I generate 64 prompts per run with letters A..Z. Each sample uses a random pair (A, B) with A != B, plus 32 random noise tokens. I fix a seed for repeatability.

---

## Methods in brief

### 1) Head ablation

* Hook the per‑head pre‑output tensor `z` at each attention block (`blocks.{layer}.attn.hook_z` in TransformerLens).
* For each head, set its slice to zero across sequence positions and run a forward pass on the batch.
* Compute **delta loss** on the final token: ablated loss minus baseline loss. Larger delta means the head was more important for this task.

### 2) Activation patching

* Build a clean prompt (`A B ... A`) and a corrupted prompt (`A C ... A`), then append the correct target `B` to both for consistent final‑token loss.
* Cache the top head’s `z` activations from the clean run.
* Re‑run the corrupted prompt while patching that head’s `z` at the prediction position from clean to corrupt.
* Measure improved loss on the final token. I also plot a histogram over multiple trials.

### 3) Logit lens snapshot

* At a late residual stream (pre‑MHA in the final block), project the state through the unembedding matrix and print the top tokens. This is a coarse sanity check that the residual is pointing at the right answer.

---

## Results (Pythia‑70M)

**Run configuration.** 64 samples, 32 noise tokens, seed 42, evaluation on final token only.

**Ablation scan summary.**

* Top head: **L2/H5**, mean **delta loss +1.448**.
* 36 of 48 heads had positive delta loss.
* Top‑10 heads mean **0.402**, median **0.290**.

**Top‑10 heads by delta loss**

| Rank | Layer | Head | Delta loss |
| ---: | ----: | ---: | ---------: |
|    1 |     2 |    5 |      1.448 |
|    2 |     1 |    1 |      0.468 |
|    3 |     3 |    0 |      0.408 |
|    4 |     0 |    6 |      0.355 |
|    5 |     1 |    0 |      0.311 |
|    6 |     5 |    7 |      0.270 |
|    7 |     1 |    3 |      0.242 |
|    8 |     3 |    5 |      0.185 |
|    9 |     2 |    7 |      0.168 |
|   10 |     2 |    6 |      0.164 |

**Activation patching.**
For the top head (L2/H5), patching from clean to corrupt at the prediction position improves the final‑token loss on the target B. The notebook prints the distribution and shows a histogram. You can write the mean improvement to disk if you want a single number to cite.

**Logit lens.**
At the last block pre‑MHA, the top tokens usually include the correct target B on the induction prompts. This is a quick sanity check, not a substitute for ablations or patching.

---

## What this shows

* Induction‑style behavior is small but visible even in a 70M model, and a few heads carry most of the effect.
* Head ablation is an informative first pass. Activation patching then checks causality more directly by restoring a corrupted run with clean activations.
* The full pipeline fits in a single notebook with a clear metric and a minimal dataset. It is easy to extend to other models and tasks.

---

## Limitations and notes

* Synthetic data is simple. Real text can produce different head rankings and multi‑token effects.
* I evaluate only the final token. You may want to check earlier positions and longer contexts.
* Activation patching here patches a single head at a single position. You can expand to multi‑position or multi‑component patching for stronger claims.
* Pythia‑70M is small. Larger models will change the layer and head indices for the top effects.
* Gemma‑3 support is experimental in this repo. If library internals change, you may need to adjust hook names.

---

## Reproducing the results

**Option A: Colab.**
Open the provided notebook and run the cells top to bottom.

* Install cell pins NumPy 1.26.4, Pandas 2.1.4, and Matplotlib 3.8.4 to avoid ABI issues.
* If Colab prompts to restart the runtime, accept, then re‑run imports.

**Option B: Local.**

* Python 3.10+ and a recent PyTorch install. A GPU is optional for Pythia‑70M.
* Install dependencies like this:

```
pip install transformers==4.41.0 datasets==2.19.0 accelerate==0.30.0 \
einops==0.7.0 matplotlib==3.8.4 pandas==2.1.4 tqdm==4.66.0 \
transformer-lens==2.3.0 nnsight==0.3.7 huggingface_hub==0.24.0 \
numpy==1.26.4
```

**Expected outputs.**

* `outputs/ablation_results_*.csv` with per‑head delta loss.
* `outputs/report_stub_*.md` which you can replace with this write‑up.
* A printed histogram for activation patching. If you want a file, add a small snippet to dump mean and variance to JSON.

---

## Repository structure

Suggested layout:

```
.
├── mech_interp_induction_heads_colab.ipynb
├── README.md  (this file)
├── outputs/
│   ├── ablation_results_pythia_YYYYMMDD_HHMMSS.csv
│   └── report_stub_pythia_YYYYMMDD_HHMMSS.md
└── scripts/  (optional helpers if you extract code from the notebook)
```

---

## Next steps

* Run the same pipeline on **Gemma‑3 1B (base)**. Report its top head and delta loss. Note any differences due to grouped query attention or sliding windows.
* Export a small JSON summary for activation patching, for example:

  * mean improvement, standard deviation, number of trials.
* Try a natural text induction dataset or a multi‑token copy task.
* If you discover a small library bug or a helpful hook, open a tiny upstream PR.

---

## Acknowledgments

* Pythia models by EleutherAI.
* TransformerLens for convenient hooks and inspection.
* Thanks to prior mechanistic interpretability work on induction heads that motivated this replication.

---

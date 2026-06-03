# Pythia-160M SAE Feature Shift

This repository contains my mechanistic interpretability experiment for checking how features change after fine-tuning Pythia-160M on Python code.

The idea is simple:

1. Load pretrained Pythia-160M.
2. Collect layer 6 activations on Python code.
3. Train a Sparse Autoencoder on those activations.
4. Fine-tune Pythia-160M on Python code.
5. Collect layer 6 activations again from the fine-tuned model.
6. Train a second Sparse Autoencoder.
7. Compare SAE features before and after fine-tuning.

The hook used is:

```text
blocks.6.hook_resid_post
```

This is the residual stream after layer 6.

## Setup

```bash
pip install -r requirements.txt
```

Use a GPU runtime. I used Google Colab because my own PC does not have a GPU.

## Run order

```bash
python finetune_pythia.py
python collect_pretrained_acts.py
python collect_finetuned_acts.py
python train_pretrained_sae.py
python train_finetuned_sae.py
python compare_features.py
python inspect_features.py
```

## Main output files

```text
models/pythia160m_code_finetuned.pt
models/pretrained_pythia160m_code_layer6_sae.pt
models/code_finetuned_pythia160m_layer6_sae.pt

results/evaluation_quantitative_summary.csv
results/proper_code_sae_feature_decoder_comparison.csv
results/feature_status_counts.csv
results/most_preserved_features.csv
results/most_changed_features.csv
results/candidate_new_code_features_strict.csv
results/qualitative_context_examples.csv
```

## What I measured

```text
MSE reconstruction loss
average L0 sparsity
decoder cosine similarity
top activating tokens
context around activating tokens
feature firing rate
max activation
```


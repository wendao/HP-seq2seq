# Autoresearch Progress v2

Fast autoresearch with EPOCHS=20. Starting from Pre-LN baseline (9b3bde3).
Target: maximize val hit rate.

## Fixed Parameters

| Param | Value |
|-------|-------|
| EPOCHS | 20 |
| SPLIT | grouped |
| FOLD | 0 |
| LR | 0.001 |
| GPU | 3 |

## Baseline Configuration

| Param | Value |
|-------|-------|
| HIDDEN_SIZE | 512 |
| EMBED_DIM | 96 |
| NUM_LAYERS | 4 |
| NUM_HEADS | 4 |
| DROPOUT | 0.1 |
| BATCH_SIZE | 64 |
| WEIGHT_DECAY | 1e-4 |
| GRAD_CLIP | 1.0 |
| LR_SCHEDULER | warmup + cosine, min LR 0.1x |
| NORM_FIRST | True (Pre-LN) |

## Experiment Log

| # | Commit | val_hit_rate | Status | Note |
|---|--------|-------------|--------|------|
| 1 | 838537f | 0.2918 | keep | baseline (EPOCHS=20, Pre-LN) |
| 2 | ce6f214 | 0.3609 | keep | LR=0.002 |
| 3 | 3d3d91e | 0.3706 | keep | BATCH_SIZE=32 |
| 4 | 7d734ed | 0.4095 | keep | DROPOUT=0.05 |
| 5 | a0d2dfb | 0.4252 | keep | DROPOUT=0 |

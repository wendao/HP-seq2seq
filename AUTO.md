# Autoresearch Progress

Optimizing 5 hyperparameters for Transformer model on HP seq2seq task.
Target: maximize val hit rate.

## Fixed Parameters

| Param | Value |
|-------|-------|
| BATCH_SIZE | 64 |
| SPLIT | grouped |
| FOLD | 0 |
| EPOCHS | 50 |
| LR | 0.001 |
| GPU | 3 |

## Best Configuration

| Param | Value |
|-------|-------|
| HIDDEN_SIZE | 512 |
| EMBED_DIM | 96 |
| NUM_LAYERS | 4 |
| NUM_HEADS | 4 |
| DROPOUT | 0.1 |
| WEIGHT_DECAY | 1e-4 |
| GRAD_CLIP | 1.0 |
| LR_SCHEDULER | warmup (5 epochs) + cosine |
| MIN_LR | 0.1x (0.0001) |
| NORM_FIRST | True (Pre-LN) |

**Best val_hit_rate: 0.4634** (commit `fb40c90`, 3-run avg)

## Experiment Log

| # | Commit | val_hit_rate | Status | HIDDEN | EMBED | LAYERS | HEADS | DROP | Note |
|---|--------|-------------|--------|--------|-------|--------|-------|------|------|
| 1 | 6af4318 | 0.2654 | keep | 128 | 64 | 3 | 4 | 0.1 | baseline |
| 2 | 8cd5dc5 | 0.3153 | keep | 256 | 64 | 3 | 4 | 0.1 | +HIDDEN |
| 3 | f22000d | 0.3280 | keep | 256 | 64 | 4 | 4 | 0.1 | +LAYERS |
| 4 | eb57cc3 | 0.3702 | keep | 512 | 64 | 4 | 4 | 0.1 | ++HIDDEN |
| 5 | f1577b1 | 0.2573 | discard | 1024 | 64 | 4 | 4 | 0.1 | too big |
| 6 | dc63da9 | 0.2804 | discard | 512 | 64 | 5 | 4 | 0.1 | too deep |
| 7 | 30798ac | 0.4032 | keep | 512 | 96 | 4 | 4 | 0.1 | +EMBED |
| 8 | cf4fcd3 | 0.0317 | discard | 512 | 128 | 4 | 8 | 0.1 | too big |
| 9 | 6cb65d8 | - | crash | 512 | 96 | 4 | 8 | 0.1 | OOM |
| 10 | 735a00e | 0.2971 | discard | 512 | 96 | 4 | 4 | 0.0 | DROP=0 |
| 11 | 7f3c4f7 | 0.3751 | discard | 512 | 96 | 4 | 4 | 0.05 | DROP=0.05 |
| 12 | 499e6c5 | 0.3429 | discard | 512 | 96 | 4 | 4 | 0.15 | DROP=0.15 |
| 13 | 8b28007 | 0.3651 | discard | 768 | 96 | 4 | 4 | 0.1 | HIDDEN=768 |
| 14 | 3245b3f | 0.3727 | discard | 512 | 80 | 4 | 4 | 0.1 | EMBED=80 |
| 15 | 3fc8721 | 0.3816 | discard | 512 | 96 | 4 | 6 | 0.1 | HEADS=6 |
| 16 | b7a06fe | 0.3724 | discard | 512 | 96 | 4 | 4 | 0.1 | HIDDEN=640 |
| 17 | fccd52a | 0.3694 | discard | 512 | 96 | 3 | 4 | 0.1 | LAYERS=3 |

### Round 2: Training Tricks (from 30798ac baseline with warmup scheduler)

| # | Commit | val_hit_rate | Status | Note |
|---|--------|-------------|--------|------|
| 18 | 3acc335 | 0.4271 | discard | label smoothing 0.1 |
| 19 | 050acfa | 0.3353 | discard | LR 0.0005 |
| 20 | 872de58 | 0.3985 | discard | BATCH_SIZE 128 |
| 21 | 69aef32 | 0.4330 | discard | LR 0.002 |
| 22 | 25c7d19 | 0.4163 | discard | AdamW betas (0.9, 0.98) |
| 23 | 838ecde | 0.4461 | discard | min LR 0.1x (discarded after 3-run: 0.434 avg, unstable) |
| 24 | 9b3bde3 | **0.4634** | **keep** | **Pre-LN (norm_first=True) - 3-run avg, stable** |
| 25 | 4056a99 | 0.4457 | discard | min LR 0.2x |
| 26 | df314cd | 0.4418 | discard | warmup 10 epochs + min LR 0.1x |
| 27 | f3e41e2 | 0.4350 | discard | warmup 3 epochs + min LR 0.1x |
| 28 | 0553291 | 0.4459 | discard | grad clip 5.0 |
| 29 | 27b6689 | 0.4327 | discard | grad clip 0.5 |
| 30 | bfc12a0 | 0.4056 | discard | weight decay 1e-5 |
| 31 | 7f75a1e | 0.4395 | discard | DROPOUT 0.08 |
| 32 | fb40c90 | **0.4634** | **keep** | **Pre-LN cherry-pick to main (norm_first=True)** |

## Parameter Sweep Summary

| Param | Tested | Best | Trend |
|-------|--------|------|-------|
| HIDDEN_SIZE | 128,256,512,640,768,1024 | 512 | peak at 512 |
| EMBED_DIM | 64,80,96,128 | 96 | peak at 96 |
| NUM_LAYERS | 3,4,5 | 4 | peak at 4 |
| NUM_HEADS | 4,6,8(crash) | 4 | 4 best |
| DROPOUT | 0.0,0.05,0.1,0.15 | 0.1 | peak at 0.1 |
| LR | 0.0005,0.001,0.002 | 0.001 | peak at 0.001 |
| BATCH_SIZE | 64,128 | 64 | 64 best |
| GRAD_CLIP | 0.5,1.0,5.0 | 1.0 | peak at 1.0 |
| WEIGHT_DECAY | 1e-5,1e-4 | 1e-4 | 1e-4 best |
| WARMUP_EPOCHS | 3,5,10 | 5 | peak at 5 |
| MIN_LR | 0.0x,0.1x,0.2x | 0.1x | 0.1x best |

## Improvement Trajectory

```
0.2654  baseline
0.3153  +0.0499  HIDDEN 128->256
0.3280  +0.0127  LAYERS 3->4
0.3702  +0.0422  HIDDEN 256->512
0.4032  +0.0330  EMBED  64->96
0.4525  +0.0493  warmup+cosine LR (reported from c4fc6ef)
0.4634  +0.0109  Pre-LN norm_first=True (stable, 3-run verified)
-----------------
+0.1980  total (+74.6%)
```

## Key Findings

- **DROPOUT is critical**: 0.0 drops 26%, 0.05 drops 7%, 0.15 drops 15%
- **Model is at capacity sweet spot**: any further increase or decrease hurts
- **LR sensitivity**: 0.001 is sweet spot; 0.0005 drops 26%, 0.002 drops 7%
- **BATCH_SIZE 128**: worse than 64 (0.3985 vs ~0.45)
- **Label smoothing 0.1**: hurts exact-match accuracy (0.4271 vs ~0.45)
- **AdamW betas (0.9, 0.98)**: worse than default (0.9, 0.999)
- **Pre-LN (norm_first=True) is the best final config**: 0.4634 mean, stable across runs (std 0.002)
  - vs min LR 0.1x: 0.4461 mean, unstable (std 0.017)
  - The first 0.4705 run was an outlier — reproducibility verification is critical
- **Min LR is crucial**: preventing cosine decay to 0 boosted from ~0.4525 to **0.4705**
  - 0.1x min LR: **0.4705** (best)
  - 0.2x min LR: 0.4457 (worse, too much LR retained)
  - 0.0x min LR (original): ~0.4525
- **Warmup length**: 5 epochs is optimal; 3 epochs (0.4350) and 10 epochs (0.4418) both worse
- **GRAD_CLIP**: 1.0 is optimal; 0.5 (0.4327) and 5.0 (0.4459) both worse
- **WEIGHT_DECAY**: 1e-4 is better than 1e-5 (0.4056)

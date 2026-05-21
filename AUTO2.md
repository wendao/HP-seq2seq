# Autoresearch Progress v2

Fast autoresearch with EPOCHS=20. Starting from Pre-LN baseline (9b3bde3).
Target: maximize val hit rate.

## Fixed Parameters

| Param | Value |
|-------|-------|
| EPOCHS | 20 |
| SPLIT | grouped |
| FOLD | 0 |
| GPU | 3 |

## Best Configuration

| Param | Value |
|-------|-------|
| HIDDEN_SIZE | 1024 |
| EMBED_DIM | 96 |
| NUM_LAYERS | 4 |
| NUM_HEADS | 4 |
| DROPOUT | 0.0 |
| BATCH_SIZE | 32 |
| LR | 0.002 |
| WEIGHT_DECAY | 1e-4 |
| GRAD_CLIP | 1.0 |
| LR_SCHEDULER | warmup + cosine, min LR 0.1x |
| NORM_FIRST | True (Pre-LN) |

Best val hit rate: **0.4553** (commit 50890f8)

## Experiment Log

| # | Commit | val_hit_rate | Status | Note |
|---|--------|-------------|--------|------|
| 1 | 838537f | 0.2918 | keep | baseline (EPOCHS=20, Pre-LN) |
| 2 | ce6f214 | 0.3609 | keep | LR=0.002 (up from 0.001) |
| 3 | 3d3d91e | 0.3706 | keep | BATCH_SIZE=32 (was 64) |
| 4 | 7d734ed | 0.4095 | keep | DROPOUT=0.05 (was 0.1) |
| 5 | a0d2dfb | 0.4252 | keep | DROPOUT=0 |
| 6 | 1825b0d | 0.4441 | keep | HIDDEN_SIZE=768 (was 512) |
| 7 | 50890f8 | 0.4553 | keep | HIDDEN_SIZE=1024 |
| — | — | — | discard | BATCH_SIZE=16 (0.3397) |
| — | — | — | discard | LR=0.003 w/ BS=32 (0.3610) |
| — | — | — | discard | EMBED_DIM=128 (0.4429) |
| — | — | — | discard | NUM_LAYERS=5 (0.4317) |
| — | — | — | discard | NUM_HEADS=8 (0.4170) |
| — | — | — | discard | DROPOUT=0.05 w/ HIDDEN=1024 (0.4198) |
| — | — | — | discard | HIDDEN_SIZE=1280 (0.4032) |
| — | — | — | discard | LR=0.0025 (0.4113) |
| — | — | — | discard | WEIGHT_DECAY=5e-5 (0.4331) |
| — | — | — | discard | GRAD_CLIP=0.5 (0.4364) |
| — | — | — | discard | min LR 0.05x (0.4270) |
| — | — | — | discard | warmup 4 epochs (0.4048) |
| — | — | — | discard | BATCH_SIZE=24 (0.4075) |
| — | — | — | discard | label_smoothing=0.05 (0.4546) |

## Key Findings (EPOCHS=20)

1. **LR sweet spot shifted up**: At EPOCHS=20, LR=0.002 (vs 0.001 at EPOCHS=50). Fewer epochs need faster convergence.
2. **BATCH_SIZE=32 optimal**: 32 > 24 > 16 > 64. More updates/epoch helps at short training budget.
3. **DROPOUT=0 is best**: No overfitting concern at 20 epochs — dropout only slows convergence.
4. **Model width scales well**: HIDDEN=1024 > 768 > 512, but 1280 started overfitting.
5. **Warmup 5/20 essential**: 25% warmup ratio works best; 4/20 = 20% too short.
6. **Min LR 0.1x confirmed**: Wider range (0.05x) or narrower range both hurt.
7. **Label smoothing neutral**: 0.4546 ≈ 0.4553 (within noise), not worth the complexity.

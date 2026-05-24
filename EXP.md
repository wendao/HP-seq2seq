# HP-seq2seq 实验记录

优化目标: `score = val_hit_rate - 0.5 * train_hit_rate`  
固定: `EPOCHS=50, SPLIT=grouped, FOLD=0, SEED=42`  
数据: 24,900 samples, input max_len=20, output max_len=18

## Baseline

| # | commit | HIDDEN | EMBED | LAYERS | HEADS | DROPOUT | BATCH | LR | WD | CLIP | Scheduler | Val | Train | Score | Status |
|---|--------|--------|-------|--------|-------|---------|-------|-----|------|------|-----------|-----|-------|-------|--------|
| 0 | `189142c` | 512 | 96 | 4 | 4 | 0.1 | 64 | 0.001 | 1e-4 | 1.0 | warmup(5)+cosine(min=0.1) | 0.3914 | 0.4124 | **0.1852** | keep |

## 正则化

| # | commit | 改动 | Val | Train | Score | vs Baseline | Status |
|---|--------|------|-----|-------|-------|-------------|--------|
| 1 | `8f0e547` | DROPOUT=0.15 | 0.3888 | - | 0.1837 | -0.0015 | discard |
| 2 | `16eb9a1` | WEIGHT_DECAY=5e-4 | 0.4071 | - | 0.1820 | -0.0032 | discard |
| 3 | `0ef7aa2` | label_smoothing=0.1 | 0.3678 | - | 0.1696 | -0.0156 | discard |
| 19 | `8c36a12` | label_smoothing=0.05 | 0.3805 | - | 0.1768 | -0.0084 | discard |

## 调度器

| # | commit | 改动 | Val | Train | Score | vs Baseline | Status |
|---|--------|------|-----|-------|-------|-------------|--------|
| 4 | `4f7dc12` | plateau scheduler | 0.3676 | - | 0.1425 | -0.0427 | discard |
| 6 | `8023aa0` | warmup+cosine min_lr=0.3 | 0.4127 | - | 0.1798 | -0.0054 | discard |
| 8 | `54cf6b2` | cosine (no warmup) | 0.3001 | - | 0.1169 | -0.0683 | discard |
| 13 | `4815b64` | warmup=10 epochs | 0.3794 | - | 0.1765 | -0.0087 | discard |
| 14 | `ca78e64` | constant LR (no scheduler) | 0.3628 | - | 0.1596 | -0.0256 | discard |
| 21 | `699e7b1` | warmup=3 epochs | 0.3582 | - | 0.1717 | -0.0135 | discard |

## 模型架构

| # | commit | 改动 | Val | Train | Score | vs Baseline | Status |
|---|--------|------|-----|-------|-------|-------------|--------|
| 5 | `cdf75c3` | EMBED_DIM=64 | 0.3636 | - | 0.1643 | -0.0209 | discard |
| 9 | `ff5c648` | HIDDEN_SIZE=768 | 0.3864 | - | 0.1666 | -0.0186 | discard |
| 11 | `8191bd7` | NUM_LAYERS=3 | 0.3980 | - | 0.1824 | -0.0028 | discard |
| 15 | `c86fcb3` | weight tying (embed=proj) | 0.3307 | - | 0.1542 | -0.0310 | discard |
| 17 | `4781894` | NUM_HEADS=8 | 0.3556 | - | 0.1530 | -0.0322 | discard |
| 20 | `2aee282` | EMBED=192, HIDDEN=1024, HEADS=8 | 0.3123 | - | 0.1150 | -0.0702 | discard |

## Batch Size

| # | commit | 改动 | Val | Train | Score | vs Baseline | Status |
|---|--------|------|-----|-------|-------|-------------|--------|
| 10 | `b99c855` | BATCH_SIZE=32 | 0.4511 | ~0.54 | 0.1814 | -0.0038 | discard |
| - | (earlier) | BATCH_SIZE=256, LR=0.002 | 0.4289 | ~0.50 | 0.1788 | -0.0064 | discard |

## 优化器

| # | commit | 改动 | Val | Train | Score | vs Baseline | Status |
|---|--------|------|-----|-------|-------|-------------|--------|
| 7 | `c230633` | SGD+momentum=0.9 | 0.0000 | 0.0000 | 0.0000 | - | discard |
| 12 | `2e3451a` | no grad clip | 0.3628 | - | 0.1792 | -0.0060 | discard |
| 16 | `d848ac3` | Adam (no weight decay) | 0.4175 | - | 0.1771 | -0.0081 | discard |

## 组合策略

| # | commit | 改动 | Val | Train | Score | vs Baseline | Status |
|---|--------|------|-----|-------|-------|-------------|--------|
| 18 | `f25b17d` | BATCH=32 + DROPOUT=0.15 | 0.4063 | - | 0.1785 | -0.0067 | discard |

## 总结

- **最优配置**: Baseline (`189142c`), score=**0.1852**
- **总实验数**: 22 (含 baseline)
- **超过 baseline**: 0
- **接近 baseline** (±0.005): DROPOUT=0.15 (0.1837), NUM_LAYERS=3 (0.1824), BATCH=32 (0.1814), min_lr=0.3 (0.1798)
- 所有超参数维度均为单峰，baseline 处于峰值
- 模型可以达到 val=0.45 (epoch 50) 但 train 也升至 0.54，score 反而不如早期 checkpoint (val=0.39, train=0.41)

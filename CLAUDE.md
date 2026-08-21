# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Project overview

HP-seq2seq maps HP (hydrophobic/polar) sequences to lattice-protein structure
sequences. Input: length-20 over `{H, P}`. Output: length-18 over `{R, L, F}`.
The optimization target is **val hit rate** — whole-sequence exact match.

All 24,900 samples are exactly 20/18 characters, so there is no padding and the
metric is a clean exact match over 18 structure tokens plus `<eos>`.

## Common commands

```bash
# Main experiment. Fold is argv[1]; hyperparameters are constants at the top of train.py.
# Device auto-selects CUDA > MPS > CPU.
python train.py 0 > run.log 2>&1
grep "Best validation hit rate" run.log
tail -n 20 run.log                      # if the run crashed

# All 5 folds + mean/std summary, logs to logs_trf/
bash run_transformer.sh
bash run_transformer.sh 0               # single fold, smoke test

# Baselines
python train_baseline.py --model lstm --split grouped --fold 0 --epochs 100 \
    --num_layers 1 --hidden_size 128 --embed_dim 64
bash run_baseline.sh                    # full matrix -> logs/<model>[_ca]_<split>_<fold>.log
MODELS="lstm rnn" SPLITS=grouped CA=0 FOLDS=0 bash run_baseline.sh    # a slice

python prepare.py                       # data sanity check
python plot_training_curves.py <job>    # training curves from {job}_{fold}.log
```

## File roles

- **`train.py`** — **the only file the autoresearch loop edits.** Transformer seq2seq,
  hyperparameters as module-level constants. Grouped split by default. Tensors live on
  the training device (`LOAD_DATA_TO_GPU`); batching via `iter_tensor_batches`, no DataLoader.
- **`train_baseline.py`** — CLI for RNN / LSTM / CNN / Transformer baselines. Uses
  `DataLoader` with `prepare.Seq2SeqDataset`. `--cross_attn` adds cross-attention to the
  non-Transformer models; `--split {random,grouped}` and `--seed` (default 42) mirror
  `train.py`. Device auto-selects CUDA > MPS > CPU.
- **`prepare.py`** — **read-only ground truth.** `load_data`, `create_vocabs`,
  `create_folds` (random), `create_grouped_folds` (output-grouped), `Seq2SeqDataset`,
  `collate_fn`. Do not modify.
- **`run_transformer.sh` / `run_baseline.sh`** — 5-fold runners.
- **`datasets/dataset20int`** — 24,900 lines, `"input_seq output_seq"`.
- **`results.tsv`** — local experiment tracker, **not committed**. 7 columns:
  `commit  val_hit_rate  train_hit_rate  score  memory_gb  status  description`.

Documentation: `README.md` / `README-cn.md` are current. `PROGRAM.md` holds the
autoresearch rules (merged from the old `PROGRAM.md` and a near-duplicate
`PROG.md`, since removed). `EXP.md`,
`AUTO.md`, `AUTO2.md` are historical rounds — their conclusions are budget-specific,
see below. `GROUPS.md` covers the split analysis.

## Model architecture (train.py)

`Seq2SeqTransformer` = `EncoderTransformer` + `DecoderTransformer`, 1,243,654 params.

- Encoder: `nn.Embedding` → `PositionalEncoding` → `nn.TransformerEncoder`, `norm_first=True`
- Decoder: `nn.Embedding` → `PositionalEncoding` → `nn.TransformerDecoder`
  (causal mask buffer + cross-attention) → `nn.Linear`

Teacher forcing is implicit: the decoder receives `targets[:, :-1]` and predicts
`targets[:, 1:]`.

## Key hyperparameters (train.py)

| Constant | Current | Notes |
|---|---|---|
| `HIDDEN_SIZE` | 512 | FFN `dim_feedforward` |
| `EMBED_DIM` | 96 | `d_model`, must be divisible by `NUM_HEADS` |
| `NUM_LAYERS` | 4 | |
| `NUM_HEADS` | 4 | |
| `DROPOUT` | 0.1 | |
| `BATCH_SIZE` | 64 | |
| `LR` | 0.001 | |
| `WEIGHT_DECAY` | 1e-4 | AdamW |
| `GRAD_CLIP` | 1.0 | `None` disables |
| `LR_SCHEDULER` | `warmup` | 5-epoch warmup 0.1×→1.0×, cosine to a 0.1× floor |
| `EPOCHS` | 200 | |
| `SPLIT` | `grouped` | fixed |
| `FOLD` | argv[1] | fixed |
| `SEED` | 42 | fixed; seeds Python / CPU / CUDA / MPS |

## Grouped vs random splits

`create_folds` partitions at random; with only 5,310 distinct outputs across 24,900
samples, the same output appears in both train and validation and models can score
well by memorising outputs. `create_grouped_folds` keeps identical outputs in one
fold, so validation outputs are unseen — a much harder, honest evaluation. LSTM drops
by roughly a factor of four between the two. `GROUPS.md` holds the comparison table;
`README.md` holds the per-fold numbers — do not restate them here, they drift.

## Things that have burned prior runs

- **Training budget dominates hyperparameters.** Three rounds (~70 experiments) at
  EPOCHS=20/50 concluded every hyperparameter was "single-peaked with a narrow peak."
  Raising EPOCHS to 200 moved val from ~0.46 to ~0.60 — more than all that tuning
  combined. Conclusions in `AUTO.md` / `AUTO2.md` / `EXP.md` hold only at their own
  EPOCHS. Best epochs now land at 177-195 out of 200, still against the ceiling:
  **EPOCHS=400 is the highest-value next experiment**, ahead of any hyperparameter.

- **`score = val - 0.5*train` is a diagnostic, not the target.** It rejected the
  200-epoch result: score *fell* (0.1852 → 0.1511) while val rose 53%. It
  systematically prefers undertrained configs. Optimize val_hit_rate.

- **Teacher-forced exact match == greedy autoregressive exact match.** Proven by
  induction and verified empirically (418 vs 418 matches, zero disagreements over
  4,980 samples). Do not "fix" the evaluation by adding greedy decoding — it cannot
  change this metric. Beam search is a different story and is not implemented.
  Five logs (`logs/lstm_3.log`, `logs/lstm_4.log`, `logs/rnn_2.log`, `logs/rnn_3.log`, `logs/rnn_4.log`) report an
  autoregressive hit rate of 0.0000 next to a teacher-forced 0.20-0.41; that is a bug
  in a lost, uncommitted script version, and those logs are also at a different model
  size than folds 0-2, so they are not comparable.

- **`NUM_HEADS` must divide `EMBED_DIM`** — 8 with 96 crashes.

- **Run-to-run noise.** Before `SEED` was added, repeat runs of one commit differed by
  0.037, and a 0.4705 result recorded as "best" turned out to be an outlier. Treat
  single-fold differences under 0.02 as noise.

- **The baselines were re-run on 2026-08-20** (70 runs, seed 42, MPS) after fixing four
  bugs in `train_baseline.py`: it crashed on its first batch (`inputs, targets, _ = batch`
  against a 2-tuple `collate_fn`), had no `--split` flag, no seed, and set
  `CUDA_LAUNCH_BLOCKING=1` at import. The re-run reproduced the pre-fix logs to within
  0.014, so the old numbers were sound — only the code to reproduce them was missing.
  Fold-to-fold std is now ≤ 0.0172 across all twelve configurations, so a 5-fold mean
  needs no repeats.

- **The transformer baseline is undertrained, not broken.** `--model transformer` scores
  0.0961 random / 0.0881 grouped. It is the only model whose train hit rate (0.0313) is
  *below* its val (0.0881); its loss was still falling at epoch 100 and its best epoch was
  93/100. At 21,894 params against `train.py`'s 1,243,654, with Post-LN and no LR
  schedule, it is not a fair transformer row — do not quote it as "how transformers do."

- **`logs/lstm_3.log`, `logs/lstm_4.log`, `logs/rnn_2.log`, `logs/rnn_3.log`, `logs/rnn_4.log` are from a different architecture.**
  Their parameter counts (LSTM 496,774, RNN 150,406) are unreachable from this file's
  models by any `--num_layers` / `--hidden_size` / `--embed_dim` combination. Excluded
  from every table.

- **`results.tsv` is local-only.** Do not commit it.

## Where the numbers live

This file holds *rules and traps*, not results. Current per-fold numbers live in
`README.md` (and `README-cn.md`); the split comparison lives in `GROUPS.md`; the
experiment history lives in `EXP.md`. When a result changes, update those — this file
should only need editing when a rule changes. Three of its numbers went stale within a
day of being written, which is why they were removed.

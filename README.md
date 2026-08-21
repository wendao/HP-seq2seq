# HP-seq2seq

Seq2seq models for the HP lattice-protein folding map: hydrophobic/polar sequence → structure sequence.

## Task

| | |
|---|---|
| Input | HP sequence, length **20**, alphabet `{H, P}` |
| Output | structure sequence, length **18**, alphabet `{R, L, F}` (relative fold directions) |
| Metric | **hit rate** — fraction of validation samples whose *entire* output sequence is predicted correctly |
| Data | `datasets/dataset20int`, 24,900 samples, 5,310 unique outputs |

Every sample is exactly 20/18 characters, so there is no padding anywhere and the
metric is a clean whole-sequence exact match over 18 structure tokens plus `<eos>`.
One wrong position scores the sample zero.

## Project structure

```
train.py              # Transformer seq2seq. The only file the autoresearch loop edits.
train_baseline.py     # RNN / LSTM / CNN / Transformer baselines (CLI)
prepare.py            # data, vocab, folds, Dataset — read-only ground truth
run_transformer.sh    # 5-fold CV for train.py, logs to logs_trf/
run_baseline.sh       # 5-fold CV for the baselines, logs to logs/
plot_training_curves.py
datasets/dataset20int
logs/                 # baseline logs
extra_logs/           # Transformer 5-fold logs (EPOCHS=200)
EXP.md GROUPS.md AUTO.md AUTO2.md PROGRAM.md    # experiment records + autoresearch rules
results.tsv           # local experiment tracker, not committed
```

## Quick start

```bash
# Transformer, single fold. Hyperparameters are constants at the top of train.py;
# argv[1] is the fold (0-4). Device is auto-selected: CUDA > MPS > CPU.
python train.py 0 > run.log 2>&1
grep "Best validation hit rate" run.log

# Transformer, all 5 folds + summary (mean / std)
bash run_transformer.sh

# Baselines. --split defaults to random; --seed defaults to 42.
python train_baseline.py --model lstm --split grouped --fold 0 --epochs 100 \
    --num_layers 1 --hidden_size 128 --embed_dim 64 --cross_attn 8

# Full matrix: 70 runs (the transformer skips the CA axis), ~10h at 100 epochs on MPS.
# Or take a slice:
bash run_baseline.sh
MODELS="lstm rnn" SPLITS=grouped CA=0 bash run_baseline.sh
FOLDS=0 EPOCHS=20 bash run_baseline.sh

# Data sanity check
python prepare.py
```

On Apple silicon `train.py` selects MPS automatically. `run_transformer.sh` exports
`PYTORCH_ENABLE_MPS_FALLBACK=1` so any kernel Metal lacks falls back to CPU rather
than aborting the run.

## train.py

`Seq2SeqTransformer` = `EncoderTransformer` + `DecoderTransformer`, Pre-LN
(`norm_first=True`), sinusoidal positional encoding, causal mask buffer in the
decoder, cross-attention to the encoder memory. 1,243,654 parameters at the
current settings.

Data is tokenized once and kept on the training device (`LOAD_DATA_TO_GPU`);
batching goes through `iter_tensor_batches` rather than a DataLoader.

### Hyperparameters (top of the file)

| Constant | Current | Notes |
|---|---|---|
| `HIDDEN_SIZE` | 512 | FFN `dim_feedforward` |
| `EMBED_DIM` | 96 | `d_model`, must be divisible by `NUM_HEADS` |
| `NUM_LAYERS` | 4 | encoder and decoder |
| `NUM_HEADS` | 4 | |
| `DROPOUT` | 0.1 | |
| `BATCH_SIZE` | 64 | |
| `LR` | 0.001 | |
| `WEIGHT_DECAY` | 1e-4 | AdamW |
| `GRAD_CLIP` | 1.0 | `None` disables |
| `LR_SCHEDULER` | `warmup` | 5-epoch linear warmup 0.1×→1.0×, then cosine to a 0.1× floor |
| `EPOCHS` | 200 | |
| `SPLIT` | `grouped` | fixed |
| `SEED` | 42 | fixed |

## Two ways to split the data

`create_folds` partitions samples at random. Because the 24,900 samples share only
5,310 distinct outputs, the same output sequence lands in both train and validation,
and a model can score well by recognising outputs it has already memorised.

`create_grouped_folds` keeps all samples sharing an output sequence in the same
fold, so validation outputs are genuinely unseen. This is the honest evaluation and
is what `train.py` uses. See `GROUPS.md` — the ranking of the architectures changes
completely between the two.

## Results

### Transformer, grouped split (current configuration)

5 folds, `EPOCHS=200`, seed 42, MPS. Source `logs_trf/trf_grp_*.log`:

| Fold | 0 | 1 | 2 | 3 | 4 | **mean** |
|---|---|---|---|---|---|---|
| val hit rate | 0.5922 | 0.6042 | 0.5904 | 0.5857 | 0.6248 | **0.5995** |
| train hit rate at best epoch | 0.8911 | 0.8855 | 0.8973 | 0.9042 | 0.9059 | 0.8968 |
| best epoch | 190 | 177 | 193 | 191 | 195 | 189 |

std 0.0157, range [0.5857, 0.6248].

The jump from the ~0.46 recorded in `AUTO.md` came almost entirely from training
longer (50 → 200 epochs), not from further hyperparameter search — best epochs land
at 177-195, so 50 epochs was stopping less than a third of the way in. They are
still pressed against the 200-epoch ceiling, so the schedule may be truncating this
run too.

#### Device and seed check

`extra_logs/trf_grp_*.log` holds an earlier batch of the same configuration, run on
CUDA **before `SEED` existed**:

| | 0 | 1 | 2 | 3 | 4 | mean | std |
|---|---|---|---|---|---|---|---|
| CUDA, unseeded | 0.6204 | 0.6140 | 0.5514 | 0.6001 | 0.6102 | 0.5992 | 0.0277 |
| MPS, seed 42 | 0.5922 | 0.6042 | 0.5904 | 0.5857 | 0.6248 | 0.5995 | 0.0157 |

The means agree to 0.0003 — switching backend and adding a seed did not move the
result. The per-fold values do not track each other at all, though: fold 2 was the
low outlier on CUDA (0.5514) and is mid-pack on MPS (0.5904), while fold 4 moves the
other way. **Fold-to-fold differences here are run noise, not fold difficulty** — an
earlier revision of this file called fold 2 "consistently the hardest" on the
strength of one unseeded run, which the seeded batch does not support.

Seeding also halved the spread (std 0.0277 → 0.0157).

### Baselines, grouped split

100 epochs, seed 42, MPS. `--num_layers 1 --hidden_size 128 --embed_dim 64`
(CNN: 3 layers), source `logs/*_grouped_?.log`:

| Model | 0 | 1 | 2 | 3 | 4 | **mean** | std | train@best |
|---|---|---|---|---|---|---|---|---|
| LSTM + cross-attn | 0.2012 | 0.1851 | 0.1739 | 0.1993 | 0.2041 | **0.1927** | 0.0128 | 1.0000 |
| RNN + cross-attn | 0.1071 | 0.1168 | 0.1034 | 0.1049 | 0.1169 | **0.1098** | 0.0066 | 0.5600 |
| LSTM | 0.1066 | 0.1056 | 0.0996 | 0.1022 | 0.1125 | **0.1053** | 0.0049 | 0.7302 |
| CNN + cross-attn | 0.0331 | 0.0276 | 0.0298 | 0.0337 | 0.0392 | **0.0327** | 0.0044 | 0.1533 |
| CNN | 0.0315 | 0.0292 | 0.0259 | 0.0224 | 0.0381 | **0.0294** | 0.0059 | 0.1467 |
| RNN | 0.0304 | 0.0267 | 0.0170 | 0.0297 | 0.0280 | **0.0264** | 0.0054 | 0.3124 |

### Baselines, random split

Same settings, source `logs/*_random_?.log`:

| Model | 0 | 1 | 2 | 3 | 4 | **mean** | std | train@best |
|---|---|---|---|---|---|---|---|---|
| LSTM + cross-attn | 0.5673 | 0.5795 | 0.5765 | 0.5620 | 0.5361 | **0.5643** | 0.0172 | 0.9996 |
| LSTM | 0.4510 | 0.4374 | 0.4584 | 0.4319 | 0.4294 | **0.4416** | 0.0126 | 0.8936 |
| RNN + cross-attn | 0.3766 | 0.3809 | 0.3727 | 0.3886 | 0.3728 | **0.3783** | 0.0067 | 0.6450 |
| RNN | 0.2309 | 0.2333 | 0.2375 | 0.2345 | 0.2356 | **0.2344** | 0.0025 | 0.3716 |
| CNN + cross-attn | 0.1321 | 0.1237 | 0.1354 | 0.1197 | 0.1281 | **0.1278** | 0.0063 | 0.1559 |
| CNN | 0.1267 | 0.1249 | 0.1185 | 0.1338 | 0.1257 | **0.1259** | 0.0055 | 0.1465 |

`logs/baseline_boxplot.html` plots both tables.

### What the two splits say

Cross-attention is what separates the recurrent models, and its value is wildly
uneven: RNN gains +61% on the random split (0.2344 → 0.3783), LSTM +28%, CNN
+1.5%. The CNN barely moves because `Seq2SeqCNN.forward` passes
`encoder_embed` — the raw input embeddings, before any convolution — as the
attention memory, while the RNN and LSTM pass `encoder_output`. Its encoder also
max-pools 20 positions down to 1. Two bottlenecks in series; attention on the
wrong end of them.

The ranking changes between splits. Random: LSTM > RNN > CNN. Grouped: RNN+CA
(0.1098) and LSTM (0.1053) are level, and both CNN variants collapse to ~0.03 —
the same band as the bare RNN. Most of the CNN's 0.126 on the random split was
memorised output structure.

### Why the transformer is not in the baseline matrix

`train_baseline.py` still supports `--model transformer`, but it is excluded from the
default matrix and from every table above. At the baseline settings — 1 layer, 21,894
parameters, plain Adam with no LR schedule, 100 epochs — it does not reach convergence:
its loss was still falling monotonically at the last epoch, and it was the only model
whose train hit rate came in *below* its val hit rate, the signature of a model that has
not yet fit its training data. A score measured there reports the training budget, not
the architecture, so it would be misleading beside the six converged rows.

To run one, give it a schedule and enough epochs to converge:
`MODELS=transformer EPOCHS=400 bash run_baseline.sh`. For a transformer result on this
task as it stands, use `train.py` / `run_transformer.sh`.

**No controlled architecture comparison exists yet.** The 0.5995 from `train.py` comes
from a model with 3× the parameters of LSTM+CA, twice the epochs, and a tuned schedule.
"A transformer produced the best result on this task" is supported; "the transformer
architecture beats LSTM at equal budget" is not — nobody has run that experiment.

## State of the baselines

Until 2026-08-20 the committed `train_baseline.py` had four problems:

| Problem | Effect |
|---|---|
| `inputs, targets, _ = batch` vs a 2-tuple `collate_fn` | `ValueError` on the first batch — the script could not run at all |
| no `--split` flag, `create_folds` hardcoded | the grouped-split baselines were impossible to produce |
| no seed | runs not reproducible |
| `os.environ['CUDA_LAUNCH_BLOCKING'] = '1'` at import | serialises every CUDA kernel launch; a large, silent slowdown |

All four are fixed. The script takes `--split {random,grouped}` and `--seed`
(default 42), and selects CUDA > MPS > CPU like `train.py`. The tables above are
the full 70-run re-run on that fixed script.

**The re-run reproduces the lost driver's numbers**, which retroactively validates
the pre-fix logs in `logs/*_grp_*.log` and `logs/*_ca_*.log`:

| | re-run (seed 42, MPS) | old logs | Δ |
|---|---|---|---|
| RNN, grouped | 0.0264 | 0.0253 | +0.0011 |
| CNN, grouped | 0.0294 | 0.0299 | −0.0005 |
| LSTM, grouped | 0.1053 | 0.1039 | +0.0014 |
| CNN + CA, random | 0.1278 | 0.1271 | +0.0007 |
| RNN + CA, random | 0.3783 | 0.3641 | +0.0142 |
| LSTM + CA, random | 0.5643 | 0.5537 | +0.0106 |

So the lost driver was functionally equivalent to the fixed one; only the code to
reproduce it was missing.

Seeding also tightened the folds. The old RNN random-split folds 0 and 1 differed
by 0.025; the new five span 0.0025 (std). Across all twelve configurations the
largest std is 0.0172 (LSTM+CA random) — fold-to-fold noise is now far smaller
than any between-model difference, so a 5-fold mean is trustworthy without repeats.

Two log families remain unreproducible and are excluded from every table:
`logs/lstm_3.log`, `logs/lstm_4.log`, `logs/rnn_2.log`, `logs/rnn_3.log`, `logs/rnn_4.log` report
parameter counts (LSTM 496,774, RNN 150,406) that **no** combination of
`--num_layers` / `--hidden_size` / `--embed_dim` produces from the architectures in
this file — a different architecture, not merely different hyperparameters. Those
five also print `Autoregressive hit rate (greedy decoding): 0.0000` next to a
teacher-forced 0.20–0.41, which the next section shows is impossible.

## On the metric

Both `train.py` and `train_baseline.py` evaluate with teacher forcing — the decoder
is fed the true prefix at every position in a single parallel pass. For a *per-token*
metric that would be optimistic, but for whole-sequence exact match it is exactly
equivalent to greedy autoregressive decoding.

By induction: at step 1 both feed `<sos>`; if teacher-forced argmax equals the true
token at every position up to *t-1*, greedy decoding has generated that same prefix,
so at step *t* both feed identical input to identical weights and produce the same
argmax. A sequence is exact-matched under teacher forcing iff greedy decoding
reproduces it. Exposure bias is real but lives entirely in the samples the model
already gets wrong, which score zero either way.

Verified directly: training the `train.py` model to a val hit rate of 0.0839 and
evaluating the same weights both ways gives 418 exact matches under teacher forcing
and 418 under greedy decoding, with zero disagreements across all 4,980 validation
samples — while 4,264 of those samples do decode to different sequences. The metric
agrees; the sequences do not.

This does **not** extend to beam search, which optimises sequence likelihood and can
recover sequences greedy misses. No beam search is implemented.

The five orphan logs mentioned above also report `Autoregressive hit rate (greedy
decoding): 0.0000` alongside a teacher-forced 0.20-0.41. Given the equivalence, that
is a bug in the lost script, not a property of the models.

## Reproducibility

`train.py` seeds Python, CPU, CUDA and MPS from `SEED = 42`. cuDNN determinism is
left off (CUDA-only and costs speed), so runs are near- but not bit-reproducible.
Before the seed was added, repeat runs of the same commit varied by up to 0.037;
treat differences below ~0.02 as noise unless averaged over folds.

Device parity: fold 0 gives 0.6204 on CUDA and 0.6039 on MPS at `DROPOUT=0.1`,
`EPOCHS=200` — a 0.017 gap, within the pre-seed run-to-run spread.

## Open directions

- Beam search at inference — the one decoding change that could actually move the metric
- 200+ epochs: fold 3 peaked at epoch 200, so the schedule may still be truncating
- Re-run every baseline on the fixed script — `logs/` currently holds results no code
  in this repository can reproduce
- Transformer under the random split, for a like-for-like row in `GROUPS.md`

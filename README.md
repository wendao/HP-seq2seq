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

# Baselines
python train_baseline.py --model lstm --fold 0 --epochs 100 \
    --num_layers 1 --hidden_size 128 --embed_dim 64 --cross_attn 8
bash run_baseline.sh

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

5 folds, `EPOCHS=200`, source `extra_logs/trf_grp_*.log`:

| Fold | 0 | 1 | 2 | 3 | 4 | **mean** |
|---|---|---|---|---|---|---|
| val hit rate | 0.6204 | 0.6140 | 0.5514 | 0.6001 | 0.6102 | **0.5992** |
| train hit rate at best epoch | 0.8883 | 0.8870 | 0.8753 | 0.9056 | 0.8875 | 0.8887 |
| best epoch | 182 | 179 | 186 | 200 | 191 | |

std 0.0277, range [0.5514, 0.6204]. Fold 2 is consistently the hardest.

The jump from the ~0.46 recorded in `AUTO.md` came almost entirely from training
longer (50 → 200 epochs), not from further hyperparameter search — best epochs land
at 179-200, so 50 epochs was stopping less than a third of the way in.

### Baselines, grouped split

100 epochs, `--num_layers 1 --hidden_size 128 --embed_dim 64` (CNN: 3 layers),
source `logs/*_grp_*.log`:

| Model | 0 | 1 | 2 | 3 | 4 | **mean** |
|---|---|---|---|---|---|---|
| LSTM | 0.1053 | 0.1096 | 0.0959 | 0.0978 | 0.1107 | **0.1039** |
| CNN | 0.0293 | 0.0365 | 0.0246 | 0.0273 | 0.0319 | **0.0299** |
| RNN | 0.0290 | 0.0313 | 0.0180 | 0.0214 | 0.0266 | **0.0253** |

### Baselines, random split

| Model | 0 | 1 | 2 | 3 | 4 | **mean** |
|---|---|---|---|---|---|---|
| LSTM + cross-attn | 0.5613 | 0.5569 | 0.5473 | 0.5509 | 0.5521 | **0.5537** |
| LSTM | 0.4267 | 0.4427 | 0.4416 | — | — | **0.4370** (3 folds) |
| RNN + cross-attn | 0.3459 | 0.3856 | 0.3620 | 0.3699 | 0.3570 | **0.3641** |
| RNN | 0.2241 | 0.2487 | — | — | — | **0.2364** (2 folds) |
| CNN + cross-attn | 0.1077 | 0.1224 | 0.1326 | 0.1401 | 0.1325 | **0.1271** |
| CNN | 0.1053 | 0.1276 | 0.1404 | 0.1250 | 0.1243 | **0.1245** |

`logs/lstm_3`, `logs/lstm_4`, `logs/rnn_2`, `logs/rnn_3` and `logs/rnn_4` were
produced by an **uncommitted, since-lost version** of `train_baseline.py` at a
different model size (LSTM 496,774 vs 381,382 params; RNN 150,406 vs 108,742), so
they are not comparable to folds 0-2 and are excluded above. Earlier revisions of
this README averaged them in; those numbers (LSTM 0.4235, RNN 0.2143) were mixing
two configurations.

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
- Re-run the LSTM/RNN random-split folds that are missing, on the committed script
- Transformer under the random split, for a like-for-like row in `GROUPS.md`

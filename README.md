# HP-seq2seq

Seq2seq training framework for HP sequence → structure mapping.

## Task

Given an input HP sequence (length 20, alphabet `{H, P}`), predict the output structure sequence (length 18, alphabet `{R, L, F}`).

**Goal:** Maximize hit rate (exact match accuracy on output sequence).

## Data

- **File:** `datasets/dataset20int`
- **Size:** 24,900 samples
- **Format:** `HPHHPPHH... RFLLFR...` (space-separated input and output)
- **Input length:** 20 (alphabet: H, P)
- **Output length:** 18 (alphabet: R, L, F, plus special tokens <pad>, <sos>, <eos>)

## Project Structure

```
HP-seq2seq/
├── datasets/
│   └── dataset20int       # Training data (24900 samples)
├── prepare.py             # Data handling (load, vocab, folds, dataset, collate)
├── train.py               # Training script with RNN/LSTM/CNN models
└── README.md
```

## prepare.py (read-only)

Data preparation module:

- `load_data(path)` → list of (input_str, output_str)
- `create_vocabs(input_seqs, output_seqs)` → input_vocab, output_vocab
- `create_folds(data, n_folds=5)` → list of (train_idx, val_idx) tuples
- `encode_data(data, input_vocab, output_vocab)` → Seq2SeqDataset
- `collate_fn(batch)` → padded sequences

## train.py

Training script with model selection and hyperparameter support.

### Usage (GPU)

```bash
python train.py --model lstm --fold 0 --epochs 100 --num_layers 1 --hidden_size 128 --embed_dim 64
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | lstm | Model type: rnn, lstm, or cnn |
| `--fold` | 0 | Cross-validation fold (0-4) |
| `--epochs` | 50 | Number of training epochs |
| `--lr` | 0.001 | Learning rate |
| `--hidden_size` | 32 | Hidden dimension size |
| `--embed_dim` | 16 | Embedding dimension |
| `--batch_size` | 64 | Batch size |
| `--num_layers` | 1 | Number of layers |
| `--cross_attn` | 0 | Number of cross-attention heads (0 = disabled) |

### Models

All models accept `--hidden_size` and `--embed_dim` to control capacity.

**RNN Seq2Seq:**
- `EncoderRNN`: Bidirectional RNN encoder
- `DecoderRNN`: RNN decoder with simplified attention
- `--num_layers`: Number of RNN layers (default 1, also controls decoder)
- `--cross_attn`: Number of cross-attention heads (0 = disabled)

**LSTM Seq2Seq:**
- `EncoderLSTM`: Bidirectional LSTM encoder
- `DecoderLSTM`: LSTM decoder
- `--num_layers`: Number of LSTM layers (default 1)
- `--cross_attn`: Number of cross-attention heads (0 = disabled)

**CNN Seq2Seq:**
- `EncoderCNN`: Multi-layer Conv1D encoder with global max pooling
- `DecoderCNN`: Step-by-step Conv1D decoder (causal by design, like RNN)
- `--num_layers`: Number of CNN layers for both encoder and decoder
- `--cross_attn`: Number of cross-attention heads (0 = disabled)

| num_layers | hidden_size | embed_dim | Parameters |
|------------|-------------|-----------|------------|
| 1 | 128 | 64 | 50,822 |
| 3 | 128 | 64 | 247,942 |
| 3 | 256 | 128 | 987,398 |

### Training Details

- **Loss:** CrossEntropyLoss (ignore_index=<pad>)
- **Optimizer:** Adam
- **Metric:** Hit Rate = exact match accuracy on output sequence
- **5-fold cross-validation** (80/20 train/val split within each fold)
- **Teacher forcing** during training

### Output Vocabulary

Special tokens: `<pad>=0`, `<sos>=1`, `<eos>=2`
Content tokens: `R=3`, `L=4`, `F=5`

## Latest Results (5-fold CV, 100 epochs)

Parameters: `--num_layers 1 --hidden_size 128 --embed_dim 64` (CNN: `--num_layers 3`)

### LSTM
| Fold | Best Val Hit Rate |
|------|-------------------|
| 0 | 0.4267 |
| 1 | 0.4427 |
| 2 | 0.4416 |
| 3 | 0.4072 |
| 4 | 0.3995 |
| **Average** | **0.4235** |

### RNN
| Fold | Best Val Hit Rate |
|------|-------------------|
| 0 | 0.2241 |
| 1 | 0.2487 |
| 2 | 0.2040 |
| 3 | 0.1979 |
| 4 | 0.1968 |
| **Average** | **0.2143** |

### CNN
| Fold | Best Val Hit Rate |
|------|-------------------|
| 0 | 0.1053 |
| 1 | 0.1276 |
| 2 | 0.1404 |
| 3 | 0.1250 |
| 4 | 0.1243 |
| **Average** | **0.1245** |

## Summary

| Model | 5-fold CV Avg Hit Rate |
|-------|----------------------|
| **LSTM** | **0.4235** |
| RNN | 0.2143 |
| CNN | 0.1245 |

## Verification

```bash
# Test data loading
python prepare.py

# Run all baselines (5-fold CV)
bash run_baseline.sh

# Or train a single fold
python train.py --model lstm --fold 0 --epochs 100 --num_layers 1 --hidden_size 128 --embed_dim 64
```

## Future Improvements

1. **Beam search** - Use beam search during inference instead of greedy decoding
2. **Transformer models** - Replace CNN/RNN with transformer-based architectures
3. **Regularization** - Dropout, weight decay, label smoothing
4. **Learning rate scheduling** - Warmup and decay
5. **Data augmentation** - Reverse complement sequences
6. **Scheduled sampling** - Gradually reduce teacher forcing ratio

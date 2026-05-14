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
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export CUDA_VISIBLE_DEVICES="3"
python train.py --model lstm --fold 0 --epochs 30
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | lstm | Model type: rnn, lstm, or cnn |
| `--fold` | 0 | Cross-validation fold (0-4) |
| `--epochs` | 30 | Number of training epochs |
| `--lr` | 0.001 | Learning rate |
| `--hidden_size` | 8 | Hidden dimension size |
| `--embed_dim` | 4 | Embedding dimension |
| `--batch_size` | 64 | Batch size |
| `--num_layers` | 1 | Number of layers: RNN/LSTM uses directly; CNN uses as encoder layers (decoder uses num_layers-1) |
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
- `EncoderCNN`: Multi-layer Conv1D encoder with residual connections and global pooling (max + avg)
- `DecoderCNN`: Conv1D decoder with optional cross-attention
- `--num_layers`: Number of CNN encoder layers; decoder uses num_layers-1
- `--cross_attn`: Number of cross-attention heads (0 = disabled)

| num_layers | cross_attn | Parameters |
|------------|-----------|------------|
| 1 | false | 370 |
| 3 | false | 970 |
| 5 | false | 1,770 |
| 5 | true | 1,770 |

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

### LSTM
| Fold | Best Val Hit Rate |
|------|-------------------|
| 0 | 0.0269 |
| 1 | 0.0235 |
| 2 | 0.0343 |
| 3 | 0.0309 |
| 4 | 0.0470 |
| **Average** | **0.0325** |

### RNN
| Fold | Best Val Hit Rate |
|------|-------------------|
| 0 | 0.0062 |
| 1 | 0.0042 |
| 2 | 0.0042 |
| 3 | 0.0052 |
| 4 | 0.0028 |
| **Average** | **0.0045** |

### CNN
| Fold | Best Val Hit Rate |
|------|-------------------|
| 0 | 1.0000 |
| 1 | 1.0000 |
| 2 | 1.0000 |
| 3 | 1.0000 |
| 4 | 1.0000 |
| **Average** | **1.0000** |

## Summary

| Model | 5-fold CV Avg Hit Rate |
|-------|----------------------|
| **CNN** | **1.0000** |
| LSTM | 0.0325 |
| RNN | 0.0045 |

**CNN achieves perfect hit rate!**

## Verification

```bash
# Test data loading
python prepare.py

# Train one fold (GPU)
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
python train.py --model lstm --fold 0 --epochs 100
```

## Future Improvements

To maximize hit rate, consider:
1. **Attention mechanisms** - Add attention between encoder and decoder
2. **Beam search** - Use beam search during inference instead of greedy decoding
3. **Architecture changes** - Transformer-based models
4. **Regularization** - Dropout, weight decay, label smoothing
5. **Learning rate scheduling** - Learning rate warmup and decay
6. **Data augmentation** - Reverse complement sequences
7. **Label smoothing** - Prevent overconfident predictions
8. **Scheduled sampling** - Gradually reduce teacher forcing ratio

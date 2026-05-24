"""Seq2seq training for HP sequence -> structure mapping. Transformer model."""

import os, sys
#os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '0')

import math
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple

import prepare

# ============ Tunable hyperparameters ============

HIDDEN_SIZE = 512
EMBED_DIM = 96
NUM_LAYERS = 4
NUM_HEADS = 4
DROPOUT = 0.1
BATCH_SIZE = 64
LR = 0.001
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
LR_SCHEDULER = "warmup"  # 'none', 'cosine', 'plateau'
EPOCHS = 200
LOAD_DATA_TO_GPU = True

# ============ Fixed hyperparameters ============

SPLIT = 'grouped'
FOLD = int(sys.argv[1])

# ===============================================


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 64, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class EncoderTransformer(nn.Module):
    """Transformer encoder."""

    def __init__(self, embed_dim: int, hidden_size: int, input_vocab_size: int,
                 n_layers: int = 3, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(input_vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embed_dim, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=hidden_size,
            batch_first=True, dropout=dropout, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(x)
        embedded = self.pos_encoding(embedded)
        src_key_padding_mask = (x == 0)
        output = self.transformer(embedded, src_key_padding_mask=src_key_padding_mask)
        pooled = output.mean(dim=1)
        return output, pooled


class DecoderTransformer(nn.Module):
    """Transformer decoder with causal self-attention and cross-attention."""

    def __init__(self, embed_dim: int, hidden_size: int, output_vocab_size: int,
                 n_layers: int = 3, n_heads: int = 8, dropout: float = 0.1,
                 max_len: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(output_vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embed_dim, dropout=dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=hidden_size,
            batch_first=True, dropout=dropout, norm_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(embed_dim, output_vocab_size)
        causal_mask = torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1)
        self.register_buffer('causal_mask', causal_mask, persistent=False)

    def forward(self, encoder_output: torch.Tensor, target: torch.Tensor,
                src_padding_mask: torch.Tensor = None) -> torch.Tensor:
        tgt_len = target.size(1)
        embedded = self.embedding(target)
        embedded = self.pos_encoding(embedded)
        tgt_mask = self.causal_mask[:tgt_len, :tgt_len]
        tgt_key_padding_mask = (target == 0)
        output = self.transformer(
            embedded, encoder_output,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )
        return self.output_proj(output)


class Seq2SeqTransformer(nn.Module):
    """Transformer-based seq2seq model with built-in cross-attention."""

    def __init__(self, embed_dim: int, hidden_size: int, input_vocab_size: int, output_vocab_size: int,
                 n_layers: int = 3, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.encoder = EncoderTransformer(embed_dim, hidden_size, input_vocab_size, n_layers, n_heads, dropout)
        self.decoder = DecoderTransformer(embed_dim, hidden_size, output_vocab_size, n_layers, n_heads, dropout)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        src_padding_mask = (inputs == 0)
        encoder_output, _ = self.encoder(inputs, lengths)
        logits = self.decoder(encoder_output, targets, src_padding_mask)
        return logits


def compute_hit_rate(logits: torch.Tensor, targets: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """Compute exact match accuracy (hit rate)."""
    predictions = logits.argmax(dim=-1)
    mask = (targets != pad_idx)
    correct = (predictions == targets) & mask
    exact_match = correct.all(dim=1)
    return exact_match.float().mean()


def encode_tensors(data, input_vocab, output_vocab, device: torch.device,
                   pad_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode fixed-length HP data once and keep it on the target device."""
    input_rows = []
    output_rows = []
    for input_seq, output_seq in data:
        input_rows.append([input_vocab.get(ch, input_vocab['<unk>']) for ch in input_seq])
        output_rows.append(
            [output_vocab['<sos>']]
            + [output_vocab.get(ch, pad_idx) for ch in output_seq]
            + [output_vocab['<eos>']]
        )

    inputs = torch.tensor(input_rows, dtype=torch.long, device=device)
    outputs = torch.tensor(output_rows, dtype=torch.long, device=device)
    lengths = (inputs != pad_idx).sum(dim=1)
    return inputs, outputs, lengths


def iter_tensor_batches(inputs: torch.Tensor, outputs: torch.Tensor, lengths: torch.Tensor,
                        batch_size: int, shuffle: bool):
    n_samples = inputs.size(0)
    if shuffle:
        indices = torch.randperm(n_samples, device=inputs.device)
    else:
        indices = torch.arange(n_samples, device=inputs.device)

    for start in range(0, n_samples, batch_size):
        batch_idx = indices[start:start + batch_size]
        yield (
            inputs.index_select(0, batch_idx),
            outputs.index_select(0, batch_idx),
            lengths.index_select(0, batch_idx),
        )


def train_epoch(model: nn.Module, tensors: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], criterion: nn.Module,
                optimizer: optim.Optimizer, batch_size: int, pad_idx: int = 0,
                grad_clip: float = None) -> Tuple[float, float]:
    model.train()
    inputs_all, targets_all, lengths_all = tensors
    total_loss = torch.zeros((), device=inputs_all.device)
    total_hit_rate = torch.zeros((), device=inputs_all.device)
    n_batches = 0
    for inputs, targets, lengths in iter_tensor_batches(inputs_all, targets_all, lengths_all,
                                                        batch_size, shuffle=True):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs, targets[:, :-1], lengths)
        loss = criterion(logits.view(-1, logits.size(-1)), targets[:, 1:].contiguous().view(-1))
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        with torch.no_grad():
            hit_rate = compute_hit_rate(logits, targets[:, 1:], pad_idx)
        total_loss += loss.detach()
        total_hit_rate += hit_rate
        n_batches += 1
    return (total_loss / n_batches).item(), (total_hit_rate / n_batches).item()


def eval_epoch(model: nn.Module, tensors: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], criterion: nn.Module,
               batch_size: int, pad_idx: int = 0) -> Tuple[float, float]:
    model.eval()
    inputs_all, targets_all, lengths_all = tensors
    total_loss = torch.zeros((), device=inputs_all.device)
    total_hit_rate = torch.zeros((), device=inputs_all.device)
    n_batches = 0
    with torch.no_grad():
        for inputs, targets, lengths in iter_tensor_batches(inputs_all, targets_all, lengths_all,
                                                            batch_size, shuffle=False):
            logits = model(inputs, targets[:, :-1], lengths)
            loss = criterion(logits.view(-1, logits.size(-1)), targets[:, 1:].contiguous().view(-1))
            hit_rate = compute_hit_rate(logits, targets[:, 1:], pad_idx)
            total_loss += loss.detach()
            total_hit_rate += hit_rate
            n_batches += 1
    return (total_loss / n_batches).item(), (total_hit_rate / n_batches).item()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda' and hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')
    print(f"Using device: {device}")

    data = prepare.load_data('datasets/dataset20int')
    print(f"Loaded {len(data)} samples")

    input_seqs = [d[0] for d in data]
    output_seqs = [d[1] for d in data]
    input_vocab, output_vocab = prepare.create_vocabs(input_seqs, output_seqs)
    print(f"Input vocab size: {len(input_vocab)}, Output vocab size: {len(output_vocab)}")

    if SPLIT == 'grouped':
        folds = prepare.create_grouped_folds(data, n_folds=5)
    else:
        folds = prepare.create_folds(data, n_folds=5)
    train_idx, val_idx = folds[FOLD]
    print(f"Fold {FOLD}: train={len(train_idx)}, val={len(val_idx)}")

    train_data = [data[i] for i in train_idx]
    val_data = [data[i] for i in val_idx]

    tensor_device = device if LOAD_DATA_TO_GPU else torch.device('cpu')
    train_tensors = encode_tensors(train_data, input_vocab, output_vocab, tensor_device, pad_idx=output_vocab['<pad>'])
    val_tensors = encode_tensors(val_data, input_vocab, output_vocab, tensor_device, pad_idx=output_vocab['<pad>'])
    if tensor_device != device:
        train_tensors = tuple(t.to(device) for t in train_tensors)
        val_tensors = tuple(t.to(device) for t in val_tensors)
    print(f"Data tensors loaded on: {train_tensors[0].device}")

    input_vocab_size = len(input_vocab)
    output_vocab_size = len(output_vocab)

    model = Seq2SeqTransformer(EMBED_DIM, HIDDEN_SIZE, input_vocab_size, output_vocab_size,
                               n_layers=NUM_LAYERS, n_heads=NUM_HEADS, dropout=DROPOUT)
    model = model.to(device)
    print(f"Model: transformer, params={sum(p.numel() for p in model.parameters()):,}")
    print(f"  num_heads={NUM_HEADS}, num_layers={NUM_LAYERS}, dropout={DROPOUT}")
    print(f"  batch_size={BATCH_SIZE}, lr={LR}, weight_decay={WEIGHT_DECAY}, grad_clip={GRAD_CLIP}")
    print(f"  lr_scheduler={LR_SCHEDULER}")

    criterion = nn.CrossEntropyLoss(ignore_index=output_vocab['<pad>'])
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    if LR_SCHEDULER == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    elif LR_SCHEDULER == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    elif LR_SCHEDULER == 'warmup':
        def lr_lambda(epoch):
            if epoch < 5:
                return 0.1 + 0.9 * epoch / 5
            else:
                progress = (epoch - 5) / (EPOCHS - 5)
                return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    best_val_hit_rate = 0
    for epoch in range(EPOCHS):
        train_loss, train_hit_rate = train_epoch(model, train_tensors, criterion, optimizer, BATCH_SIZE,
                                                  pad_idx=output_vocab['<pad>'],
                                                  grad_clip=GRAD_CLIP if GRAD_CLIP else None)
        val_loss, val_hit_rate = eval_epoch(model, val_tensors, criterion, BATCH_SIZE, pad_idx=output_vocab['<pad>'])

        if scheduler is not None:
            if LR_SCHEDULER in ('cosine', 'warmup'):
                scheduler.step()
            elif LR_SCHEDULER == 'plateau':
                scheduler.step(val_hit_rate)

        if val_hit_rate > best_val_hit_rate:
            best_val_hit_rate = val_hit_rate

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Hit Rate: {train_hit_rate:.4f} | "
              f"Val Loss: {val_loss:.4f}, Hit Rate: {val_hit_rate:.4f}")

    print(f"\nBest validation hit rate: {best_val_hit_rate:.4f}")


if __name__ == '__main__':
    main()

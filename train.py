"""Seq2seq training for HP sequence -> structure mapping. Transformer model."""

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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
LR_SCHEDULER = 'cosine'  # 'none', 'cosine', 'plateau'
EPOCHS = 50

# ============ Fixed hyperparameters ============

SPLIT = 'grouped'
FOLD = 0

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
            batch_first=True, dropout=dropout
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
                 n_layers: int = 3, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(output_vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embed_dim, dropout=dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=hidden_size,
            batch_first=True, dropout=dropout
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(embed_dim, output_vocab_size)

    def forward(self, encoder_output: torch.Tensor, target: torch.Tensor,
                src_padding_mask: torch.Tensor = None) -> torch.Tensor:
        tgt_len = target.size(1)
        embedded = self.embedding(target)
        embedded = self.pos_encoding(embedded)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(target.device)
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


def compute_hit_rate(logits: torch.Tensor, targets: torch.Tensor, pad_idx: int = 0) -> float:
    """Compute exact match accuracy (hit rate)."""
    predictions = logits.argmax(dim=-1)
    mask = (targets != pad_idx)
    correct = (predictions == targets) & mask
    exact_match = correct.all(dim=1)
    return exact_match.float().mean().item()


def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
                optimizer: optim.Optimizer, device: torch.device, pad_idx: int = 0,
                grad_clip: float = None) -> Tuple[float, float]:
    model.train()
    total_loss, total_hit_rate, n_batches = 0, 0, 0
    for batch in dataloader:
        inputs, targets, _ = batch
        inputs, targets = inputs.to(device), targets.to(device)
        lengths = (inputs != pad_idx).sum(dim=1)
        optimizer.zero_grad()
        logits = model(inputs, targets[:, :-1], lengths)
        loss = criterion(logits.view(-1, logits.size(-1)), targets[:, 1:].contiguous().view(-1))
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        with torch.no_grad():
            hit_rate = compute_hit_rate(logits, targets[:, 1:], pad_idx)
        total_loss += loss.item()
        total_hit_rate += hit_rate
        n_batches += 1
    return total_loss / n_batches, total_hit_rate / n_batches


def eval_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
               device: torch.device, pad_idx: int = 0) -> Tuple[float, float]:
    model.eval()
    total_loss, total_hit_rate, n_batches = 0, 0, 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets, _ = batch
            inputs, targets = inputs.to(device), targets.to(device)
            lengths = (inputs != pad_idx).sum(dim=1)
            logits = model(inputs, targets[:, :-1], lengths)
            loss = criterion(logits.view(-1, logits.size(-1)), targets[:, 1:].contiguous().view(-1))
            hit_rate = compute_hit_rate(logits, targets[:, 1:], pad_idx)
            total_loss += loss.item()
            total_hit_rate += hit_rate
            n_batches += 1
    return total_loss / n_batches, total_hit_rate / n_batches


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    train_dataset = prepare.Seq2SeqDataset(train_data, input_vocab, output_vocab)
    val_dataset = prepare.Seq2SeqDataset(val_data, input_vocab, output_vocab)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=prepare.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=prepare.collate_fn)

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
    else:
        scheduler = None

    best_val_hit_rate = 0
    for epoch in range(EPOCHS):
        train_loss, train_hit_rate = train_epoch(model, train_loader, criterion, optimizer, device,
                                                  pad_idx=output_vocab['<pad>'], grad_clip=GRAD_CLIP if GRAD_CLIP else None)
        val_loss, val_hit_rate = eval_epoch(model, val_loader, criterion, device, pad_idx=output_vocab['<pad>'])

        if scheduler is not None:
            if LR_SCHEDULER == 'cosine':
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

"""Seq2seq training for HP sequence → structure mapping."""

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple

import prepare


# ============ Cross-Attention Module ============

class CrossAttention(nn.Module):
    """Shared cross-attention for all model types.

    Q: from decoder's target embedding or hidden state
    K,V: from encoder output (full sequence for RNN/LSTM, embeddings for CNN)
    """

    def __init__(self, embed_dim: int, n_heads: int = 8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(embed_dim)

    def forward(self, target_embed: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # target_embed: (batch, tgt_len, embed_dim)  - Q
        # memory: (batch, src_len, embed_dim)          - K,V
        attn_out, _ = self.cross_attn(target_embed, memory, memory)
        return self.cross_norm(target_embed + attn_out)


# ============ RNN Models ============

class EncoderRNN(nn.Module):
    """Simple RNN encoder."""

    def __init__(self, embed_dim: int, hidden_size: int, input_vocab_size: int, num_layers: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(input_vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, hidden = self.rnn(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        # hidden shape: (num_layers * num_directions, batch, hidden)
        # Take top layer (last layer) forward + backward: (batch, hidden*2)
        top_fwd = hidden[self.num_layers - 1]
        top_bwd = hidden[2 * self.num_layers - 1]
        combined_hidden = torch.cat([top_fwd, top_bwd], dim=1).contiguous()
        return output, combined_hidden


class DecoderRNN(nn.Module):
    """RNN decoder with optional cross-attention."""

    def __init__(self, embed_dim: int, hidden_size: int, output_vocab_size: int,
                 num_layers: int = 1, cross_attn: int = 0,
                 cross_attn_module: nn.Module = None):
        super().__init__()
        self.embedding = nn.Embedding(output_vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim + hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.output_proj = nn.Linear(hidden_size, output_vocab_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cross_attn_enabled = cross_attn > 0
        self.cross_attn = cross_attn_module
        # Project encoder output (hidden*2) to embed_dim for cross-attention K,V
        self.enc_proj = nn.Linear(hidden_size * 2, embed_dim)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor,
                encoder_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(x)
        # hidden: (num_layers, batch, hidden) -> use top layer for concatenation
        top_hidden = hidden[-1]  # (batch, hidden)
        hidden_expanded = top_hidden.unsqueeze(1).expand(-1, x.size(1), -1)

        if self.cross_attn_enabled and self.cross_attn is not None:
            memory = self.enc_proj(encoder_output)  # (batch, src_len, embed_dim)
            embedded = self.cross_attn(embedded, memory)

        rnn_input = torch.cat([embedded, hidden_expanded], dim=2)
        output, new_hidden = self.rnn(rnn_input, hidden)
        logits = self.output_proj(output)
        return logits, new_hidden[-1]


class Seq2SeqRNN(nn.Module):
    """RNN-based seq2seq model."""

    def __init__(self, embed_dim: int, hidden_size: int, input_vocab_size: int, output_vocab_size: int,
                 num_layers: int = 1, cross_attn: int = 0,
                 cross_attn_module: nn.Module = None):
        super().__init__()
        self.encoder = EncoderRNN(embed_dim, hidden_size, input_vocab_size, num_layers=num_layers)
        self.decoder = DecoderRNN(embed_dim, hidden_size, output_vocab_size, num_layers=num_layers,
                                  cross_attn=cross_attn, cross_attn_module=cross_attn_module)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        encoder_output, hidden = self.encoder(inputs, lengths)
        # hidden is (batch, hidden*2), extract forward direction and expand to num_layers
        forward_hidden = hidden[:, :self.decoder.hidden_size]
        expanded_hidden = forward_hidden.unsqueeze(0).expand(
            self.decoder.num_layers, -1, -1).contiguous()
        logits, _ = self.decoder(targets, expanded_hidden, encoder_output)
        return logits


# ============ LSTM Models ============

class EncoderLSTM(nn.Module):
    """LSTM encoder."""

    def __init__(self, embed_dim: int, hidden_size: int, input_vocab_size: int, num_layers: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(input_vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, (hidden, cell) = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        # Take top layer forward + backward: (batch, hidden*2)
        combined_hidden = torch.cat([hidden[self.num_layers - 1], hidden[2 * self.num_layers - 1]], dim=1)
        combined_cell = torch.cat([cell[self.num_layers - 1], cell[2 * self.num_layers - 1]], dim=1)
        return output, (combined_hidden, combined_cell)


class DecoderLSTM(nn.Module):
    """LSTM decoder with optional cross-attention."""

    def __init__(self, embed_dim: int, hidden_size: int, output_vocab_size: int,
                 num_layers: int = 1, cross_attn: int = 0,
                 cross_attn_module: nn.Module = None):
        super().__init__()
        self.embedding = nn.Embedding(output_vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim + hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.output_proj = nn.Linear(hidden_size, output_vocab_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cross_attn_enabled = cross_attn > 0
        self.cross_attn = cross_attn_module
        self.enc_proj = nn.Linear(hidden_size * 2, embed_dim)

    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor],
                encoder_output: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        embedded = self.embedding(x)
        hidden_h, hidden_c = hidden
        hidden_h = hidden_h.contiguous()
        hidden_c = hidden_c.contiguous()
        # hidden_h: (num_layers, batch, hidden) -> use top layer for concatenation
        top_hidden_h = hidden_h[-1]  # (batch, hidden)
        hidden_h_expanded = top_hidden_h.unsqueeze(1).expand(-1, x.size(1), -1)

        if self.cross_attn_enabled and self.cross_attn is not None:
            memory = self.enc_proj(encoder_output)  # (batch, src_len, embed_dim)
            embedded = self.cross_attn(embedded, memory)

        rnn_input = torch.cat([embedded, hidden_h_expanded], dim=2)
        output, (new_h, new_c) = self.lstm(rnn_input, (hidden_h, hidden_c))
        logits = self.output_proj(output)
        return logits, (new_h[-1], new_c[-1])


class Seq2SeqLSTM(nn.Module):
    """LSTM-based seq2seq model."""

    def __init__(self, embed_dim: int, hidden_size: int, input_vocab_size: int, output_vocab_size: int,
                 num_layers: int = 1, cross_attn: int = 0,
                 cross_attn_module: nn.Module = None):
        super().__init__()
        self.encoder = EncoderLSTM(embed_dim, hidden_size, input_vocab_size, num_layers=num_layers)
        self.decoder = DecoderLSTM(embed_dim, hidden_size, output_vocab_size, num_layers=num_layers,
                                   cross_attn=cross_attn, cross_attn_module=cross_attn_module)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        encoder_output, (hidden, cell) = self.encoder(inputs, lengths)
        # hidden/cell are (batch, hidden*2), expand forward direction to num_layers
        forward_h = hidden[:, :self.decoder.hidden_size]
        forward_c = cell[:, :self.decoder.hidden_size]
        expanded_h = forward_h.unsqueeze(0).expand(self.decoder.num_layers, -1, -1).contiguous()
        expanded_c = forward_c.unsqueeze(0).expand(self.decoder.num_layers, -1, -1).contiguous()
        logits, _ = self.decoder(targets, (expanded_h, expanded_c), encoder_output)
        return logits


# ============ CNN Models ============

class EncoderCNN(nn.Module):
    """CNN encoder with configurable complexity via cnn_level."""

    def __init__(self, embed_dim: int, hidden_size: int, input_vocab_size: int, n_layers: int = 5):
        super().__init__()
        self.embedding = nn.Embedding(input_vocab_size, embed_dim, padding_idx=0)
        in_ch = embed_dim
        self.convs = nn.ModuleList()
        for i in range(n_layers):
            self.convs.append(nn.Conv1d(in_ch, hidden_size, kernel_size=3, padding=1))
            in_ch = hidden_size
        self.n_layers = n_layers

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        h = embedded.transpose(1, 2)  # (batch, embed_dim, seq_len)

        for i, conv in enumerate(self.convs):
            h = torch.relu(conv(h))
            if i > 0:
                h = h + residual
            if i < self.n_layers - 1:
                residual = h

        global_max = torch.max(h, dim=2)[0]
        global_avg = h.mean(dim=2)
        pooled = torch.cat([global_max, global_avg], dim=1)  # (batch, hidden*2)
        # Return both pooled vector and embedding sequence for cross-attention
        return pooled, embedded


class DecoderCNN(nn.Module):
    """CNN decoder with optional cross-attention."""

    def __init__(self, embed_dim: int, hidden_size: int, output_vocab_size: int,
                 n_layers: int = 4, cross_attn: bool = False,
                 cross_attn_module: nn.Module = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attn_enabled = cross_attn
        self.cross_attn = cross_attn_module

        self.embedding = nn.Embedding(output_vocab_size, embed_dim)

        self.convs = nn.ModuleList()
        in_ch = embed_dim
        for _ in range(n_layers):
            self.convs.append(nn.Conv1d(in_ch, hidden_size, kernel_size=3, padding=1))
            in_ch = hidden_size

        self.output_proj = nn.Linear(hidden_size, output_vocab_size)

    def forward(self, encoder_hidden: torch.Tensor, target: torch.Tensor,
                memory: torch.Tensor = None) -> torch.Tensor:
        batch_size = encoder_hidden.size(0)
        seq_len = target.size(1)

        embedded = self.embedding(target)  # (batch, tgt_len, embed_dim)

        if self.cross_attn_enabled and self.cross_attn is not None and memory is not None:
            embedded = self.cross_attn(embedded, memory)

        x = embedded.transpose(1, 2)
        h = x
        for i, conv in enumerate(self.convs):
            h = torch.relu(conv(h))
            if i > 0:
                h = h + residual
            if i < len(self.convs) - 1:
                residual = h

        logits = self.output_proj(h.transpose(1, 2))
        return logits


class Seq2SeqCNN(nn.Module):
    """CNN-based seq2seq model."""

    def __init__(self, embed_dim: int, hidden_size: int, input_vocab_size: int, output_vocab_size: int,
                 cnn_n_layers: int = 5, cross_attn: int = 0,
                 cross_attn_module: nn.Module = None):
        super().__init__()
        self.encoder = EncoderCNN(embed_dim, hidden_size, input_vocab_size, n_layers=cnn_n_layers)
        decoder_n_layers = max(1, cnn_n_layers - 1)
        self.decoder = DecoderCNN(embed_dim, hidden_size, output_vocab_size, n_layers=decoder_n_layers,
                                  cross_attn=cross_attn, cross_attn_module=cross_attn_module)
        self.project = nn.Linear(hidden_size * 2, embed_dim)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        encoder_hidden, encoder_embed = self.encoder(inputs, lengths)
        decoder_init = self.project(encoder_hidden)
        memory = encoder_embed  # (batch, src_len, embed_dim) for cross-attn
        logits = self.decoder(decoder_init, targets, memory)
        return logits


# ============ Training ============

def compute_hit_rate(logits: torch.Tensor, targets: torch.Tensor, pad_idx: int = 0) -> float:
    """Compute exact match accuracy (hit rate)."""
    predictions = logits.argmax(dim=-1)
    mask = (targets != pad_idx)
    correct = (predictions == targets) & mask
    exact_match = correct.all(dim=1)
    return exact_match.float().mean().item()


def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
                 optimizer: optim.Optimizer, device: torch.device, pad_idx: int = 0) -> Tuple[float, float]:
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


def parse_args():
    parser = argparse.ArgumentParser(description='HP-seq2seq training')
    parser.add_argument('--model', type=str, default='lstm', choices=['rnn', 'lstm', 'cnn'])
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--embed_dim', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=1, help='Number of layers (RNN/LSTM: num_layers; CNN: encoder=num_layers, decoder=num_layers-1)')
    parser.add_argument('--cross_attn', type=int, default=0, help='Number of cross-attention heads (0=disabled)')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data = prepare.load_data('datasets/dataset20int')
    print(f"Loaded {len(data)} samples")

    input_seqs = [d[0] for d in data]
    output_seqs = [d[1] for d in data]
    input_vocab, output_vocab = prepare.create_vocabs(input_seqs, output_seqs)
    print(f"Input vocab size: {len(input_vocab)}, Output vocab size: {len(output_vocab)}")

    folds = prepare.create_folds(data, n_folds=5)
    train_idx, val_idx = folds[args.fold]
    print(f"Fold {args.fold}: train={len(train_idx)}, val={len(val_idx)}")

    train_data = [data[i] for i in train_idx]
    val_data = [data[i] for i in val_idx]

    train_dataset = prepare.Seq2SeqDataset(train_data, input_vocab, output_vocab)
    val_dataset = prepare.Seq2SeqDataset(val_data, input_vocab, output_vocab)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=prepare.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=prepare.collate_fn)

    input_vocab_size = len(input_vocab)
    output_vocab_size = len(output_vocab)

    # Build cross-attention module once, share across encoder-decoder
    cross_attn_module = None
    if args.cross_attn > 0:
        cross_attn_module = CrossAttention(args.embed_dim, n_heads=args.cross_attn)

    if args.model == 'rnn':
        model = Seq2SeqRNN(args.embed_dim, args.hidden_size, input_vocab_size, output_vocab_size,
                          num_layers=args.num_layers, cross_attn=args.cross_attn,
                          cross_attn_module=cross_attn_module)
    elif args.model == 'lstm':
        model = Seq2SeqLSTM(args.embed_dim, args.hidden_size, input_vocab_size, output_vocab_size,
                           num_layers=args.num_layers, cross_attn=args.cross_attn,
                           cross_attn_module=cross_attn_module)
    else:
        model = Seq2SeqCNN(args.embed_dim, args.hidden_size, input_vocab_size, output_vocab_size,
                          cnn_n_layers=args.num_layers, cross_attn=args.cross_attn,
                          cross_attn_module=cross_attn_module)

    model = model.to(device)
    print(f"Model: {args.model}, params={sum(p.numel() for p in model.parameters()):,}")
    if args.cross_attn:
        print(f"  cross_attn=enabled")

    criterion = nn.CrossEntropyLoss(ignore_index=output_vocab['<pad>'])
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_hit_rate = 0
    for epoch in range(args.epochs):
        train_loss, train_hit_rate = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_hit_rate = eval_epoch(model, val_loader, criterion, device)

        if val_hit_rate > best_val_hit_rate:
            best_val_hit_rate = val_hit_rate

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f}, Hit Rate: {train_hit_rate:.4f} | "
              f"Val Loss: {val_loss:.4f}, Hit Rate: {val_hit_rate:.4f}")

    print(f"\nBest validation hit rate: {best_val_hit_rate:.4f}")


if __name__ == '__main__':
    main()

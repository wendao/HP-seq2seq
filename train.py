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


# ============ Models ============

class EncoderRNN(nn.Module):
    """Simple RNN encoder."""

    def __init__(self, embed_dim: int, hidden_size: int, input_vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(input_vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True, bidirectional=True)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, hidden = self.rnn(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        # Combine bidirectional hidden states
        hidden = torch.cat([hidden[0], hidden[1]], dim=1).contiguous()
        return output, hidden


class DecoderRNN(nn.Module):
    """Simple RNN decoder with attention."""

    def __init__(self, embed_dim: int, hidden_size: int, output_vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(output_vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim + hidden_size, hidden_size, batch_first=True)
        self.output_proj = nn.Linear(hidden_size, output_vocab_size)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor, hidden: torch.Tensor, encoder_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len) of token ids
        # hidden: (batch, hidden_size)
        # encoder_output: (batch, src_len, hidden_size)
        batch_size = x.size(0)
        seq_len = x.size(1)

        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        hidden_expanded = hidden.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, hidden_size)

        # Concatenate embedded input with hidden state (simplified attention)
        rnn_input = torch.cat([embedded, hidden_expanded], dim=2)
        output, new_hidden = self.rnn(rnn_input, hidden.unsqueeze(0).contiguous())
        logits = self.output_proj(output)  # (batch, seq_len, vocab_size)
        return logits, new_hidden.squeeze(0)


class Seq2SeqRNN(nn.Module):
    """RNN-based seq2seq model."""

    def __init__(self, embed_dim: int, hidden_size: int, input_vocab_size: int, output_vocab_size: int):
        super().__init__()
        self.encoder = EncoderRNN(embed_dim, hidden_size, input_vocab_size)
        self.decoder = DecoderRNN(embed_dim, hidden_size, output_vocab_size)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # Encoder
        encoder_output, hidden = self.encoder(inputs, lengths)
        hidden = hidden[:, :self.decoder.hidden_size].contiguous()  # Take forward direction only

        # Decoder
        logits, _ = self.decoder(targets, hidden, encoder_output)
        return logits


# ============ LSTM Models ============

class EncoderLSTM(nn.Module):
    """LSTM encoder."""

    def __init__(self, embed_dim: int, hidden_size: int, input_vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(input_vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True, bidirectional=True)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, (hidden, cell) = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        # Combine bidirectional hidden states
        hidden = torch.cat([hidden[0], hidden[1]], dim=1)
        cell = torch.cat([cell[0], cell[1]], dim=1)
        return output, (hidden, cell)


class DecoderLSTM(nn.Module):
    """LSTM decoder."""

    def __init__(self, embed_dim: int, hidden_size: int, output_vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(output_vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim + hidden_size, hidden_size, batch_first=True)
        self.output_proj = nn.Linear(hidden_size, output_vocab_size)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor], encoder_output: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size = x.size(0)
        seq_len = x.size(1)

        embedded = self.embedding(x)
        hidden_h, hidden_c = hidden
        hidden_h = hidden_h.contiguous()
        hidden_c = hidden_c.contiguous()
        hidden_h_expanded = hidden_h.unsqueeze(1).expand(-1, seq_len, -1)

        rnn_input = torch.cat([embedded, hidden_h_expanded], dim=2)
        output, (new_h, new_c) = self.lstm(rnn_input, (hidden_h.unsqueeze(0), hidden_c.unsqueeze(0)))
        logits = self.output_proj(output)
        return logits, (new_h.squeeze(0), new_c.squeeze(0))


class Seq2SeqLSTM(nn.Module):
    """LSTM-based seq2seq model."""

    def __init__(self, embed_dim: int, hidden_size: int, input_vocab_size: int, output_vocab_size: int):
        super().__init__()
        self.encoder = EncoderLSTM(embed_dim, hidden_size, input_vocab_size)
        self.decoder = DecoderLSTM(embed_dim, hidden_size, output_vocab_size)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        encoder_output, (hidden, cell) = self.encoder(inputs, lengths)
        # Use only forward direction
        hidden = hidden[:, :self.decoder.hidden_size].contiguous()
        cell = cell[:, :self.decoder.hidden_size].contiguous()

        logits, _ = self.decoder(targets, (hidden, cell), encoder_output)
        return logits


# ============ CNN Models ============

class EncoderCNN(nn.Module):
    """CNN encoder for seq2seq."""

    def __init__(self, embed_dim: int, hidden_size: int, input_vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(input_vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = embedded.transpose(1, 2)  # (batch, embed_dim, seq_len)

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        # Global max pooling over sequence
        pooled = torch.max(x, dim=2)[0]  # (batch, hidden_size)
        return pooled


class DecoderCNN(nn.Module):
    """CNN decoder for seq2seq."""

    def __init__(self, embed_dim: int, hidden_size: int, output_vocab_size: int, max_len: int = 20):
        super().__init__()
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.output_proj = nn.Linear(hidden_size, output_vocab_size)

    def forward(self, encoder_hidden: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # encoder_hidden: (batch, hidden_size) - from encoder
        # target: (batch, target_len)
        batch_size = encoder_hidden.size(0)
        seq_len = target.size(1)

        # Initialize hidden state
        h = encoder_hidden.unsqueeze(2)  # (batch, hidden_size, 1)

        # Embed target tokens
        embedded = self.embedding(target)  # (batch, target_len, embed_dim)
        x = embedded.transpose(1, 2)  # (batch, embed_dim, target_len)

        # Causal convolution: process step by step
        outputs = []
        for t in range(seq_len):
            x_t = x[:, :, t:t+1]  # (batch, embed_dim, 1)
            h = torch.relu(self.conv1(x_t) + h)
            h = torch.relu(self.conv2(h) + h)
            h = torch.relu(self.conv3(h) + h)
            logits = self.output_proj(h.squeeze(2))  # (batch, vocab_size)
            outputs.append(logits)

        return torch.stack(outputs, dim=1)  # (batch, target_len, vocab_size)


class Seq2SeqCNN(nn.Module):
    """CNN-based seq2seq model."""

    def __init__(self, embed_dim: int, hidden_size: int, input_vocab_size: int, output_vocab_size: int):
        super().__init__()
        self.encoder = EncoderCNN(embed_dim, hidden_size, input_vocab_size)
        self.decoder = DecoderCNN(embed_dim, hidden_size, output_vocab_size)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        encoder_hidden = self.encoder(inputs, lengths)
        logits = self.decoder(encoder_hidden, targets)
        return logits


# ============ Training ============

def compute_hit_rate(logits: torch.Tensor, targets: torch.Tensor, pad_idx: int = 0) -> float:
    """Compute exact match accuracy (hit rate)."""
    predictions = logits.argmax(dim=-1)  # (batch, seq_len)
    targets = targets  # (batch, seq_len)

    # Mask out padding
    mask = (targets != pad_idx)
    correct = (predictions == targets) & mask
    # Exact match: all non-pad tokens must match
    exact_match = correct.all(dim=1)
    return exact_match.float().mean().item()


def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device, pad_idx: int = 0) -> Tuple[float, float]:
    """Train one epoch."""
    model.train()
    total_loss = 0
    total_hit_rate = 0
    n_batches = 0

    for batch in dataloader:
        inputs, targets, _ = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Compute sequence lengths (accounting for padding)
        lengths = (inputs != pad_idx).sum(dim=1)

        optimizer.zero_grad()

        # Forward pass
        logits = model(inputs, targets[:, :-1], lengths)  # Teacher forcing

        # Reshape for loss computation
        # logits: (batch, seq_len, vocab_size) -> (batch * seq_len, vocab_size)
        # targets: (batch, seq_len) -> (batch * seq_len,)
        loss = criterion(logits.view(-1, logits.size(-1)), targets[:, 1:].contiguous().view(-1))

        loss.backward()
        optimizer.step()

        # Compute hit rate
        with torch.no_grad():
            hit_rate = compute_hit_rate(logits, targets[:, 1:], pad_idx)

        total_loss += loss.item()
        total_hit_rate += hit_rate
        n_batches += 1

    return total_loss / n_batches, total_hit_rate / n_batches


def eval_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device, pad_idx: int = 0) -> Tuple[float, float]:
    """Evaluate one epoch."""
    model.eval()
    total_loss = 0
    total_hit_rate = 0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs, targets, _ = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

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
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    data = prepare.load_data('datasets/dataset20int')
    print(f"Loaded {len(data)} samples")

    # Create vocabularies
    input_seqs = [d[0] for d in data]
    output_seqs = [d[1] for d in data]
    input_vocab, output_vocab = prepare.create_vocabs(input_seqs, output_seqs)
    print(f"Input vocab size: {len(input_vocab)}, Output vocab size: {len(output_vocab)}")

    # Create folds
    folds = prepare.create_folds(data, n_folds=5)
    train_idx, val_idx = folds[args.fold]
    print(f"Fold {args.fold}: train={len(train_idx)}, val={len(val_idx)}")

    # Split data
    train_data = [data[i] for i in train_idx]
    val_data = [data[i] for i in val_idx]

    # Create datasets
    train_dataset = prepare.Seq2SeqDataset(train_data, input_vocab, output_vocab)
    val_dataset = prepare.Seq2SeqDataset(val_data, input_vocab, output_vocab)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=prepare.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=prepare.collate_fn)

    # Create model
    input_vocab_size = len(input_vocab)
    output_vocab_size = len(output_vocab)

    if args.model == 'rnn':
        model = Seq2SeqRNN(args.embed_dim, args.hidden_size, input_vocab_size, output_vocab_size)
    elif args.model == 'lstm':
        model = Seq2SeqLSTM(args.embed_dim, args.hidden_size, input_vocab_size, output_vocab_size)
    else:
        model = Seq2SeqCNN(args.embed_dim, args.hidden_size, input_vocab_size, output_vocab_size)

    model = model.to(device)
    print(f"Model: {args.model}, params={sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=output_vocab['<pad>'])
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
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
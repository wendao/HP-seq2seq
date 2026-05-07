"""Data preparation for HP-seq2seq."""

from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict


def load_data(path: str) -> List[Tuple[str, str]]:
    """Load dataset from file. Each line: input_seq output_seq (space-separated)."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 2:
                input_seq, output_seq = parts
                data.append((input_seq, output_seq))
    return data


def create_vocabs(input_seqs: List[str], output_seqs: List[str]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Create vocabularies for input and output sequences."""
    input_vocab = {'<pad>': 0, '<unk>': 1}
    output_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2}

    for seq in input_seqs:
        for ch in seq:
            if ch not in input_vocab:
                input_vocab[ch] = len(input_vocab)

    for seq in output_seqs:
        for ch in seq:
            if ch not in output_vocab:
                output_vocab[ch] = len(output_vocab)

    return input_vocab, output_vocab


def create_folds(data: List[Tuple[str, str]], n_folds: int = 5) -> List[Tuple[List[int], List[int]]]:
    """Create n-fold cross-validation splits. Returns list of (train_idx, val_idx) tuples."""
    n = len(data)
    indices = list(range(n))
    fold_size = n // n_folds

    folds = []
    for i in range(n_folds):
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < n_folds - 1 else n
        val_idx = indices[val_start:val_end]
        train_idx = indices[:val_start] + indices[val_end:]
        folds.append((train_idx, val_idx))

    return folds


class Seq2SeqDataset(Dataset):
    """Dataset for seq2seq training."""

    def __init__(self, data: List[Tuple[str, str]], input_vocab: Dict[str, int], output_vocab: Dict[str, int]):
        self.data = data
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.reverse_output_vocab = {v: k for k, v in output_vocab.items()}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_seq, output_seq = self.data[idx]

        # Encode input
        input_ids = [self.input_vocab.get(ch, self.input_vocab['<unk>']) for ch in input_seq]

        # Encode output: add <sos> at start, <eos> at end
        output_ids = [self.output_vocab['<sos>']]
        output_ids += [self.output_vocab.get(ch, self.output_vocab['<pad>']) for ch in output_seq]
        output_ids.append(self.output_vocab['<eos>'])

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(output_ids, dtype=torch.long)
        )


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function: pad sequences to same length in a batch."""
    input_ids, output_ids = zip(*batch)

    # Pad inputs
    input_lens = [len(x) for x in input_ids]
    max_input_len = max(input_lens)
    padded_inputs = []
    for inp in input_ids:
        pad_len = max_input_len - len(inp)
        padded_inputs.append(torch.cat([inp, torch.zeros(pad_len, dtype=torch.long)]))

    # Pad outputs (teacher forcing)
    output_lens = [len(x) for x in output_ids]
    max_output_len = max(output_lens)
    padded_outputs = []
    for out in output_ids:
        pad_len = max_output_len - len(out)
        padded_outputs.append(torch.cat([out, torch.zeros(pad_len, dtype=torch.long)]))

    # Also create target: decoder input shifted right (for loss computation)
    # Target = output_ids[1:] with <eos> at end
    targets = []
    for out in output_ids:
        # Remove <sos> from beginning, add <eos> at end
        target = out[1:]
        targets.append(target)

    # Pad targets
    max_target_len = max(len(t) for t in targets)
    padded_targets = []
    for t in targets:
        pad_len = max_target_len - len(t)
        padded_targets.append(torch.cat([t, torch.zeros(pad_len, dtype=torch.long)]))

    return (
        torch.stack(padded_inputs),
        torch.stack(padded_outputs),
        torch.stack(padded_targets)
    )


def encode_data(data: List[Tuple[str, str]], input_vocab: Dict[str, int], output_vocab: Dict[str, int]) -> Dataset:
    """Encode data into tensor dataset."""
    return Seq2SeqDataset(data, input_vocab, output_vocab)


if __name__ == '__main__':
    # Test data loading
    data = load_data('datasets/dataset20int')
    print(f"Loaded {len(data)} samples")
    print(f"Sample: {data[0]}")

    input_seqs = [d[0] for d in data]
    output_seqs = [d[1] for d in data]

    input_vocab, output_vocab = create_vocabs(input_seqs, output_seqs)
    print(f"Input vocab: {input_vocab}")
    print(f"Output vocab: {output_vocab}")

    folds = create_folds(data, n_folds=5)
    print(f"Created {len(folds)} folds")
    print(f"Fold 0 size: train={len(folds[0][0])}, val={len(folds[0][1])}")
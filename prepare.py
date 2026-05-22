"""Data preparation for HP-seq2seq."""

from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset
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


def create_grouped_folds(data, n_folds=5):
    groups = defaultdict(list)
    for i, (_, out) in enumerate(data):
        groups[out].append(i)
    group_items = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
    fold_samples = [[] for _ in range(n_folds)]
    fold_sizes = [0] * n_folds
    for _, indices in group_items:
        target = min(range(n_folds), key=lambda f: fold_sizes[f])
        fold_samples[target].extend(indices)
        fold_sizes[target] += len(indices)
    all_indices = set(range(len(data)))
    folds = []
    for i in range(n_folds):
        val_idx = fold_samples[i]
        train_idx = list(all_indices - set(val_idx))
        folds.append((train_idx, val_idx))
    return folds


class Seq2SeqDataset(Dataset):
    """Dataset for seq2seq training. Pre-tokenizes and pre-pads all data to avoid
    per-epoch CPU overhead. Data is small enough to live entirely in CPU memory."""

    def __init__(self, data: List[Tuple[str, str]], input_vocab: Dict[str, int], output_vocab: Dict[str, int]):
        input_ids_list = []
        output_ids_list = []

        for input_seq, output_seq in data:
            inp = [input_vocab.get(ch, input_vocab['<unk>']) for ch in input_seq]
            out = [output_vocab['<sos>']]
            out += [output_vocab.get(ch, output_vocab['<pad>']) for ch in output_seq]
            out.append(output_vocab['<eos>'])
            input_ids_list.append(torch.tensor(inp, dtype=torch.long))
            output_ids_list.append(torch.tensor(out, dtype=torch.long))

        max_input_len = max(len(x) for x in input_ids_list)
        max_output_len = max(len(x) for x in output_ids_list)

        self.inputs = torch.zeros(len(data), max_input_len, dtype=torch.long)
        self.outputs = torch.zeros(len(data), max_output_len, dtype=torch.long)

        for i, (inp, out) in enumerate(zip(input_ids_list, output_ids_list)):
            self.inputs[i, :len(inp)] = inp
            self.outputs[i, :len(out)] = out

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.outputs[idx]


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate function: stacks pre-padded tensors. Data is already padded in Seq2SeqDataset."""
    inputs, outputs = zip(*batch)
    return torch.stack(inputs), torch.stack(outputs)


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
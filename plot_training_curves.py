#!/usr/bin/env python3
"""Plot training curves from log files with format: {job}.{fold}.log"""

import re
import sys
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <job> [n_folds=5] [output_prefix='']")
    print(f"  job:      prefix of log files, e.g. rnn, lstm, cnn")
    print(f"  n_folds:  number of folds (default 5)")
    print(f"  output_prefix: prefix for output files (default: same as job)")
    sys.exit(1)

job = sys.argv[1]
n_folds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
out_prefix = sys.argv[3] if len(sys.argv) > 3 else job

# Parse all log files
all_data = []

for fold in range(n_folds):
    fname = f"{job}_{fold}.log"
    try:
        with open(fname) as fh:
            lines = fh.readlines()
    except FileNotFoundError:
        print(f"Warning: {fname} not found, skipping")
        continue

    train_loss, val_loss = [], []
    train_hr, val_hr = [], []
    for line in lines:
        m = re.match(
            r'Epoch \d+/\d+ \| Train Loss: ([\d.]+), Hit Rate: ([\d.]+) \| Val Loss: ([\d.]+), Hit Rate: ([\d.]+)',
            line
        )
        if m:
            train_loss.append(float(m.group(1)))
            train_hr.append(float(m.group(2)))
            val_loss.append(float(m.group(3)))
            val_hr.append(float(m.group(4)))

    if train_loss:
        all_data.append({
            'train_loss': np.array(train_loss),
            'val_loss': np.array(val_loss),
            'train_hr': np.array(train_hr),
            'val_hr': np.array(val_hr),
        })

if not all_data:
    print("No data found.")
    sys.exit(1)

n_epochs = len(all_data[0]['train_loss'])
epochs = np.arange(1, n_epochs + 1)

# Compute mean and std
train_loss_mean = np.mean([d['train_loss'] for d in all_data], axis=0)
train_loss_std = np.std([d['train_loss'] for d in all_data], axis=0)
val_loss_mean = np.mean([d['val_loss'] for d in all_data], axis=0)
val_loss_std = np.std([d['val_loss'] for d in all_data], axis=0)
train_hr_mean = np.mean([d['train_hr'] for d in all_data], axis=0)
train_hr_std = np.std([d['train_hr'] for d in all_data], axis=0)
val_hr_mean = np.mean([d['val_hr'] for d in all_data], axis=0)
val_hr_std = np.std([d['val_hr'] for d in all_data], axis=0)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(epochs, train_loss_mean, 'b-', label='Train', linewidth=2)
ax.plot(epochs, val_loss_mean, 'r-', label='Val', linewidth=2)
ax.fill_between(epochs, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std, alpha=0.2, color='blue')
ax.fill_between(epochs, val_loss_mean - val_loss_std, val_loss_mean + val_loss_std, alpha=0.2, color='red')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title(f'{job.upper()} Loss ({len(all_data)}-fold mean ± std)')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(epochs, train_hr_mean, 'b-', label='Train', linewidth=2)
ax.plot(epochs, val_hr_mean, 'r-', label='Val', linewidth=2)
ax.fill_between(epochs, train_hr_mean - train_hr_std, train_hr_mean + train_hr_std, alpha=0.2, color='blue')
ax.fill_between(epochs, val_hr_mean - val_hr_std, val_hr_mean + val_hr_std, alpha=0.2, color='red')
ax.set_xlabel('Epoch')
ax.set_ylabel('Hit Rate')
ax.set_title(f'{job.upper()} Hit Rate ({len(all_data)}-fold mean ± std)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_png = f"{out_prefix}_training_plot.png"
plt.savefig(out_png, dpi=150)
plt.close()
print(f"Saved {out_png}")

# Print summary
best_vals = [max(d['val_hr']) for d in all_data]
print(f"\n{job.upper()} Best Val HR per fold: {[f'{v:.4f}' for v in best_vals]}")
print(f"{job.upper()} Mean ± Std: {np.mean(best_vals):.4f} ± {np.std(best_vals):.4f}")

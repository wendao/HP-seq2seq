#!/bin/bash
# 5-fold cross-validation for the Transformer in train.py.
#
# Hyperparameters live at the top of train.py — this script only sweeps the fold.
# Device is auto-selected inside train.py (CUDA > MPS > CPU); on Apple silicon
# it picks MPS. Logs go to logs_trf/, one file per fold.
#
#   bash run_transformer.sh            # all 5 folds
#   bash run_transformer.sh 0          # single fold (smoke test)

set -u

LOG_DIR=logs_trf
FOLDS=${@:-0 1 2 3 4}

# Let unimplemented MPS kernels fall back to CPU instead of hard-failing.
export PYTORCH_ENABLE_MPS_FALLBACK=1

mkdir -p "$LOG_DIR"

for fold in $FOLDS; do
  echo "=== fold $fold ==="
  start=$(date +%s)
  python train.py "$fold" > "$LOG_DIR/trf_grp_${fold}.log" 2>&1
  status=$?
  elapsed=$(( $(date +%s) - start ))
  if [ $status -ne 0 ]; then
    echo "  FAILED (exit $status) after ${elapsed}s — see $LOG_DIR/trf_grp_${fold}.log"
    tail -n 5 "$LOG_DIR/trf_grp_${fold}.log"
  else
    echo "  $(grep 'Best validation hit rate' "$LOG_DIR/trf_grp_${fold}.log")  [${elapsed}s]"
  fi
done

echo
echo "=== summary ==="
python3 - "$LOG_DIR" <<'PY'
import re, sys, glob, statistics, os

log_dir = sys.argv[1]
rows = []
for path in sorted(glob.glob(os.path.join(log_dir, "trf_grp_*.log"))):
    text = open(path).read()
    val = re.search(r"Best validation hit rate: ([\d.]+)", text)
    trn = re.search(r"Train hit rate at best epoch: ([\d.]+)", text)
    if val:
        rows.append((os.path.basename(path), float(val.group(1)),
                     float(trn.group(1)) if trn else float("nan")))

if not rows:
    print("no completed runs found in", log_dir)
    sys.exit(1)

print(f"{'log':<20} {'val':>8} {'train':>8}")
for name, v, t in rows:
    print(f"{name:<20} {v:>8.4f} {t:>8.4f}")

vals = [v for _, v, _ in rows]
mean = statistics.mean(vals)
std = statistics.stdev(vals) if len(vals) > 1 else 0.0
print(f"\nval hit rate: mean={mean:.4f}  std={std:.4f}  n={len(vals)}")
print(f"range: [{min(vals):.4f}, {max(vals):.4f}]")
PY

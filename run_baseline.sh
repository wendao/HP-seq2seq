#!/bin/bash
# Baseline 5-fold CV for train_baseline.py.
#
# Default matrix: {rnn, lstm, cnn} x {random, grouped} x {no-CA, CA} x 5 folds = 60 runs.
# Measured on M-series MPS at 100 epochs: RNN ~415s/run, RNN+CA ~457s, full matrix ~8h.
#
# The transformer is deliberately NOT in the default matrix. train_baseline.py still
# supports --model transformer, but at these settings (1 layer, plain Adam, no LR
# schedule, 100 epochs) it does not reach convergence, so its score measures the
# training budget rather than the architecture. Run it only with a schedule and enough
# epochs to converge:  MODELS=transformer EPOCHS=400 bash run_baseline.sh
# For a transformer result on this task, use train.py / run_transformer.sh instead.
#
#   bash run_baseline.sh                                  # everything
#   MODELS="lstm rnn" SPLITS=grouped CA=0 bash run_baseline.sh
#   FOLDS=0 bash run_baseline.sh                          # fold 0 only, quick pass
#
# Logs: logs/<model>[_ca]_<split>_<fold>.log
# Device auto-selects CUDA > MPS > CPU inside train_baseline.py.

set -u

MODELS=${MODELS:-"rnn lstm cnn"}
SPLITS=${SPLITS:-"random grouped"}
CA=${CA:-"0 8"}
FOLDS=${FOLDS:-"0 1 2 3 4"}
EPOCHS=${EPOCHS:-100}
SEED=${SEED:-42}
LOG_DIR=${LOG_DIR:-logs}

export PYTORCH_ENABLE_MPS_FALLBACK=1
mkdir -p "$LOG_DIR"

for model in $MODELS; do
  # cross-attention is built into the transformer decoder; the flag is ignored there
  ca_list=$CA
  [ "$model" = "transformer" ] && ca_list=0
  # the CNN historically used 3 layers, everything else 1
  layers=1
  [ "$model" = "cnn" ] && layers=3

  for ca in $ca_list; do
    for split in $SPLITS; do
      for fold in $FOLDS; do
        tag=$model
        [ "$ca" != "0" ] && tag="${model}_ca"
        log="$LOG_DIR/${tag}_${split}_${fold}.log"

        echo "=== $tag / $split / fold $fold ==="
        start=$(date +%s)
        python train_baseline.py \
          --model "$model" --split "$split" --fold "$fold" \
          --epochs "$EPOCHS" --seed "$SEED" \
          --num_layers "$layers" --hidden_size 128 --embed_dim 64 \
          --num_heads 4 --cross_attn "$ca" > "$log" 2>&1
        status=$?
        elapsed=$(( $(date +%s) - start ))
        if [ $status -ne 0 ]; then
          echo "  FAILED (exit $status) after ${elapsed}s - see $log"
          tail -n 5 "$log"
        else
          echo "  $(grep 'Best validation hit rate' "$log")  [${elapsed}s]"
        fi
      done
    done
  done
done

echo
echo "=== summary ==="
python3 - "$LOG_DIR" <<'PY'
import re, sys, glob, os, statistics
from collections import defaultdict

log_dir = sys.argv[1]
groups = defaultdict(dict)
for path in sorted(glob.glob(os.path.join(log_dir, "*_*_?.log"))):
    m = re.match(r"(.+)_(random|grouped)_(\d)\.log$", os.path.basename(path))
    if not m:
        continue
    text = open(path).read()
    v = re.search(r"Best validation hit rate: ([\d.]+)", text)
    if v:
        groups[(m.group(1), m.group(2))][int(m.group(3))] = float(v.group(1))

if not groups:
    print("no completed runs found in", log_dir)
    sys.exit(0)

print(f"{'model':<16}{'split':<10}" + "".join(f"{'f%d' % i:>9}" for i in range(5)) + f"{'mean':>9}{'n':>4}")
for (tag, split) in sorted(groups):
    d = groups[(tag, split)]
    cells = "".join(f"{d[i]:>9.4f}" if i in d else f"{'--':>9}" for i in range(5))
    vals = list(d.values())
    print(f"{tag:<16}{split:<10}{cells}{statistics.mean(vals):>9.4f}{len(vals):>4}")
PY

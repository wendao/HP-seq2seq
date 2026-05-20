#!/bin/bash
# Run baseline experiments: 1 layer, no cross-attention, 100 epochs

mkdir -p logs

for model in rnn lstm cnn; do
  for fold in 0 1 2 3 4; do
    echo "Running $model fold $fold ..."
    python train_baseline.py --model $model --fold $fold --epochs 100 > logs/${model}_${fold}.log
  done
done

echo "All experiments done."

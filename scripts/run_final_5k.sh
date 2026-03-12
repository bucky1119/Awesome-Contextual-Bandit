#!/bin/bash
set -e

PYTHON_EXEC="/home/csg/miniconda3/envs/bandits/bin/python"

echo "============================================="
echo "1. Running Baselines (ag_news TF-IDF/SVD)"
echo "============================================="
$PYTHON_EXEC run_dataset_step.py baselines \
  --datasets ag_news \
  --baselines linucb ucb1 \
  --n_rounds 5000 \
  --cold_start_n 1000 \
  --seed 42

echo "============================================="
echo "2. Running OurMethod (ag_news TEXT)"
echo "============================================="
$PYTHON_EXEC run_dataset_step.py ourmethod \
  --datasets ag_news \
  --n_rounds 5000 \
  --cold_start_n 1000 \
  --z_dim 32 \
  --offline_freq 200 \
  --offline_epochs 20 \
  --warmup_epochs 20 \
  --model qwen2_5_7b \
  --seed 42

echo "============================================="
echo "3. Plotting Results (Regret & Reward curves)"
echo "============================================="
$PYTHON_EXEC run_dataset_step.py plot \
  --datasets ag_news \
  --algorithms linucb ucb1 ourmethod:qwen2_5_7b \
  --n_rounds 5000 \
  --seed 42

echo "============================================="
echo "4. Analyzing OurMethod Telemetry"
echo "============================================="
# Find the latest OurMethod run dir
LATEST_DIR=$(ls -td results/ag_news/ourmethod_qwen2_5_7b_n5000_seed42_* | head -1)
if [ -n "$LATEST_DIR" ] && [ -f "$LATEST_DIR/decisions.jsonl" ]; then
    echo "Found log: $LATEST_DIR/decisions.jsonl"
    $PYTHON_EXEC scripts/analyze_5k_log.py "$LATEST_DIR/decisions.jsonl" "$LATEST_DIR"
else
    echo "Could not find decisions.jsonl"
fi

echo "All complete."

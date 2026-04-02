# MovieLens Stage-1 Candidate-Set Bandit

This directory provides an independent MovieLens-1M stage-1 pipeline for candidate-set contextual bandit experiments.

## 1) Why old classification interfaces are not reused

Existing modules were designed for classification-style contextual bandits:

1. `bandits/data/data_sampler.py` builds `dataset=[context | one-hot reward]` samples, not per-round candidate sets with per-candidate joint features.
2. Existing LinUCB/Neural-LinUCB implementations assume a single context and fixed global actions.
3. MovieLens stage-1 requires per-round input `[K, d]` (here `K=5`) where candidates change every round.

Therefore stage-1 is implemented in this standalone directory without changing old interfaces.

## 2) Stage-1 task definition

- Positive feedback: `rating >= 4`
- Negative feedback: `rating <= 2`
- Drop rule: `rating == 3`
- For each positive record, construct one round:
  - 1 positive movie
  - 4 negative movies sampled from the same user's low-rating pool
  - shuffle candidate order and reward order together
  - reward vector has exactly one `1` and four `0`

## 3) Features and final dimension

Stage-1 uses static features only.

- User static feature:
  - Gender: 2-dim one-hot (`F/M`)
  - Age: 7-dim one-hot
  - Occupation: 21-dim one-hot
  - Zip-code not used
- Movie static feature:
  - Genres: 18-dim multi-hot
  - Year: extracted from title and min-max normalized to `[0, 1]`
  - title text not used

Final candidate feature:

- `x_(t,a) = concat(user_static(u), movie_static(a))`
- `feature_dim = (2 + 7 + 21) + (18 + 1) = 49`

## 4) K=5 meaning and stage-1 simplification

`K=5` means each round has exactly 5 candidates (1 positive + 4 negatives). The selected action is the candidate index in the current round.

Stage-1 simplification: negatives are sampled from the user's global low-rating pool (not strictly only from history before current timestamp). This is kept intentionally simple for stage-1 static-feature validation. Future history-aware stages should avoid potential temporal leakage.

## 5) Shared-model algorithm design

### Shared LinUCB

- One shared parameter set for all candidates in all rounds:
  - `A_inv`, `b`, `theta_hat`
- Score each candidate with shared UCB:
  - `p_a = x_a^T theta_hat + alpha * sqrt(x_a^T A_inv x_a)`
- Choose argmax candidate index.
- Update using only the chosen candidate and observed reward.

### Shared Neural-LinUCB

- Network maps one candidate feature to latent vector `z(x)`.
- No action-specific output head.
- Shared LinUCB head in latent space:
  - one shared `A_inv`, `b`, `theta_hat`
- Supports periodic network retraining and rebuilding shared linear head from replay buffer.

## 6) Files

- `prepare_movielens_stage1.py`: build dataset `datasets/movielens_stage1_k5.npz`
- `dataset.py`: dataset wrapper (`__len__`, `get_round`)
- `linucb_shared_candidate.py`: shared-parameter LinUCB
- `neural_linucb_shared_candidate.py`: shared-parameter Neural-LinUCB
- `run_movielens_stage1.py`: unified CLI runner
- `utils.py`: seed/json/dir helpers

## 7) Dataset output format

Generated file: `datasets/movielens_stage1_k5.npz`

- `candidate_features`: `[N, 5, 49]`
- `reward_vectors`: `[N, 5]`
- `positive_indices`: `[N]`
- `user_ids`: `[N]`
- `movie_ids`: `[N, 5]`
- `feature_dim`
- `K`
- `feature_schema_json`

## 8) Reproducibility

Set seed in both preprocessing and training command (`--seed`):

- negative sampling
- candidate shuffling
- numpy/python random
- torch initialization and training randomness

## 9) Usage

### Step A: prepare dataset

```bash
python movielens_bandit_stage1/prepare_movielens_stage1.py \
  --users datasets/raw/ml-1m/users.dat \
  --movies datasets/raw/ml-1m/movies.dat \
  --ratings datasets/raw/ml-1m/ratings.dat \
  --output datasets/movielens_stage1_k5.npz \
  --seed 42
```

### Step B1: run one algorithm

```bash
python movielens_bandit_stage1/run_movielens_stage1.py run \
  --algorithms linucb_shared \
  --data_path datasets/movielens_stage1_k5.npz \
  --n_rounds 5000 \
  --seed 42 \
  --alpha 1.0 \
  --lambda_prior 1.0
```

### Step B2: run multiple algorithms in one experiment directory

```bash
python movielens_bandit_stage1/run_movielens_stage1.py run \
  --algorithms linucb_shared neural_linucb_shared \
  --data_path datasets/movielens_stage1_k5.npz \
  --n_rounds 5000 \
  --seed 42 \
  --alpha 1.0 \
  --lambda_prior 1.0 \
  --latent_dim 32 \
  --hidden_size 64 \
  --hidden_layers 2 \
  --train_every 100 \
  --epochs 5 \
  --lr 1e-3 \
  --buffer_size 10000
```

### Step B3: re-plot from existing experiment

```bash
python movielens_bandit_stage1/run_movielens_stage1.py plot \
  --exp_dir results/movielens_stage1/<your_experiment_dir> \
  --rolling_window 100
```

Or find latest run by `n_rounds + seed`:

```bash
python movielens_bandit_stage1/run_movielens_stage1.py plot \
  --n_rounds 5000 \
  --seed 42
```

## 10) Result directory and generated artifacts

Output follows project `results/` style and is isolated under MovieLens stage-1:

- `results/movielens_stage1/YYYYMMDD_HHMMSS_movielens_stage1_k5_<n_rounds>r_seed<seed>_<algo1>_<algo2>.../`

Auto-generated files:

- `run.log`: runtime log with periodic metrics
- `run_config.json`: CLI and run configuration
- `<algo>.npz`: per-algorithm trajectory log
- `summary.json`: group final metrics
- `run_summary.md`: one markdown report for all algorithms in this run
- `metrics.csv`: per-round metrics (all algorithms in one table)
- `cumulative_regret_curve.png`
- `average_reward_curve.png`
- `accuracy_curve.png`

Metric definitions:

- instantaneous reward: `chosen_reward`
- instantaneous regret: `1 - chosen_reward`
- cumulative regret: sum of instantaneous regret
- average reward: mean reward
- accuracy: hit rate of selecting positive candidate

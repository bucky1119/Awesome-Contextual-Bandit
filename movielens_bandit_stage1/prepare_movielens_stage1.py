# /home/csg/Awesome-contextual-bandits/movielens_bandit_stage1/prepare_movielens_stage1.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


RAW_DIR = Path("/home/csg/Awesome-contextual-bandits/datasets/raw/ml-1m")
OUT_DIR = Path("/home/csg/Awesome-contextual-bandits/datasets")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MOVIES_PATH = RAW_DIR / "movies.dat"
RATINGS_PATH = RAW_DIR / "ratings.dat"

# ===== first-version environment settings =====
NUM_ARMS = 200
MIN_RATINGS_PER_USER = 200
SEED = 42
MAX_ROUNDS = None  # None = use all eligible movies as rounds

GENRE_LIST = [
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]
GENRE_TO_IDX = {g: i for i, g in enumerate(GENRE_LIST)}
GENRE_DIM = len(GENRE_LIST)  # MovieLens 1M has 18 genres in movies.dat


def load_movies(movies_path: Path) -> pd.DataFrame:
    movies = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        header=None,
        names=["MovieID", "Title", "Genres"],
        encoding="latin-1",
    )
    return movies


def load_ratings(ratings_path: Path) -> pd.DataFrame:
    ratings = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        header=None,
        names=["UserID", "MovieID", "Rating", "Timestamp"],
        encoding="latin-1",
    )
    return ratings


def build_movie_genre_map(movies: pd.DataFrame) -> Dict[int, np.ndarray]:
    """
    Build movie -> 18-dim multi-hot genre vector.
    """
    movie_genre_map: Dict[int, np.ndarray] = {}

    for _, row in movies.iterrows():
        mid = int(row["MovieID"])
        vec = np.zeros(GENRE_DIM, dtype=np.float32)

        genres = str(row["Genres"]).split("|")
        for g in genres:
            if g in GENRE_TO_IDX:
                vec[GENRE_TO_IDX[g]] = 1.0

        movie_genre_map[mid] = vec

    return movie_genre_map


def select_active_users(
    ratings: pd.DataFrame,
    num_arms: int,
    min_ratings_per_user: int,
    seed: int,
) -> np.ndarray:
    """
    Randomly select fixed users as arms.
    """
    user_counts = ratings.groupby("UserID").size()
    eligible_users = user_counts[user_counts > min_ratings_per_user].index.to_numpy(dtype=np.int64)

    if len(eligible_users) < num_arms:
        raise ValueError(
            f"Not enough eligible users: need {num_arms}, got {len(eligible_users)} "
            f"with > {min_ratings_per_user} ratings."
        )

    rng = np.random.default_rng(seed)
    selected_users = rng.choice(eligible_users, size=num_arms, replace=False)
    selected_users = np.sort(selected_users.astype(np.int64))
    return selected_users


def build_user_preference_vectors(
    ratings: pd.DataFrame,
    movie_genre_map: Dict[int, np.ndarray],
    selected_users: np.ndarray,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Build user preference vectors as average ratings over genres.

    For each selected user u:
        pref_u[g] = average rating the user gave to movies containing genre g
    If user has never rated a genre, pref_u[g] = 0.

    Returns:
        arm_features: [num_arms, genre_dim]
        user_to_arm_idx: mapping user_id -> arm index
    """
    selected_user_set = set(int(u) for u in selected_users)

    # Keep only rows for selected users and movies with known genre vectors
    ratings = ratings[ratings["UserID"].isin(selected_user_set)].copy()
    ratings = ratings[ratings["MovieID"].isin(movie_genre_map.keys())].copy()

    num_arms = len(selected_users)
    arm_features = np.zeros((num_arms, GENRE_DIM), dtype=np.float32)
    user_to_arm_idx = {int(uid): i for i, uid in enumerate(selected_users)}

    # accumulate sums and counts per genre
    rating_sum = np.zeros((num_arms, GENRE_DIM), dtype=np.float64)
    rating_count = np.zeros((num_arms, GENRE_DIM), dtype=np.float64)

    for _, row in ratings.iterrows():
        uid = int(row["UserID"])
        mid = int(row["MovieID"])
        rating = float(row["Rating"])

        arm_idx = user_to_arm_idx[uid]
        genre_vec = movie_genre_map[mid]  # multi-hot

        active = genre_vec > 0
        rating_sum[arm_idx, active] += rating
        rating_count[arm_idx, active] += 1.0

    mask = rating_count > 0
    arm_features[mask] = (rating_sum[mask] / rating_count[mask]).astype(np.float32)
    return arm_features, user_to_arm_idx


def build_round_contexts(
    movies: pd.DataFrame,
    movie_genre_map: Dict[int, np.ndarray],
    seed: int,
    max_rounds: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Each round corresponds to one arriving movie.
    Context is the movie's genre multi-hot vector.

    Returns:
        movie_ids: [T]
        contexts: [T, genre_dim]
    """
    eligible_movie_ids: List[int] = []
    context_list: List[np.ndarray] = []

    for _, row in movies.iterrows():
        mid = int(row["MovieID"])
        if mid not in movie_genre_map:
            continue
        genre_vec = movie_genre_map[mid]
        if np.sum(genre_vec) == 0:
            continue
        eligible_movie_ids.append(mid)
        context_list.append(genre_vec)

    movie_ids = np.asarray(eligible_movie_ids, dtype=np.int64)
    contexts = np.stack(context_list, axis=0).astype(np.float32)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(movie_ids))
    movie_ids = movie_ids[perm]
    contexts = contexts[perm]

    if max_rounds is not None:
        movie_ids = movie_ids[:max_rounds]
        contexts = contexts[:max_rounds]

    return movie_ids, contexts


def compute_expected_reward_matrix(
    arm_features: np.ndarray,
    contexts: np.ndarray,
) -> np.ndarray:
    """
    Compute mu_{t,i} following the paper-inspired idea.

    arm_features[i] : user preference vector, shape [genre_dim]
    contexts[t]     : current movie genre vector, shape [genre_dim]

    We use:
        overlap = number of active shared genres
        score   = <g_t, p_i> / (2 * overlap)
        mu      = 2 / (1 + exp(-score)) - 1

    If overlap == 0, set mu = 0.

    Returns:
        mu_matrix: [T, num_arms], values in [0, 1)
    """
    T = contexts.shape[0]
    N = arm_features.shape[0]

    mu_matrix = np.zeros((T, N), dtype=np.float32)

    for t in range(T):
        g = contexts[t]  # [genre_dim], multi-hot
        for i in range(N):
            p = arm_features[i]  # [genre_dim], average ratings over genres

            overlap_mask = (g > 0) & (p > 0)
            overlap = int(np.sum(overlap_mask))

            if overlap == 0:
                mu = 0.0
            else:
                score = float(np.dot(g, p)) / float(2.0 * overlap)
                mu = 2.0 / (1.0 + np.exp(-score)) - 1.0

            # numerical clipping
            mu_matrix[t, i] = np.float32(np.clip(mu, 0.0, 1.0))

    return mu_matrix


def main():
    movies = load_movies(MOVIES_PATH)
    ratings = load_ratings(RATINGS_PATH)

    movie_genre_map = build_movie_genre_map(movies)

    selected_users = select_active_users(
        ratings=ratings,
        num_arms=NUM_ARMS,
        min_ratings_per_user=MIN_RATINGS_PER_USER,
        seed=SEED,
    )

    arm_features, user_to_arm_idx = build_user_preference_vectors(
        ratings=ratings,
        movie_genre_map=movie_genre_map,
        selected_users=selected_users,
    )

    movie_ids, contexts = build_round_contexts(
        movies=movies,
        movie_genre_map=movie_genre_map,
        seed=SEED,
        max_rounds=MAX_ROUNDS,
    )

    mu_matrix = compute_expected_reward_matrix(
        arm_features=arm_features,
        contexts=contexts,
    )

    out_path = OUT_DIR / f"movielens_paper_env_{NUM_ARMS}arms_seed{SEED}.npz"

    feature_schema = {
        "environment_version": "paper_style_first_version",
        "arm_definition": "fixed users as arms",
        "num_arms": int(NUM_ARMS),
        "user_selection_rule": f"randomly sampled users with > {MIN_RATINGS_PER_USER} ratings",
        "round_definition": "each round corresponds to one arriving movie",
        "context_definition": "movie genre multi-hot vector",
        "arm_feature_definition": "user genre preference vector (average rating over genres)",
        "genre_dim": int(GENRE_DIM),
        "note": (
            "The reference paper describes 20-genre preference vectors, "
            "but MovieLens 1M movies.dat contains 18 genres, so this environment uses 18 dimensions."
        ),
        "mu_definition": (
            "mu_{t,i} = 2 / (1 + exp(- <g_t, p_i> / (2 * overlap))) - 1, "
            "with mu=0 when overlap=0."
        ),
        "regret_definition_single_arm": "max_i mu[t,i] - mu[t,a_t]",
    }

    np.savez_compressed(
        out_path,
        selected_user_ids=selected_users.astype(np.int64),   # [N]
        arm_features=arm_features.astype(np.float32),        # [N, genre_dim]
        movie_ids=movie_ids.astype(np.int64),                # [T]
        contexts=contexts.astype(np.float32),                # [T, genre_dim]
        mu_matrix=mu_matrix.astype(np.float32),              # [T, N]
        num_arms=np.array([NUM_ARMS], dtype=np.int64),
        genre_dim=np.array([GENRE_DIM], dtype=np.int64),
        feature_schema_json=np.array([json.dumps(feature_schema)], dtype=object),
    )

    print(f"Saved paper-style MovieLens environment to: {out_path}")
    print(f"selected_user_ids shape: {selected_users.shape}")
    print(f"arm_features shape: {arm_features.shape}")
    print(f"contexts shape: {contexts.shape}")
    print(f"mu_matrix shape: {mu_matrix.shape}")
    print(f"genre_dim: {GENRE_DIM}")


if __name__ == "__main__":
    main()
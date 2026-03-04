"""
Prepare the 20 Newsgroups semantic dataset for contextual bandit experiments.

This script:
1. Downloads the 20 Newsgroups dataset (6 diverse categories)
2. Applies TF-IDF vectorization + TruncatedSVD (LSA) for 50-dim semantic features
3. Converts classification → bandit format (correct class → reward 1, otherwise 0)
4. Saves pre-processed data as datasets/newsgroups.npz

Categories selected (6 diverse topics):
  0: comp.graphics        (computers)
  1: rec.sport.baseball   (sports)
  2: sci.med              (medicine)
  3: sci.space            (space)
  4: talk.politics.guns   (politics)
  5: soc.religion.christian (religion)
"""

import ssl
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import os

# ── bypass SSL issues on macOS ──────────────────────────────────────────
ssl._create_default_https_context = ssl._create_unverified_context

# ── config ──────────────────────────────────────────────────────────────
CATEGORIES = [
    'comp.graphics',
    'rec.sport.baseball',
    'sci.med',
    'sci.space',
    'talk.politics.guns',
    'soc.religion.christian',
]
CONTEXT_DIM = 50      # LSA dimensions
MAX_TFIDF_FEATURES = 10000
OUTPUT_DIR = 'datasets'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'newsgroups.npz')

# ── download & vectorise ────────────────────────────────────────────────
print("Fetching 20 Newsgroups (6 categories) ...")
data = fetch_20newsgroups(subset='all', categories=CATEGORIES,
                          shuffle=True, random_state=42,
                          remove=('headers', 'footers', 'quotes'))

print(f"  Total documents : {len(data.data)}")
print(f"  Categories      : {data.target_names}")

# TF-IDF
print(f"  TF-IDF (max_features={MAX_TFIDF_FEATURES}) ...")
tfidf = TfidfVectorizer(max_features=MAX_TFIDF_FEATURES,
                         stop_words='english', dtype=np.float32)
X_tfidf = tfidf.fit_transform(data.data)

# LSA (TruncatedSVD) for dimensionality reduction
print(f"  TruncatedSVD → {CONTEXT_DIM} dims ...")
svd = TruncatedSVD(n_components=CONTEXT_DIM, random_state=42)
X_lsa = svd.fit_transform(X_tfidf)
explained_var = svd.explained_variance_ratio_.sum()
print(f"  Explained variance: {explained_var:.2%}")

# standardize features
scaler = StandardScaler()
contexts = scaler.fit_transform(X_lsa).astype(np.float32)

# ── remap labels to 0 .. K-1 ───────────────────────────────────────────
labels = data.target.astype(int)
num_actions = len(CATEGORIES)
assert labels.min() == 0 and labels.max() == num_actions - 1

# ── convert to bandit format ────────────────────────────────────────────
n = len(labels)
rewards = np.zeros((n, num_actions), dtype=np.float32)
rewards[np.arange(n), labels] = 1.0
opt_rewards = np.ones(n, dtype=np.float32)
opt_actions = labels.copy()

# dataset = [contexts | rewards]  shape (n, context_dim + num_actions)
dataset = np.hstack((contexts, rewards))

# ── save ────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.savez_compressed(OUTPUT_FILE,
                    dataset=dataset,
                    opt_rewards=opt_rewards,
                    opt_actions=opt_actions,
                    context_dim=CONTEXT_DIM,
                    num_actions=num_actions,
                    categories=np.array(CATEGORIES))

print(f"\nSaved to {OUTPUT_FILE}")
print(f"  dataset shape  : {dataset.shape}  (n={n}, context_dim={CONTEXT_DIM}, num_actions={num_actions})")
print(f"  opt_rewards    : {opt_rewards.shape}")
print(f"  opt_actions    : {opt_actions.shape}")
print(f"  Label distribution:")
for i, cat in enumerate(CATEGORIES):
    count = (labels == i).sum()
    print(f"    {i}: {cat:30s} → {count:5d} samples")
print("Done.")

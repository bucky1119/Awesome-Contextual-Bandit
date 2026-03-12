import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def main():
    in_path = "datasets/ag_news.npz"
    out_path = "datasets/ag_news_tfidf_svd100.npz"
    
    print(f"Loading {in_path}...")
    d = np.load(in_path, allow_pickle=True)
    texts = d['texts']
    opt_rewards = d['opt_rewards']
    opt_actions = d['opt_actions']
    num_actions = int(d['num_actions'])
    dataset_orig = d['dataset']
    
    n_samples = len(texts)
    print(f"n_samples: {n_samples}")
    
    print("Applying TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1,2))
    X_tfidf = vectorizer.fit_transform(texts)
    
    print("Applying TruncatedSVD...")
    svd = TruncatedSVD(n_components=100, random_state=42)
    X_svd = svd.fit_transform(X_tfidf).astype(np.float32)
    
    # Original reward one-hot
    reward_onehot = dataset_orig[:, -num_actions:]
    
    # New dataset
    dataset_new = np.concatenate([X_svd, reward_onehot], axis=1)
    print(f"New dataset shape: {dataset_new.shape}")
    
    # Check
    print(f"context_dim == 100 ? {dataset_new.shape[1] - num_actions == 100}")
    print("Sample X_svd (first 3 rows, first 10 dims):")
    print(X_svd[:3, :10])
    
    print("Sample reward_onehot (first 3 rows):")
    print(reward_onehot[:3])
    
    print(f"Saving to {out_path}...")
    np.savez_compressed(
        out_path,
        dataset=dataset_new,
        opt_rewards=opt_rewards,
        opt_actions=opt_actions,
        num_actions=num_actions,
        texts=texts
    )
    print("Done!")

if __name__ == "__main__":
    main()

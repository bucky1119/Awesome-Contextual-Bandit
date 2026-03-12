import numpy as np
import sys

def main():
    path = "datasets/ag_news_tfidf_svd100.npz"
    try:
        d = np.load(path, allow_pickle=True)
    except FileNotFoundError:
        print(f"[ERROR] {path} not found.")
        sys.exit(1)
        
    dataset = d['dataset']
    opt_actions = d['opt_actions']
    opt_rewards = d['opt_rewards']
    num_actions = int(d['num_actions'])
    texts = d['texts']
    
    n_samples = len(texts)
    context_dim = dataset.shape[1] - num_actions
    
    print("=== Step 0: Check Dataset ===")
    print(f"Path: {path}")
    print(f"n_samples: {n_samples}")
    print(f"context_dim: {context_dim}")
    print(f"num_actions: {num_actions}")
    
    print(f"\ntexts[0] (first 200 chars): {str(texts[0])[:200]}")
    print(f"\nopt_actions (first 10): {opt_actions[:10]}")
    
    print("\nreward one-hot check (random 3 samples):")
    np.random.seed(42)
    indices = np.random.choice(n_samples, 3, replace=False)
    for i in indices:
        row_rewards = dataset[i, -num_actions:]
        action = opt_actions[i]
        print(f"Index {i}, label {action}: {row_rewards}")

if __name__ == "__main__":
    main()
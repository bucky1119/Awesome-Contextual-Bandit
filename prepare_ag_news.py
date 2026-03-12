import os
import sys
import numpy as np
import argparse
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(description="Prepare AG News dataset for contextual bandits.")
    parser.add_argument("--max_samples", type=int, default=20000, help="Max number of samples to process.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")
    parser.add_argument("--out", type=str, default="datasets/ag_news.npz", help="Output file path.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    
    # Create directory if it doesn't exist
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print("Loading AG News dataset from huggingface...")
    try:
        ds = load_dataset("ag_news", split="train")
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        sys.exit(1)

    print(f"Original dataset size: {len(ds)}")
    
    # Shuffle and subset
    ds = ds.shuffle(seed=args.seed)
    if args.max_samples < len(ds):
        ds = ds.select(range(args.max_samples))
        print(f"Subset size mapped to: {args.max_samples}")

    texts = []
    labels = []
    
    for item in ds:
        texts.append(item['text'].replace('\\', ' '))
        labels.append(item['label'])

    texts = np.array(texts, dtype=object)
    labels = np.array(labels, dtype=int)
    
    n = len(texts)
    num_actions = 4
    context_dim = 15  # minimal placeholder required
    
    # In this dataset, the explicit floating features are ignored by OurMethod LLM
    # since we will pass the actual text. We'll populate placeholder zeroes.
    feature_matrix = np.zeros((n, context_dim), dtype=np.float32)
    
    # Rewards matrix: one-hot for the correct label
    rewards_matrix = np.zeros((n, num_actions), dtype=np.float32)
    rewards_matrix[np.arange(n), labels] = 1.0
    
    # Combine feature_matrix and rewards_matrix as (n, context_dim + num_actions)
    dataset_matrix = np.concatenate([feature_matrix, rewards_matrix], axis=1)
    
    opt_actions = labels
    opt_rewards = np.ones(n, dtype=np.float32)
    
    # Save the prepared npz
    print(f"Saving to {args.out} ...")
    np.savez_compressed(
        args.out,
        dataset=dataset_matrix,
        opt_rewards=opt_rewards,
        opt_actions=opt_actions,
        num_actions=num_actions,
        texts=texts
    )
    print("Done! Dataset prepared.")

if __name__ == "__main__":
    main()
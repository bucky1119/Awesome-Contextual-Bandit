import numpy as np

def main():
    d = np.load("datasets/ag_news.npz", allow_pickle=True)
    print("Files:", d.files)
    dataset = d['dataset']
    opt_actions = d['opt_actions']
    opt_rewards = d['opt_rewards']
    num_actions = int(d['num_actions'])
    texts = d['texts']
    
    n_samples, total_dim = dataset.shape
    context_dim = total_dim - num_actions
    
    print(f"n_samples={n_samples}, context_dim={context_dim}, num_actions={num_actions}")
    print(f"\ntexts[0] (first 200 chars):\n{texts[0][:200]}...")
    print(f"\nopt_actions (first 10): {opt_actions[:10]}")
    
    print("\nChecking randomly selected rewards:")
    import random
    for i in random.sample(range(n_samples), 3):
        print(f"Row {i} - action {opt_actions[i]}: features: {dataset[i, :context_dim]}, rewards part: {dataset[i, context_dim:]}")
        assert dataset[i, context_dim + opt_actions[i]] == 1.0
        
if __name__ == "__main__":
    main()

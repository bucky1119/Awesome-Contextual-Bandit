import numpy as np

## python3 check_ag_news_npz.py

def inspect_dataset(path, name, expected_context_dim=None, print_n=10):
    print("\n" + "=" * 80)
    print(f"[{name}] {path}")
    print("=" * 80)

    d = np.load(path, allow_pickle=True)
    print("keys:", d.files)

    dataset = d["dataset"]
    opt_rewards = d["opt_rewards"]
    opt_actions = d["opt_actions"]
    num_actions = int(d["num_actions"])
    texts = d["texts"]

    n = len(texts)
    context_dim = dataset.shape[1] - num_actions

    print(f"dataset shape      : {dataset.shape}")
    print(f"opt_rewards shape  : {opt_rewards.shape}")
    print(f"opt_actions shape  : {opt_actions.shape}")
    print(f"texts shape        : {texts.shape}")
    print(f"num_actions        : {num_actions}")
    print(f"context_dim(calc)  : {context_dim}")

    ok = True
    if not (len(dataset) == len(opt_rewards) == len(opt_actions) == len(texts)):
        print("[ERROR] 长度不一致")
        ok = False

    reward_onehot = dataset[:, -num_actions:]
    row_sums = reward_onehot.sum(axis=1)
    if not np.allclose(row_sums, 1.0):
        print("[ERROR] reward one-hot 每行和不为1")
        ok = False

    if not np.array_equal(np.argmax(reward_onehot, axis=1), opt_actions):
        print("[ERROR] reward one-hot 的 argmax 与 opt_actions 不一致")
        ok = False

    if not np.allclose(opt_rewards, 1.0):
        print("[WARN] opt_rewards 不是全 1（理论上应为全1）")

    if expected_context_dim is not None and context_dim != expected_context_dim:
        print(f"[ERROR] context_dim={context_dim}, 期望={expected_context_dim}")
        ok = False

    print("[CHECK RESULT]:", "PASS ✅" if ok else "FAIL ❌")

    k = min(print_n, n)
    print(f"\n--- 前 {k} 条样本 ---")
    for i in range(k):
        text_preview = str(texts[i]).replace("\n", " ")[:120]
        ctx_preview = dataset[i, : min(8, context_dim)] if context_dim > 0 else []
        rw = reward_onehot[i]
        print(f"[{i}] opt_action={int(opt_actions[i])}, opt_reward={float(opt_rewards[i]):.1f}")
        print(f"    text: {text_preview}")
        print(f"    ctx[:8]: {ctx_preview}")
        print(f"    reward_onehot: {rw}")


def main():
    inspect_dataset(
        "datasets/ag_news.npz",
        "AG_NEWS_RAW",
        expected_context_dim=15,
        print_n=10,
    )
    inspect_dataset(
        "datasets/ag_news_tfidf_svd100.npz",
        "AG_NEWS_TFIDF_SVD100",
        expected_context_dim=100,
        print_n=10,
    )


if __name__ == "__main__":
    main()

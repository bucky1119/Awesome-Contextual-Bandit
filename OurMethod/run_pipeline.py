#!/usr/bin/env python3
"""OurMethod 统一入口：按 Step 0‑7 组织的增量流水线。

用法:
  python OurMethod/run_pipeline.py test_step0
  python OurMethod/run_pipeline.py test_step1
  ...
  python OurMethod/run_pipeline.py test_step7
  python OurMethod/run_pipeline.py test_all          # 顺序跑 step0‑7
  python OurMethod/run_pipeline.py online_stub        # 在线 stub 跑 2000 轮
  python OurMethod/run_pipeline.py offline_train      # 离线训练 fφ
  python OurMethod/run_pipeline.py full               # online + offline 完整闭环
"""

from __future__ import annotations

import argparse
import os
import sys
import importlib
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)


def _run_test_module(name: str) -> bool:
    """运行 tests/self_check_stepX.py 并返回是否全部通过。"""
    mod_name = f"OurMethod.tests.{name}"
    try:
        mod = importlib.import_module(mod_name)
    except ImportError as e:
        print(f"[FAIL] 无法导入 {mod_name}: {e}")
        return False
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(mod)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


def cmd_test_step(step: int) -> bool:
    print(f"\n{'='*60}\n  自检 Step {step}\n{'='*60}")
    return _run_test_module(f"self_check_step{step}")


def cmd_test_all() -> bool:
    ok = True
    for s in range(8):
        if not cmd_test_step(s):
            ok = False
            print(f"\n*** Step {s} 自检未通过，后续步骤可能受影响 ***\n")
    return ok


def cmd_online_stub(n_rounds: int = 2000, seed: int = 42):
    """在线 stub 闭环：不调用真实 LLM。"""
    import numpy as np
    from OurMethod.core.protocol import Arm
    from OurMethod.core.prompt_builder import StructuredPromptBuilder
    from OurMethod.core.frozen_llm import StubFrozenLLM
    from OurMethod.core.compressor import MLPCompressor
    from OurMethod.core.combined_policy import CombinedUCBPolicy
    from OurMethod.core.online_runner import OnlineRunner
    from OurMethod.core.env_simulated import SimulatedEnvironment

    K, d = 5, 10
    env = SimulatedEnvironment(num_actions=K, context_dim=d, noise_std=0.1, seed=seed)
    llm = StubFrozenLLM(hidden_dim=64, generate_ds_llm=True)
    comp = MLPCompressor(input_dim=64, output_dim=16)
    policy = CombinedUCBPolicy(num_actions=K, z_dim=16)
    pb = StructuredPromptBuilder()
    arms = [Arm(arm_id=i, name=f"arm_{i}") for i in range(K)]

    log_path = os.path.join(LOG_DIR, "decisions.jsonl")
    # 清空旧日志
    if os.path.exists(log_path):
        os.remove(log_path)

    runner = OnlineRunner(
        prompt_builder=pb,
        llm_encoder=llm,
        compressor=comp,
        policy=policy,
        arms=arms,
        env_reward_fn=env.reward,
        env_optimal_fn=env.optimal_reward,
        context_dim=d,
        log_path=log_path,
        seed=seed,
    )
    print(f"\n在线 stub 运行 {n_rounds} 轮 ...")
    runner.run(n_rounds, verbose=True)
    s = runner.summary()
    print(f"\n结果: {s}")
    print(f"日志: {log_path}")
    return log_path


def cmd_offline_train(
    jsonl_path: Optional[str] = None,
    n_epochs: int = 30,
    ckpt_path: Optional[str] = None,
):
    """离线训练 fφ（需要先跑过 online_stub 产生 decisions.jsonl）。"""
    from OurMethod.core.protocol import Arm
    from OurMethod.core.compressor import MLPCompressor
    from OurMethod.core.combined_policy import CombinedUCBPolicy
    from OurMethod.core.offline_dataset import OfflineDataset
    from OurMethod.core.train_compressor import OfflineTrainer

    K, z_dim = 5, 16
    if jsonl_path is None:
        jsonl_path = os.path.join(LOG_DIR, "decisions.jsonl")
    if ckpt_path is None:
        ckpt_path = os.path.join(LOG_DIR, "compressor.pt")

    if not os.path.exists(jsonl_path):
        print(f"[错误] 找不到 {jsonl_path}，请先运行 online_stub。")
        return

    ds = OfflineDataset()
    ds.load_jsonl(jsonl_path)
    print(f"加载了 {len(ds)} 条记录。")

    h_dim = ds.h_list[0].shape[0] if ds.h_list else 64
    comp = MLPCompressor(input_dim=h_dim, output_dim=z_dim)
    policy = CombinedUCBPolicy(num_actions=K, z_dim=z_dim)

    # 先用数据构建 LinUCB 初始统计（模拟在线阶段的状态）
    H_all, arms_all, rewards_all = ds.as_arrays()
    z_all = comp.forward(H_all)
    policy.rebuild_linucb(z_all, arms_all, rewards_all)

    trainer = OfflineTrainer(
        compressor=comp, policy=policy, dataset=ds, batch_size=64,
    )
    print(f"\n离线训练 fφ ({n_epochs} epochs) ...")
    losses = trainer.train(n_epochs=n_epochs)
    trainer.rebuild_policy_stats()
    trainer.save_checkpoint(ckpt_path)
    print(f"\nCheckpoint 已保存: {ckpt_path}")
    print(f"最终 loss: {losses[-1]:.6f}" if losses else "无可训练数据")


def cmd_full(n_rounds: int = 2000, n_epochs: int = 30, seed: int = 42):
    """完整闭环：online → offline → 验证。"""
    log_path = cmd_online_stub(n_rounds=n_rounds, seed=seed)
    cmd_offline_train(jsonl_path=log_path, n_epochs=n_epochs)
    print("\n完整闭环完成。")


# ---- typing 仅 cmd_offline_train 用到 ---- #
from typing import Optional


def main():
    p = argparse.ArgumentParser(description="OurMethod pipeline")
    p.add_argument("command", choices=[
        "test_step0", "test_step1", "test_step2", "test_step3",
        "test_step4", "test_step5", "test_step6", "test_step7",
        "test_all", "online_stub", "offline_train", "full",
    ])
    p.add_argument("--n_rounds", type=int, default=2000)
    p.add_argument("--n_epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--jsonl", type=str, default=None)
    p.add_argument("--ckpt", type=str, default=None)
    args = p.parse_args()

    cmd = args.command
    if cmd.startswith("test_step"):
        step = int(cmd[-1])
        ok = cmd_test_step(step)
        sys.exit(0 if ok else 1)
    elif cmd == "test_all":
        ok = cmd_test_all()
        sys.exit(0 if ok else 1)
    elif cmd == "online_stub":
        cmd_online_stub(n_rounds=args.n_rounds, seed=args.seed)
    elif cmd == "offline_train":
        cmd_offline_train(jsonl_path=args.jsonl, n_epochs=args.n_epochs, ckpt_path=args.ckpt)
    elif cmd == "full":
        cmd_full(n_rounds=args.n_rounds, n_epochs=args.n_epochs, seed=args.seed)


if __name__ == "__main__":
    main()

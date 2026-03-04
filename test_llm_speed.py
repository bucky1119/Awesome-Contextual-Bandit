#!/usr/bin/env python3
"""测试 SmolLM2 的加载和推理速度"""
import time
import os
os.environ["HF_HUB_OFFLINE"] = "1"  # 使用本地缓存

print("=== 测试 SmolLM2 加载和推理速度 ===")
start = time.time()

from OurMethod.core.frozen_llm import GenerativeFrozenLLM

# SmolLM2 用 MPS 加速 (Apple Silicon)
llm = GenerativeFrozenLLM(
    model_name="HuggingFaceTB/SmolLM2-360M-Instruct",
    device="mps",
    generate_ds_llm=True
)
load_time = time.time() - start
print(f"模型加载时间: {load_time:.2f}s")
print(f"Hidden dim: {llm.get_hidden_dim()}")
print(f"Device: mps")
print()

# 测试单次推理
import numpy as np
prompt = "Context features: " + ", ".join([f"{x:.3f}" for x in np.random.randn(50)]) + "\nPredict best action from 6 choices."

print("测试 3 次推理...")
times = []
for i in range(3):
    start = time.time()
    h_t, ds_llm = llm.encode(prompt, num_arms=6)
    t = time.time() - start
    times.append(t)
    print(f"  第 {i+1} 次: {t:.2f}s")

avg_time = sum(times[1:]) / len(times[1:])  # 跳过首次(可能有预热)
print(f"\n平均推理时间(跳过首次): {avg_time:.2f}s")
print(f"h_t shape: {h_t.shape}")
print(f"\n预估 2000 轮总时间: {2000 * avg_time / 60:.1f} 分钟 = {2000 * avg_time / 3600:.2f} 小时")

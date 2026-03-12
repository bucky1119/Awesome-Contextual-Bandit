import json
import os
import sys
import numpy as np
from datetime import datetime

def analyze_ourmethod_run(log_path):
    if not os.path.exists(log_path):
        print(f"Error: {log_path} not found.")
        return

    history = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                history.append(json.loads(line))
    
    n = len(history)
    if n == 0:
        print("Log is empty.")
        return

    # 指标计算
    rewards = [rec.get("reward", 0) for rec in history]
    correct = [1 if rec.get("reward", 0) > 0.9 else 0 for rec in history] # 假设 reward=1 为正确
    
    cum_regret = 0
    regret_history = []
    # 简单模拟 regret (opt_reward=1)
    for r in rewards:
        cum_regret += (1.0 - r)
        regret_history.append(cum_regret)

    # UCB 来源统计
    sources = [rec.get("min_source", "unknown") for rec in history]
    src_counts = {s: sources.count(s) for s in set(sources)}
    
    # 统计数值均值 (Radius, UCB Mean)
    rad_s1 = [np.mean(rec["radius_linucb"]) for rec in history if "radius_linucb" in rec]
    rad_s3 = [np.mean(rec["radius_llm"]) for rec in history if "radius_llm" in rec]
    ucb_s1 = [np.mean(rec["ucb_s1"]) for rec in history if "ucb_s1" in rec]
    ucb_s3 = [np.mean(rec["ucb_s3"]) for rec in history if "ucb_s3" in rec]

    report_dir = os.path.dirname(log_path)
    report_file = os.path.join(report_dir, "ourmethod_run_summary.md")
    
    with open(report_file, "w", encoding='utf-8') as f:
        f.write(f"# OurMethod 运行阶段性总结报告\n\n")
        f.write(f"- **总执行步数**: {n}\n")
        f.write(f"- **平均奖励 (Avg. Reward)**: {np.mean(rewards):.4f}\n")
        f.write(f"- **当前累计遗憾 (Cum. Regret)**: {cum_regret:.1f}\n")
        f.write(f"- **分类准确率 (Est. Accuracy)**: {np.mean(correct):.4f}\n\n")
        
        f.write("## 1. 决策来源分布 (Selection Source Distribution)\n\n")
        for s, count in src_counts.items():
            f.write(f"- **{s}**: {count} 次 ({count/n*100:.1f}%)\n")
            
        f.write("\n## 2. 详细置信度监控 (Confidence & UCB Stats)\n\n")
        if rad_s1: f.write(f"- **LinUCB 平均半径 (Radius S1)**: {np.mean(rad_s1):.4f}\n")
        if rad_s3: f.write(f"- **LLM 平均半径 (Radius S3)**: {np.mean(rad_s3):.4f}\n")
        if ucb_s1: f.write(f"- **LinUCB 平均 UCB 分值 (Mean S1)**: {np.mean(ucb_s1):.4f}\n")
        if ucb_s3: f.write(f"- **LLM 平均 UCB 分值 (Mean S3)**: {np.mean(ucb_s3):.4f}\n")
        
        f.write("\n## 3. 运行环境与时间\n")
        f.write(f"- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
    print(f"Summary report created: {report_file}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_ourmethod_run(sys.argv[1])

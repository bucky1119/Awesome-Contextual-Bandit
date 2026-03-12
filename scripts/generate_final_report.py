import json
import os
import glob
import pandas as pd
from datetime import datetime

def generate_report(dataset="ag_news", n_rounds=5000, seed=42):
    exp_dir_pattern = f"results/{dataset}_{n_rounds}r_seed{seed}"
    exp_dirs = glob.glob(exp_dir_pattern)
    if not exp_dirs:
        print(f"No results found for {exp_dir_pattern}")
        return
    
    report_path = os.path.join(exp_dirs[0], "final_experiment_report.md")
    
    # 1. Collect Baseline Summaries
    baselines_file = os.path.join(exp_dirs[0], "baselines_summary.json")
    results = {}
    if os.path.exists(baselines_file):
        with open(baselines_file, 'r') as f:
            results.update(json.load(f)["results"])
            
    # 2. Collect OurMethod Summaries
    ourmethod_files = glob.glob(os.path.join(exp_dirs[0], "ourmethod__*_summary.json"))
    for f_path in ourmethod_files:
        with open(f_path, 'r') as f:
            results.update({os.path.basename(f_path).replace("_summary.json", ""): json.load(f)["results"]})
            
    # 3. Collect UCB Dominance Analytics
    dominance_file = glob.glob(os.path.join(exp_dirs[0], "ourmethod_ucb_dominance_analysis_5000r.json"))
    dominance_data = None
    if dominance_file:
        with open(dominance_file[0], 'r') as f:
            dominance_data = json.load(f)

    with open(report_path, "w", encoding='utf-8') as md:
        md.write(f"# 实验总结报告 (Ablation Study Summary)\n\n")
        md.write(f"- **数据集**: {dataset}\n")
        md.write(f"- **总轮数**: {n_rounds}\n")
        md.write(f"- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        md.write("## 1. 算法性能对比 (Performance Comparison)\n\n")
        md.write("| Algorithm | Cum. Regret | Avg. Reward | Accuracy |\n")
        md.write("| :--- | :---: | :---: | :---: |\n")
        for algo, data in results.items():
            md.write(f"| {algo} | {data['cumulative_regret']:.1f} | {data['average_reward']:.4f} | {data['accuracy']:.4f} |\n")
        
        if dominance_data:
            md.write("\n## 2. OurMethod UCB 自适应分析 (Dynamic Selection Analysis)\n\n")
            md.write("展示不同阶段中各 UCB 分量的被选中比例及其对应数值：\n\n")
            md.write("| Stage | S1 (LinUCB) % | S2 (UCB1) % | S3 (LLM) % | Mean UCB1 | Mean UCB3 |\n")
            md.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
            for stage, d in dominance_data.items():
                pct = d["percentages"]
                md.write(f"| {stage} | {pct['s1']*100:.1f}% | {pct['s2']*100:.1f}% | {pct['s3']*100:.1f}% | {d['ucb_mean_s1']:.4f} | {d['ucb_mean_s3']:.4f} |\n")
        
        md.write("\n## 3. 结论 (Conclusions)\n")
        md.write("- 观察各阶段被选中源的变化，可以验证在何种情况下文本特征（S3）比传统特征（S1/S2）更具有区分度。\n")
        md.write("- 请查看该目录下的 `plot_*.png` 获取具体的遗憾与奖励增长曲线。\n")

    print(f"Report generated at: {report_path}")

if __name__ == "__main__":
    generate_report()

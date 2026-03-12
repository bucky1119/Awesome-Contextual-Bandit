import subprocess
import sys

def main():
    cmd = [
        "/home/csg/miniconda3/envs/bandits/bin/python", "run_dataset_step.py", "ourmethod",
        "--datasets", "ag_news",
        "--model", "stub",
        "--n_rounds", "50",
        "--cold_start_n", "10"
    ]
    print(f"Running smoke test: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("Smoke test failed!")
        print(result.stderr)
        sys.exit(result.returncode)
    else:
        print("Smoke test passed successfully!")

if __name__ == "__main__":
    main()
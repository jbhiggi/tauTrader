#!/usr/bin/env python3
import sys
import subprocess

# Edit this once; everything else reuses it
CONFIGS = ["config"]  # put your YAML config filenames here

def run(cmd):
    print("▶", " ".join(cmd))
    p = subprocess.run(cmd)
    if p.returncode != 0:
        sys.exit(p.returncode)

def main():
    py = sys.executable  # use current Python (safe for conda/venv)

    for cfg in CONFIGS:
        run([py, "experiments/train_model.py", cfg])
        run([py, "experiments/train_kfold.py", cfg])
        run([py, "experiments/map_tau.py", cfg])

    print("\n✅ All experiments completed.")

if __name__ == "__main__":
    main()

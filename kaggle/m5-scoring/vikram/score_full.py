"""Milestone 5: score F-03 on valid and the full test sets.

Push from the repository root with:  kaggle kernels push -p kaggle/m5-scoring/vikram --accelerator NvidiaTeslaT4
About 45 min per XLS-R model on 2 x T4. Output: /kaggle/working/scores/<run>/predictions_full.csv.gz, results_full.json
"""
import glob, subprocess, time

XLSR_RUNS = ['F-03']   # fp16 (same EER as fp32; only borderline scores move slightly)
FP32_RUNS = []   # baselines: small, scored in fp32

def run(cmd, cwd=None):
    print("$", cmd, flush=True)
    t = time.time()
    code = subprocess.run(cmd, shell=True, cwd=cwd).returncode
    print(f"exit {code} after {(time.time() - t) / 60:.1f} min", flush=True)
    if code:
        raise SystemExit(code)

run("git clone -q --depth 1 -b milestone-5 https://github.com/thehimanshhu/Group-3-DS-and-AI-Lab-Project /tmp/repo")
run("git -C /tmp/repo log --oneline -1; nvidia-smi -L")
DATA = glob.glob("/kaggle/input/datasets/*/dsai-group3-m4-data") + glob.glob("/kaggle/input/dsai-group3-m4-data")
EN = glob.glob("/kaggle/input/notebooks/*/dsai-group3-m5-full-tests-en") + glob.glob("/kaggle/input/dsai-group3-m5-full-tests-en")
print("data:", DATA, "| full English sets:", EN, flush=True)
common = f"--full --data {DATA[0]} {EN[0]} --checkpoints /kaggle/input --out /kaggle/working/scores"
if XLSR_RUNS:
    run(f"python -m src.score {' '.join(XLSR_RUNS)} {common}", cwd="/tmp/repo")
if FP32_RUNS:
    run(f"python -m src.score {' '.join(FP32_RUNS)} {common} --fp32", cwd="/tmp/repo")

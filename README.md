# Multi-Model-Surrogate-Ensemble-Integration-with-CMA-ES-for-Efficient-Black-Box-Optimization
Multi-Model Surrogate Ensemble Integration


Recommended repo structure
multi-model-surrogate-ensemble-cmaes/
├─ src/                         # your package/modules go here (add your .py from the zip)
│  └─ __init__.py
├─ notebooks/                   # put any exploratory .ipynb here
├─ results/                     # drop raw result CSV/TSV/JSON/XLSX here
├─ tools/
│  └─ summarize_results.py      # auto-aggregates comparison results (see below)
├─ README.md
├─ requirements.txt             # fill from your imports (or let me infer once I can open the zip)
├─ .gitignore
├─ CONTRIBUTING.md              # (optional)
└─ LICENSE                      # (optional – tell me which one you want)

README.md (drop this into the repo root)
# Multi-Model Surrogate Ensemble + CMA-ES for Efficient Black-Box Optimization

This repository contains code and experiments for integrating **multi-model surrogate ensembles** with **CMA-ES** to accelerate black-box optimization. The goal is to blend complementary surrogate regressors (e.g., GP, tree ensembles, kernels) to guide candidate selection while CMA-ES explores the search space efficiently.

---

## 🔧 Features

- **CMA-ES** outer-loop optimization for continuous, derivative-free problems  
- **Surrogate ensemble** (plug-and-play: GP, RF/ET, GBM, SVR, etc.) for sample efficiency  
- **Acquisition-driven** candidate evaluation (e.g., expected improvement / UCB variants)  
- **Reproducible experiments** & **batchable runs**  
- **Auto-comparison tool** to summarize results across runs into a single CSV

---

## 🗂️ Repository Layout



src/ # core library code
notebooks/ # exploratory analysis & demos
results/ # raw per-run outputs (CSV/TSV/JSON/XLSX)
tools/
summarize_results.py # aggregates results into COMPARISON_RESULTS.csv
README.md
requirements.txt
.gitignore


---

## 🚀 Quickstart

```bash
# 1) (optional) create & activate a virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# 2) install dependencies
pip install -r requirements.txt

# 3) run your experiments
# (example) python src/run_experiment.py --problem branin --budget 200 --seed 0

# 4) summarize results for comparison (see section below)
python tools/summarize_results.py --results results --out COMPARISON_RESULTS.csv


If you need help packaging your current scripts into src/, ping me and I’ll wire it up.

📊 Comparison Results

After your runs, put all raw outputs inside results/.
If your files contain columns like model/method/and score/loss/objective, you can auto-aggregate:

python tools/summarize_results.py --results results --out COMPARISON_RESULTS.csv


This produces COMPARISON_RESULTS.csv with:

model (or first detected method column)

count, mean, min, max for the primary score/loss column

Tip: Lower-is-better metrics (loss, error, RMSE) will sort ascending by default. If your metric is higher-is-better, just invert or modify the sorter in the script.

🧠 How It Works (high level)

Sampling: CMA-ES proposes candidates from an adaptive Gaussian search distribution.

Evaluation: The black-box function is queried (expensive step).

Surrogates: Fit/refresh an ensemble (e.g., GP + RF + GBM) on observed data.

Acquisition: Use the ensemble’s predictive mean/uncertainty to pick promising points.

Update: Feed back the evaluated results into CMA-ES and repeat.

This hybrid balances global exploration (CMA-ES) and local sample-efficiency (ensemble surrogates).

🧪 Reproducibility Checklist

Fix seeds for numpy/torch/sklearn where applicable.

Log each run’s parameters and seed in your result files.

Keep per-run outputs in results/<experiment>/<seed>.csv.

Use tools/summarize_results.py to generate the cross-run comparison table.

📎 Citation

If you use this repository in academic work, please cite (add your BibTeX here).


---

# tools/summarize_results.py

Place this in `tools/summarize_results.py`:

```python
import argparse
from pathlib import Path
import pandas as pd

MODEL_KEYS = ["model","method","surrogate","algo","algorithm","estimator","ensemble"]
SCORE_KEYS = ["score","rmse","mae","mse","nll","objective","fitness","value","best","error","loss"]

def infer_columns(df):
    cols = [str(c).lower() for c in df.columns]
    model_col = next((c for c in cols if any(k in c for k in MODEL_KEYS)), None)
    score_col = next((c for c in cols if any(k in c for k in SCORE_KEYS)), None)
    return model_col, score_col

def read_table(p: Path):
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower() == ".tsv":
        return pd.read_csv(p, sep="\t")
    if p.suffix.lower() == ".json":
        return pd.read_json(p)
    if p.suffix.lower() == ".xlsx":
        return pd.read_excel(p)
    return None

def main(results_dir: Path, out_path: Path):
    if not results_dir.exists():
        print(f"[warn] results directory not found: {results_dir}")
        return

    frames = []
    for f in results_dir.rglob("*"):
        if f.suffix.lower() in {".csv",".tsv",".json",".xlsx"}:
            try:
                t = read_table(f)
                if t is not None and not t.empty:
                    t["_source_file"] = f.relative_to(results_dir).as_posix()
                    frames.append(t)
            except Exception as e:
                print(f"[skip] {f}: {e}")

    if not frames:
        print("[warn] no readable tables found.")
        return

    df = pd.concat(frames, ignore_index=True)
    df.columns = [str(c).lower() for c in df.columns]

    model_col, score_col = infer_columns(df)
    if model_col is None or score_col is None:
        print("[warn] could not infer model/score columns.")
        print(f"columns: {list(df.columns)}")
        return

    summary = (df.groupby(model_col)[score_col]
                 .agg(["count","mean","min","max"])
                 .reset_index()
                 .sort_values("mean", ascending=True))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(f"[ok] wrote {out_path} ({len(summary)} rows) using '{model_col}' vs '{score_col}'")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("COMPARISON_RESULTS.csv"))
    args = ap.parse_args()
    main(args.results, args.out)

.gitignore
# Python
__pycache__/
*.py[cod]
*.so
*.dylib
.venv/
env/
venv/
build/
dist/
*.egg-info/
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db

# Logs & artifacts
*.log
*.pkl
*.pickle

# Editors/IDE
.vscode/
.idea/

# Data (keep only small CSVs you need)
data/
results/*.npz

CONTRIBUTING.md (optional)
# Contributing

1. Create a new branch from `main`.
2. Keep each experiment in its own script/notebook; write outputs to `results/experiment_name/`.
3. Ensure your run produces a tabular file with columns identifying the **model/method** and the **score/loss**.
4. Run `python tools/summarize_results.py` and commit the updated `COMPARISON_RESULTS.csv` if it changes.
5. Open a PR; include a short description and example command lines to reproduce.

requirements.txt

I can auto-infer this precisely once I can scan your code, but for now these are the likely core deps in this kind of project. Keep or trim as needed:

numpy
scipy
pandas
scikit-learn
matplotlib
tqdm
cma
pyyaml
joblib


(If you use GP frameworks or gradient boosters, add e.g. gpytorch, xgboost, lightgbm, catboost.)



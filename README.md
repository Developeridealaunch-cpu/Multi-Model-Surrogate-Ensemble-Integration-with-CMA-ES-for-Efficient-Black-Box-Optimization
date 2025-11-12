# 🌌 **Multi-Model Surrogate Ensemble + CMA-ES High-Efficiency Surrogate-Assisted Black-Box Optimization**

🚀 **Overview**

A unified research framework combining multi-model surrogate ensembles with CMA-ES (Covariance Matrix Adaptation Evolution Strategy) — designed to drastically reduce expensive evaluations in scientific, simulation, and engineering optimization.

**Core Loop:**

CMA-ES Exploration → Surrogate Prediction → Uncertainty Estimation → Acquisition Ranking →
True Evaluation (Top-K) → Surrogate Retraining

✨ **Key Highlights**

**Feature	Description**

🧩 Multi-Model Surrogates	GP, SVR, RBF, Polynomial, MC-Dropout (BNN-like)

⚙️ CMA-ES Integration	Adaptive, global, derivative-free optimizer

🔍 Uncertainty-Aware Sampling	UCB, LCB, and EI acquisition

🧠 Novel Algorithms	ESR–CMA-ES • DAE–SMC–CMA • MSES–CMA

🧰 Automated Benchmarking	Comparison, visualization, and summary tools

⚡ Efficiency	5–10× fewer expensive evaluations vs classical CMA-ES

🧑‍💻 Extensible	Plug-and-play for new surrogates, encoders, or priors

🧱 **Installation**

**Requirements**

Python ≥ 3.11

pip

Setup

pip install -r requirements.txt

(Recommended) Virtual Environment
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
.\.venv\Scripts\activate       # Windows

⚡ **Quick Start**

# ✅ Verify installation
python -c "print('CMA-ES + Surrogate Framework Ready!')"

# 🚀 Run demo optimization
python run_cmaes_surrogate_demo.py --function sphere --dim 5 --max_evals 100

# 🔬 Compare CMA-ES vs Surrogate-CMA-ES
python run_comparison.py --functions sphere,rastrigin,rosenbrock --dim 3 --runs 5 --max_evals 120 --include_variants

# 📊 Generate summary metrics
python tools/summarize_results.py --results results --out COMPARISON_RESULTS.csv

💡 **Example Usage**

🧠 **Example 1 — Surrogate-Assisted CMA-ES**

from surrogate.surrogate_ensemble import SurrogateEnsemble
from optimizer.cma_es_optimizer import CMAESOptimizer
import numpy as np

def sphere(x): return np.sum(x**2)
bounds = [(-5, 5)] * 3

model = SurrogateEnsemble(input_dim=3, n_models=5)
opt = CMAESOptimizer(dim=3, bounds=bounds, surrogate=model, max_evals=150)
res = opt.optimize(sphere, verbose=True)
print(res["best_x"], res["best_y"])

⚖️ **Example 2 — Pure CMA-ES vs Surrogate-CMA-ES**

from optimizer.baselines import pure_cmaes, surrogate_cmaes
import numpy as np

def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2*np.pi*x))

print(pure_cmaes(rastrigin, dim=5))
print(surrogate_cmaes(rastrigin, dim=5))

🧩 **Project Structure**

project-root/
│
├── surrogate/
│   ├── surrogate_ensemble.py        # Multi-model ensemble
│   └── gp_model.py                  # Gaussian Process wrapper
│
├── optimizer/
│   ├── cma_es_optimizer.py          # CMA-ES core + surrogate integration
│   ├── acquisition.py               # EI, UCB, LCB functions
│   └── baselines.py                 # Pure CMA-ES + baseline methods
│
├── benchmarks/
│   ├── sphere.py
│   ├── rastrigin.py
│   └── rosenbrock.py
│
├── tools/
│   ├── summarize_results.py
│   ├── plot_results.py
│   ├── plot_convergence.py
│   ├── novelty_compare.py
│   └── evaluate_metrics.py
│
├── data/
│   └── bbob_samples.csv             # Synthetic benchmark dataset
│
├── results/                         # Outputs (CSV + PNG)
│   ├── comparison.csv
│   ├── convergence_history.csv
│   ├── surrogate_metrics.csv
│   ├── optimization_metrics.csv
│   ├── novelty_performance.csv
│   └── *.png                        # All plots
│
├── run_cmaes_surrogate_demo.py
├── run_comparison.py
└── requirements.txt

🧮 **Algorithm Details**

🔹 **Surrogate Ensemble**

Models: GP, SVR, RBF, Polynomial, MC-Dropout (BNN-like)

Prediction fusion via weighted mean aggregation

Uncertainty = ensemble variance

Acquisition
=
𝜇
−
𝑘
𝜎
Acquisition=μ−kσ

🔹 **CMA-ES Integration**

CMA-ES generates candidate samples

Surrogate predicts & ranks via acquisition

Top-K real evaluations refine CMA-ES covariance

Surrogate retrains periodically

📈 **Results & Metrics**

🧠 **Surrogate Metrics**

**Metric Meaning**

τ	Kendall-τ Rank Correlation
RDE	Relative Distance Error
RMSE	Root Mean Square Error
Corr	Inter-model Consistency
Calibration	Reliability of uncertainty estimation

⚙️ **Optimization Metrics**

**Metric Definition**

ERT	Expected Running Time (evaluations to target)
N_eval	Evaluations to reach global optimum
Best_f(x)	Best solution quality
Success_rate	% of runs reaching target
COCO Visualization	log(FE) vs f(x) curves

🧪 **Novel Variants (New Contributions)**

<details> <summary>🌟 **ESR–CMA-ES — Ensemble Surrogate Rank CMA-ES**</summary>

Idea: Aggregates ranks across surrogates for robust candidate selection.
Benefits: Noise-resistant, scale-independent, stable across landscapes.

</details> <details> <summary>🤖 **DAE–SMC-CMA — Dual Adaptive Ensemble + Surrogate Model Control**</summary>

Idea: Two adaptive layers — surrogate reliability & CMA-ES evolution control.
Benefits: Prevents overconfidence, dynamically adjusts surrogate trust.

</details> <details> <summary>🌐 **MSES-CMA — Multi-Scale Ensemble Surrogate CMA-ES**</summary>

Idea: Multi-scale surrogates for global–local structure capture.
Benefits: Excellent balance between exploration & exploitation.

</details>

🧠 **Optional Enhancements**

Transformer-Based Embeddings — Landscape encoding for structured generalization

Meta-Learned Priors — Warm-start surrogate hyperparameters

Adaptive Switching — Surrogate trust based on uncertainty & ensemble agreement

📊 **Evaluation Outputs**

All results are auto-saved under /results/:

**File Description**

comparison.csv	Method-wise optimization performance
novelty_performance.csv	Novelty vs performance metrics
surrogate_metrics.csv	Surrogate accuracy metrics
optimization_metrics.csv	ERT, success rate, etc.
*.png	Plots: performance, convergence, metrics

🧩 **Dataset**

BBOB-style dataset for surrogate training and testing:
data/bbob_samples.csv — 500 samples each for Sphere, Rastrigin, Rosenbrock (3D).

🧰 **Troubleshooting**

Issue	Fix

ImportError	Reinstall dependencies via pip install -r requirements.txt
Slow surrogates	Reduce ensemble size or dimension
Divergent CMA-ES	Ensure finite, ordered bounds
Empty outputs	Check that /results/ contains CSVs

🤝 **Contributing**

💡 Pull Requests Welcome!

Follow consistent code style

Document new surrogates or acquisition functions

Add reproducible test cases

🧾 **License**

This repository is for research and educational use only.
Please cite CMA-ES and surrogate modeling literature in derived publications.

🧬 **Citation**

Hansen, N. (2006). The CMA Evolution Strategy: A Comparing Review.
Surrogates in Black-Box Optimization — Springer, 2021.

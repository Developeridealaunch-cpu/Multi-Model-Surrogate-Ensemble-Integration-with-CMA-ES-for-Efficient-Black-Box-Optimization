# Multi-Model-Surrogate-Ensemble-Integration-with-CMA-ES-for-Efficient-Black-Box-Optimization
Multi-Model Surrogate Ensemble Integration


🚀 Multi-Model Surrogate Ensemble + CMA-ES Framework

Project Type: Surrogate-assisted black-box optimization
Methodology: Ensemble surrogate models + CMA-ES exploration
Goal: Efficient optimization with drastically fewer expensive function evaluations

This repository implements a multi-model surrogate ensemble framework integrated with CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
The system supports efficient black-box optimization, uncertainty-driven sampling, flexible surrogate selection, and automated benchmarking.

📋 Table of Contents

Overview

Key Features

Installation

Quick Start

Usage Examples

Project Structure

Algorithm Details

Results & Metrics

Troubleshooting

Contributing

🎯 Overview

This framework integrates CMA-ES with a surrogate ensemble to reduce the number of expensive objective evaluations required for black-box optimization.

Core Principles

CMA-ES explores the search space globally.

A surrogate ensemble approximates the objective function using multiple models (e.g., GP, Random Forest, Gradient Boosting, SVR).

Uncertainty estimation identifies where the surrogate is unreliable.

Acquisition-based sampling selects new points to evaluate on the real objective.

The surrogate is iteratively updated, improving accuracy over time.

This hybrid approach increases efficiency and robustness compared to standalone CMA-ES or neural-network-based surrogate models.

✅ Key Features

✅ Multi-model surrogate ensemble (GP + RF + GBM + SVR + Custom models)
✅ CMA-ES integration for robust global optimization
✅ Uncertainty-aware sampling using ensemble variance
✅ Support for black-box functions (scientific, engineering, simulation-based)
✅ Batch evaluation mode for parallel systems
✅ Lightweight dependencies (no deep-learning frameworks)
✅ Automated comparison tools
✅ Results summarization (COMPARISON_RESULTS.csv)
✅ Extensible design for custom surrogates and acquisition strategies

🔧 Installation
Prerequisites

Python 3.8+

pip package manager

Install Dependencies
pip install -r requirements.txt

Recommended Virtual Environment
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.\.venv\Scripts\activate    # Windows

🚀 Quick Start
1. Verify the installation
python -c "print('✅ CMA-ES + Surrogate Framework Ready!')"

2. Run a demo optimization
python run_cmaes_surrogate_demo.py --function sphere --dim 5 --max_evals 100

3. Benchmark comparison (CMA-ES vs Surrogate-CMA-ES)
python run_comparison.py --functions sphere,rastrigin,rosenbrock --dim 2 --runs 5

4. Generate summary metrics
python tools/summarize_results.py --results results --out COMPARISON_RESULTS.csv

💡 Usage Examples
Example 1: Basic Surrogate-Assisted CMA-ES
from surrogate.surrogate_ensemble import SurrogateEnsemble
from optimizer.cma_es_optimizer import CMAESOptimizer
import numpy as np

def sphere(x):
    return np.sum(x**2)

bounds = [(-5, 5)] * 3

model = SurrogateEnsemble(input_dim=3, n_models=5)
optimizer = CMAESOptimizer(dim=3, bounds=bounds, surrogate=model, max_evals=150)

result = optimizer.optimize(sphere, verbose=True)
print(result["best_x"], result["best_y"])

Example 2: Pure CMA-ES vs Surrogate-CMA-ES
from optimizer.baselines import pure_cmaes, surrogate_cmaes
import numpy as np

def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2*np.pi*x))

best1 = pure_cmaes(rastrigin, dim=5)
best2 = surrogate_cmaes(rastrigin, dim=5)

print(best1, best2)

Example 3: Optimization on Benchmark Functions
python run_benchmarks.py --functions sphere,rosenbrock --dim 2,5,10 --runs 10

📁 Project Structure
project-root/
├── surrogate/                          # Surrogate models
│   ├── surrogate_ensemble.py           # Multi-model ensemble logic
│   └── gp_model.py                     # Gaussian Process wrapper
│
├── optimizer/                          # Optimization components
│   ├── cma_es_optimizer.py             # Surrogate-assisted CMA-ES core
│   ├── acquisition.py                  # UCB/LCB/EI-based strategies
│   └── baselines.py                    # Pure CMA-ES + baseline methods
│
├── benchmarks/                         # Benchmark functions
│   ├── sphere.py
│   ├── rastrigin.py
│   └── rosenbrock.py
│
├── tools/
│   └── summarize_results.py            # Comparison summarizer
│
├── results/                            # Saved results
├── run_cmaes_surrogate_demo.py
├── run_comparison.py
├── requirements.txt
└── README.md

🧠 Algorithm Details
Surrogate Ensemble Architecture

Models: configurable mixture (GP, RF, GBM, SVR)

Uncertainty: computed as variance across ensemble predictions

Training: retrained periodically with new evaluated samples

Acquisition score:

acquisition = mean_prediction - k * uncertainty     # exploit–explore balance

CMA-ES Integration

CMA-ES generates candidate points

Surrogate predicts:

Expected function values

Uncertainty

Acquisition ranks candidates

Top-K candidates are evaluated on the true function

CMA-ES updates using true evaluations

Surrogate retrains

Default Hyperparameters

Population size: dynamic, depends on dimension

Batch size: 4–16 (user-set)

Ensemble size: 5 models

Uncertainty weight: k = 1–3

Retrain frequency: 1 iteration

📊 Results & Metrics
Metrics Automatically Computed

Best value achieved

Evaluations to reach threshold

Mean/Min/Max performance across runs

Variance-based robustness

Comparison across optimizers

Auto-Generated Files
results/experiment_YYYYMMDD/
├── runs.csv
├── metrics.json
├── convergence.png
├── performance_summary.png
└── COMPARISON_RESULTS.csv   (via summarize_results.py)

🐛 Troubleshooting
Common Issues & Fixes
Issue	Fix
ImportError	Install dependencies via pip install -r requirements.txt
Slow surrogate	Reduce ensemble size or dimensionality
Divergent CMA-ES	Check bounds (must be finite and ordered)
Empty comparison outputs	Ensure results folder contains CSV/TSV/JSON
🤝 Contributing

Follow existing coding style

Keep surrogate interface consistent

Add tests for new models or acquisitions

Document new hyperparameters

Use small reproducible examples in PRs

📄 License

This project is for research and educational use.
Please cite relevant CMA-ES and surrogate modeling literature if used in publications.



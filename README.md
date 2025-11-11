🚀 **Multi-Model Surrogate Ensemble + CMA-ES Framework High-Efficiency Surrogate-Assisted Black-Box Optimization**

A unified framework combining multi-model surrogate ensembles with CMA-ES, designed to drastically reduce the number of expensive objective evaluations in scientific and engineering optimization.

📌 **Table of Contents**

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

Novel Methods Introduced

License

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🎯 **Overview**

This framework tightly integrates CMA-ES with a multi-model surrogate ensemble to perform sample-efficient optimization on expensive, noisy, or simulation-based black-box functions.

Core Workflow
CMA-ES Exploration → Surrogate Ensemble Prediction →  
Uncertainty Estimation → Acquisition Ranking →  
True Evaluation (Top-K) → Surrogate Retraining

Key Principles

CMA-ES provides global exploration and adaptive covariance shaping.

Surrogate ensemble approximates the expensive objective using multiple regressors (GP, RF, GBM, SVR, custom).

Ensemble variance yields uncertainty estimation and trust-control.

Acquisition strategies guide efficient candidate selection.

Surrogates are updated iteratively, improving accuracy and stability.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

✅ **Key Features**

Multi-model surrogate ensemble (GP + RF + GBM + SVR + custom)

CMA-ES integration for robust black-box optimization

Uncertainty-aware acquisition (UCB, LCB, EI, variance-based)

Optional batch-evaluation for parallel systems

Minimal dependencies; no deep learning required

Automatic benchmarking & comparison toolset

Results summarization with convergence metrics

Extensible design for new models & acquisition functions.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🔧 **Installation**
Requirements

Python < 3.11

pip

Install
pip install -r requirements.txt

Virtual Environment (recommended)
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
.\.venv\Scripts\activate       # Windows

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🚀 **Quick Start**
1. Verify Installation
python -c "print('✅ CMA-ES + Surrogate Framework Ready!')"

2. Run Demo Optimization
python run_cmaes_surrogate_demo.py --function sphere --dim 5 --max_evals 100

3. Compare CMA-ES vs Surrogate-CMA-ES
python run_comparison.py --functions sphere,rastrigin,rosenbrock --dim 2 --runs 5

4. Generate Summary Metrics
python tools/summarize_results.py --results results --out COMPARISON_RESULTS.csv

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

💡 **Usage Examples**
✅ Example 1 — Basic Surrogate-Assisted CMA-ES
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

✅ Example 2 — Pure CMA-ES vs Surrogate-CMA-ES
from optimizer.baselines import pure_cmaes, surrogate_cmaes
import numpy as np

def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2*np.pi*x))

best1 = pure_cmaes(rastrigin, dim=5)
best2 = surrogate_cmaes(rastrigin, dim=5)

print(best1, best2)

📁 Project Structure
project-root/
│
├── surrogate/
│   ├── surrogate_ensemble.py
│   └── gp_model.py
│
├── optimizer/
│   ├── cma_es_optimizer.py
│   ├── acquisition.py
│   └── baselines.py
│
├── benchmarks/
│   ├── sphere.py
│   ├── rastrigin.py
│   └── rosenbrock.py
│
├── tools/
│   └── summarize_results.py
│
├── results/
├── run_cmaes_surrogate_demo.py
├── run_comparison.py
└── requirements.txt

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🧠 **Algorithm Details**
Surrogate Ensemble

GP, RF, GBM, SVR, and optional custom models

Prediction fusion via weighted mean

Uncertainty = ensemble variance

Retraining at each iteration

Acquisition Score:

acquisition = mean_prediction – k * uncertainty

CMA-ES Integration

CMA-ES proposes candidate points

Surrogate ranks candidates using acquisition

Top-K candidates are evaluated on the true objective

CMA-ES updates using real evaluations

Surrogate retrains with new samples

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

📊 **Results & Metrics**

Automatically computed:

Best value

Evaluations to threshold

Mean / Min / Max performance

Variance & robustness

Multi-run comparison

Generated files (per experiment):

results/experiment_YYYYMMDD/
│── runs.csv
│── metrics.json
│── convergence.png
│── performance_summary.png
└── COMPARISON_RESULTS.csv

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🐛 **Troubleshooting**
Issue	Fix
ImportError	Reinstall via pip install -r requirements.txt
Slow surrogate	Reduce ensemble size or dimensionality
CMA-ES divergence	Verify bounds are finite and ordered
Empty outputs	Ensure results folder contains valid experiment logs

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

✅ **Three Novel Surrogate-Assisted CMA-ES Variants**

These are the new contributions introduced in this work.
All variants support optional Transformer embeddings and meta-learned priors.

1. ✅ ESR–CMA-ES — Ensemble Surrogate Rank CMA-ES
Core Idea

Selection is based on rank aggregation from multiple surrogate models instead of raw predictions.

Mechanism

Each surrogate ranks candidates

Ranks are aggregated (Borda/median rank)

CMA-ES evaluates top-ranked points

Scale-invariant and noise resistant

Strengths

High robustness

Low sensitivity to surrogate miscalibration

Strong performance on noisy/multimodal landscapes

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

2. ✅ **DAE–SMC-CMA — Dual Adaptive Ensemble + Surrogate Model Control**
Core Idea

Adaptive trust-control of the surrogate based on:

Surrogate uncertainty

Ensemble agreement

Mechanism

Two adaptive signals control surrogate influence

CMA-ES switches between exploitation/exploration

Prevents overconfidence and collapse

Strengths

Most stable

Highly sample-efficient

Adapts online to unknown landscapes

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

3. ✅ **MSES-CMA — Multi-Scale Ensemble Surrogate CMA-ES**
Core Idea

Ensemble contains models trained at multiple scales:

global (coarse)

medium-scale

local (fine)

Mechanism

Scale-aware prediction fusion

Local sensitivity used for refined sampling

Multi-resolution surrogate landscape

Strengths

Excellent global–local balance

Strong on ill-conditioned or hybrid functions

Avoids deceptive local minima

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

✅ **Optional Enhancements (All Variants)**
Transformer-based embeddings

Encode structured input patterns

Improve cross-task generalization

Meta-learned priors

Learned covariance

Learned CMA-ES mean

Learned surrogate hyperparameters

Adaptive switching

Uncertainty triggers

Ensemble agreement checks

Surrogate quality thresholds

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

✅ **Summary of Novel Variants**
Variant	Core Mechanism	Strengths	Best Use Cases
ESR–CMA-ES	Rank-based surrogate selection	Noise-resistant, stable	Multimodal/noisy landscapes
DAE–SMC-CMA	Adaptive surrogate trust-control	Most stable + efficient	Unknown/dynamic problems
MSES-CMA	Multi-scale surrogate fusion	Strong global–local balance	Ill-conditioned/hybrid functions
📄 License

Research and educational use.
Cite CMA-ES & surrogate modeling literature when used academically.
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

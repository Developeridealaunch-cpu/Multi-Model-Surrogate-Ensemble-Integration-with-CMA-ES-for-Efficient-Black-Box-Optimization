# Multi-Model Surrogate Ensemble + CMA-ES (Full Research Skeleton)

Includes:
- LHS sampler, normalization/preprocessing
- Surrogate ensemble (GP, SVR, RBF, Polynomial, MC-Dropout "BNN-like")
- ESR / DAE–SMC / MSES variants
- Baselines and benchmarks
- Novelty comparison utilities and plots
- Results CSVs and figures

## Quick Start
```bash
pip install -r requirements.txt
python run_cmaes_surrogate_demo.py --function sphere --dim 3 --max_evals 100 --variant ESR
python run_comparison.py --functions sphere,rastrigin --dim 3 --runs 3 --max_evals 120 --include_variants
python tools/plot_results.py --csv results/comparison.csv --out results/performance_summary.png
python tools/plot_convergence.py --history_csv results/convergence_history.csv --out results/convergence.png
python tools/novelty_compare.py --function sphere --dim 3 --runs 3 --max_evals 120 --out_csv results/novelty_performance.csv --out_png results/novelty_vs_performance.png
python tools/summarize_results.py --results results --out COMPARISON_RESULTS.csv
```

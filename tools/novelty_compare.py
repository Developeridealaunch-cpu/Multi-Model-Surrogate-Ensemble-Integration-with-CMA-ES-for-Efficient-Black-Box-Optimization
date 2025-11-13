"""
Novelty comparison:
- Runs ESR, DAE-SMC, MSES
- Records optimization best_y and novelty signals from ensemble (disagreement, rank_stability, diversity)
- Exports novelty_performance.csv
- Plots novelty_vs_performance.png (novelty score vs best_y)
"""
import argparse, os, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os

# Add project root to Python path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from surrogate.surrogate_ensemble import SurrogateEnsemble
from optimizer.cma_es_optimizer import CMAESOptimizer
from benchmarks.sphere import sphere
from benchmarks.rastrigin import rastrigin

def novelty_score(sig):
    # Combine signals into a single novelty score (higher should mean "more novel" guidance)
    # Use: novelty = diversity + disagreement - rank_stability
    return float(sig['diversity'] + sig['disagreement'] - sig['rank_stability'])

def run_variant(f, dim, bounds, variant, seed, max_evals=100):
    model = SurrogateEnsemble(input_dim=dim, n_models=5, random_state=seed)
    opt = CMAESOptimizer(dim=dim, bounds=bounds, surrogate=model, max_evals=max_evals, seed=seed, variant=variant)
    res = opt.optimize(f, verbose=False)
    # probe novelty signals at the last candidate cloud around best_x
    # create a small grid near best_x for measuring ensemble behavior
    rng = np.random.RandomState(seed)
    around = res['best_x'] + 0.1*rng.randn(64, dim)
    around = np.clip(around, [b[0] for b in bounds], [b[1] for b in bounds])
    sig = model.novelty_signals(around)
    score = novelty_score(sig)
    return res, sig, score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--function', type=str, default='sphere', choices=['sphere','rastrigin'])
    ap.add_argument('--dim', type=int, default=3)
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--max_evals', type=int, default=120)
    ap.add_argument('--out_csv', type=str, default='../results/novelty_performance.csv')
    ap.add_argument('--out_png', type=str, default='../results/novelty_vs_performance.png')
    args = ap.parse_args()

    f = sphere if args.function=='sphere' else rastrigin
    bounds = [(-5,5)]*args.dim
    rows = []

    for variant in ['ESR','DAE-SMC','MSES']:
        for r in range(args.runs):
            res, sig, score = run_variant(f, args.dim, bounds, variant, seed=r, max_evals=args.max_evals)
            rows.append({
                'function': args.function,
                'dim': args.dim,
                'run': r,
                'variant': variant,
                'best_y': res['best_y'],
                **sig,
                'novelty_score': score
            })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    # Plot novelty vs performance: x=novelty_score, y=best_y
    fig, ax = plt.subplots()
    for v in ['ESR','DAE-SMC','MSES']:
        sub = df[df['variant']==v]
        ax.scatter(sub['novelty_score'], sub['best_y'], label=v)
    ax.set_xlabel("Novelty score (higher = more novel guidance)")
    ax.set_ylabel("Best f(x) (lower is better)")
    ax.set_title(f"Novelty vs Performance — {args.function} (dim={args.dim})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=160)
    plt.close(fig)

    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_png}")

if __name__ == '__main__':
    main()

import argparse, os, csv
import numpy as np
from surrogate.surrogate_ensemble import SurrogateEnsemble
from optimizer.cma_es_optimizer import CMAESOptimizer
from optimizer.baselines import pure_cmaes

from benchmarks.sphere import sphere
from benchmarks.rastrigin import rastrigin
from benchmarks.rosenbrock import rosenbrock

FUNS = {"sphere": sphere, "rastrigin": rastrigin, "rosenbrock": rosenbrock}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--functions", type=str, default="sphere,rastrigin,rosenbrock")
    ap.add_argument("--dim", type=str, default="2")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max_evals", type=int, default=150)
    ap.add_argument("--include_variants", action="store_true", help="Compare ESR/DAE-SMC/MSES")
    args = ap.parse_args()

    funs = [f.strip() for f in args.functions.split(",")]
    dims = [int(x) for x in args.dim.split(",")]

    os.makedirs("results", exist_ok=True)
    out_csv = os.path.join("results", "comparison.csv")
    hist_csv = os.path.join("results", "convergence_history.csv")

    with open(out_csv, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["function","dim","run","method","best_y"])
    with open(hist_csv, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["function","dim","run","method","evals","best_y"])

    for fn in funs:
        for d in dims:
            bounds = [(-5,5)]*d
            f = FUNS[fn]
            for r in range(args.runs):
                # Pure CMA-ES
                bx, by = pure_cmaes(f, d, bounds=bounds, max_evals=args.max_evals, seed=r)
                with open(out_csv, "a", newline="") as fh:
                    csv.writer(fh).writerow([fn, d, r, "pure_cmaes", by])

                # Surrogate CMA-ES variants
                variants = ["ESR"]
                if args.include_variants:
                    variants = ["ESR","DAE-SMC","MSES"]
                for v in variants:
                    model = SurrogateEnsemble(input_dim=d, n_models=5, random_state=r)
                    opt = CMAESOptimizer(dim=d, bounds=bounds, surrogate=model, max_evals=args.max_evals, seed=r, variant=v)
                    res = opt.optimize(f, verbose=False)
                    with open(out_csv, "a", newline="") as fh:
                        csv.writer(fh).writerow([fn, d, r, f"surrogate_cmaes_{v}", res["best_y"]])
                    for ev, best in res["history"]:
                        with open(hist_csv, "a", newline="") as fh:
                            csv.writer(fh).writerow([fn, d, r, f"surrogate_cmaes_{v}", ev, best])

    print(f"Wrote {out_csv}")
    print(f"Wrote {hist_csv}")

if __name__ == "__main__":
    main()

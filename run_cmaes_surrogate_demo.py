from surrogate.surrogate_ensemble import SurrogateEnsemble
from optimizer.cma_es_optimizer import CMAESOptimizer
from benchmarks.sphere import sphere
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--function", type=str, default="sphere")
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--max_evals", type=int, default=100)
    ap.add_argument("--variant", type=str, default="ESR", choices=["ESR","DAE-SMC","MSES"])
    args = ap.parse_args()

    if args.function == "sphere":
        from benchmarks.sphere import sphere as f
    elif args.function == "rastrigin":
        from benchmarks.rastrigin import rastrigin as f
    else:
        from benchmarks.rosenbrock import rosenbrock as f

    bounds = [(-5,5)] * args.dim
    model = SurrogateEnsemble(input_dim=args.dim, n_models=5, random_state=0)
    opt = CMAESOptimizer(dim=args.dim, bounds=bounds, surrogate=model, max_evals=args.max_evals, variant=args.variant)
    res = opt.optimize(lambda x: f(x[:args.dim]), verbose=True)
    print("Best:", res["best_y"], "at", res["best_x"])

if __name__ == "__main__":
    main()

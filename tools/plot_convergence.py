import argparse, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--history_csv', type=str, default='../results/convergence_history.csv')
    ap.add_argument('--out', type=str, default='../results/convergence.png')
    args = ap.parse_args()

    if not os.path.exists(args.history_csv):
        print("No convergence history found at", args.history_csv)
        return

    df = pd.read_csv(args.history_csv)  # columns: function,dim,run,method,evals,best_y
    head = df[['function','dim','method']].drop_duplicates().head(3)
    sel_methods = head['method'].unique().tolist()
    func = head.iloc[0]['function']
    dim = int(head.iloc[0]['dim'])

    fig, ax = plt.subplots()
    for m in sel_methods:
        sub = df[(df['function']==func) & (df['dim']==dim) & (df['method']==m)]
        agg = sub.groupby('evals')['best_y'].mean().reset_index()
        ax.plot(agg['evals'], agg['best_y'], label=m)

    ax.set_title(f"Convergence (avg across runs) — {func} (dim={dim})")
    ax.set_xlabel("Function evaluations")
    ax.set_ylabel("Best f(x) so far (lower is better)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=160)
    plt.close(fig)
    print(f"Wrote {args.out}")

if __name__ == '__main__':
    main()

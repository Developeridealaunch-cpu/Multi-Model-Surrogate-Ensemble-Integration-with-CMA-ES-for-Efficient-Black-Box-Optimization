import argparse, os
import pandas as pd
import matplotlib.pyplot as plt

def plot_performance_table(csv_path: str, out_png: str):
    df = pd.read_csv(csv_path)
    g = df.groupby(['function','dim','method'])['best_y'].agg(['mean','min','max','std']).reset_index()
    k = g[['function','dim']].drop_duplicates().iloc[0]
    sub = g[(g['function']==k['function']) & (g['dim']==k['dim'])].copy()
    sub = sub.sort_values('mean')
    fig, ax = plt.subplots()
    ax.bar(sub['method'], sub['mean'])
    ax.set_title(f"Performance summary (mean best_y) — {k['function']} (dim={k['dim']})")
    ax.set_ylabel("Mean best_y (lower is better)")
    ax.set_xlabel("Method")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str, default='results/comparison.csv')
    ap.add_argument('--out', type=str, default='results/performance_summary.png')
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plot_performance_table(args.csv, args.out)
    print(f"Wrote {args.out}")

if __name__ == '__main__':
    main()

"""
Summarize results CSVs into COMPARISON_RESULTS.csv (if present).
"""
import argparse, os, glob, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=str, default="../results")
    ap.add_argument("--out", type=str, default="COMPARISON_RESULTS.csv")
    args = ap.parse_args()
    rows = []
    for fn in glob.glob(os.path.join(args.results, "*.csv")):
        try:
            df = pd.read_csv(fn)
            if "best_y" in df.columns:
                rows.append({
                    "file": os.path.basename(fn),
                    "min_best_y": df["best_y"].min(),
                    "mean_best_y": df["best_y"].mean(),
                    "runs": len(df)
                })
        except Exception:
            pass
    if len(rows)==0:
        print("No result CSVs found; nothing to summarize.")
        return
    outdf = pd.DataFrame(rows)
    outdf.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()

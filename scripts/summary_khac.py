import glob, os
import pandas as pd

files = sorted(glob.glob("results/khac/*.csv")) + ["results/paper-0.30-1_FAP.csv"]
rows = []
for p in files:
    df = pd.read_csv(p)
    if "trace_f1" not in df.columns:
        continue
    df.insert(0, "file", os.path.basename(p))
    rows.append(df)

out = pd.concat(rows, ignore_index=True)
cols = ["file", "index", "trace_f1", "event_f1", "attr_f1", "attr_AP", "time"]
out = out[cols].round(4)
out.to_csv("results/summary_khac.csv", index=False)
print(out.to_string(index=False))

seeds = out[out["file"].isin(
    ["paper-0.30-1_FAP.csv", "paper_seed1.csv", "paper_seed2.csv"])]
print("\nTrung binh va do lech chuan qua cac seed (42, 1, 2):")
print(seeds[["trace_f1", "event_f1", "attr_f1"]].agg(["mean", "std"]).round(4))
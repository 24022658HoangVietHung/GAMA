import glob, os
import pandas as pd

rows = []
for path in sorted(glob.glob("results/*_*.csv")):
    name = os.path.basename(path)
    if name in ("summary.csv", "test_paper.csv"):
        continue
    df = pd.read_csv(path)
    if "trace_f1" not in df.columns:
        continue
    tf = name.replace(".csv", "").split("_")[-1]
    df.insert(1, "TF", tf)
    rows.append(df)

out = pd.concat(rows, ignore_index=True)
cols = ["index", "TF", "trace_f1", "event_f1", "attr_f1",
        "trace_AP", "attr_AP", "time"]
out = out[cols].round(4)
out.to_csv("results/summary.csv", index=False)
print(out.to_string(index=False))
import subprocess, sys, time, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
(ROOT / "logs").mkdir(exist_ok=True)
(ROOT / "results").mkdir(exist_ok=True)

LOGS = ["paper-0.30-1", "small-0.30-1"]
TFS = ["AN", "PAV", "FAP"]
SEED = 42

for log in LOGS:
    for tf in TFS:
        res = ROOT / "results" / f"{log}_{tf}.csv"
        if res.exists():
            print("Bo qua (da co):", res.name)
            continue
        cmd = [sys.executable, "main.py", "--mode", "eval", "--TF", tf,
               "--only", log, "--n_epochs", "20", "--seed", str(SEED),
               "--res", str(res)]
        out = ROOT / "logs" / f"{log}_{tf}.txt"
        start = time.time()
        with open(out, "w", encoding="utf-8") as f:
            f.write("CMD: " + " ".join(cmd) + "\n")
            p = subprocess.run(cmd, cwd=ROOT, stdout=f,
                               stderr=subprocess.STDOUT, text=True)
            f.write(f"\nRETURNCODE: {p.returncode}\n")
            f.write(f"SECONDS: {time.time() - start:.1f}\n")
        print(f"{log} {tf}: xong ({time.time() - start:.0f}s), log: {out.name}")
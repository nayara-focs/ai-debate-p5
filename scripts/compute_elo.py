import argparse, csv, numpy as np, config
from pathlib import Path

from ai_debate_p5.stats.elo_bt import win_matrix_from_log, fit_bt

parser = argparse.ArgumentParser(description="Compute Bradley-Terry Elo ratings")
parser.add_argument("log_json", help="debate log produced by run_debate.py")
parser.add_argument("--out", default="elo.csv", help="CSV file to write")
args = parser.parse_args()

ids = [d["id"] for d in config.DEBATERS]
W   = win_matrix_from_log(args.log_json, ids)
E, COV = fit_bt(W)

print("\nRaw E_hat :", np.round(E, 6))
print("CI half-widths :", np.round(1.96 * np.sqrt(np.diag(COV)), 6))

out = Path(args.out)
with out.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "elo_mean", "elo_ci95"])
    for i, d in enumerate(ids):
        se  = np.sqrt(COV[i, i])
        writer.writerow([d, f"{E[i]:.3f}", f"±{1.96*se:.3f}"])
print(f"[ok] Elo ratings saved to {out}")
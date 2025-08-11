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
ci = 1.96 * np.sqrt(np.diag(COV))
print("CI half-widths :", np.round(ci, 6))

out = Path(args.out)
with out.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "elo_mean", "elo_ci95"])
    for i, d in enumerate(ids):
        writer.writerow([d, f"{E[i]:.3f}", f"{ci[i]:.3f}"])
print(f"[ok] Elo ratings saved to {out}")

with Path(args.out).with_name("elo_pairwise.csv").open("w", newline="") as f2:
    w = csv.writer(f2)
    w.writerow(["i","j","diff_mean","diff_ci95"])
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            diff = E[i] - E[j]
            se   = np.sqrt(COV[i,i] + COV[j,j] - 2*COV[i,j])
            w.writerow([ids[i], ids[j], f"{diff:.3f}", f"{1.96*se:.3f}"])
print("[ok] Pairwise diffs saved to elo_pairwise.csv")
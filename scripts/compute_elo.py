import argparse, csv, json, numpy as np, config
from pathlib import Path

from ai_debate_p5.stats.elo_bt import fit_bt  

def _win_matrix_from_matches(matches, ids):
    """Build W[i,j] = wins of debater ids[i] over ids[j] from match records."""
    n = len(ids)
    idx = {d_id: i for i, d_id in enumerate(ids)}
    W = np.zeros((n, n), dtype=float)
    for m in matches:
        winner_label = m.get("winner")
        sid = m.get("side_to_debater_id", {})
        if not winner_label or winner_label not in sid:
            continue
        w_id = sid[winner_label]
        # opponent is the other strategy label
        other_label = "Strategy 2" if winner_label == "Strategy 1" else "Strategy 1"
        l_id = sid.get(other_label)
        if w_id in idx and l_id in idx:
            W[idx[w_id], idx[l_id]] += 1.0
    return W

def _fit_and_write(ids, W, out_csv_path: Path):
    E, COV = fit_bt(W)
    ci = 1.96 * np.sqrt(np.diag(COV))

    out_csv_path = Path(out_csv_path)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with out_csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "elo_mean", "elo_ci95"])
        for i, d in enumerate(ids):
            w.writerow([d, f"{E[i]:.3f}", f"{ci[i]:.3f}"])
    print(f"[ok] Elo ratings saved to {out_csv_path}")

    with out_csv_path.with_name(out_csv_path.stem + "_pairwise.csv").open("w", newline="") as f2:
        w2 = csv.writer(f2)
        w2.writerow(["i","j","diff_mean","diff_ci95"])
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                diff = E[i] - E[j]
                se   = np.sqrt(COV[i,i] + COV[j,j] - 2*COV[i,j])
                w2.writerow([ids[i], ids[j], f"{diff:.3f}", f"{1.96*se:.3f}"])
    print(f"[ok] Pairwise diffs saved to {out_csv_path.with_name(out_csv_path.stem + '_pairwise.csv')}")

def _sanitize(tag: str) -> str:
    return tag.replace("+", "p").replace("/", "_")

def main():
    parser = argparse.ArgumentParser(description="Compute Bradley–Terry Elo ratings")
    parser.add_argument("log_json", help="debate log produced by run_debate.py")
    parser.add_argument("--out", default="elo.csv", help="CSV file to write (or prefix if --split-by-order)")
    parser.add_argument("--filter-order",
                        choices=["P5+FCC","FCC+P5","CONCAT_UNSPECIFIED"],
                        help="If set, compute Elo using only matches with this context_order.")
    parser.add_argument("--split-by-order", action="store_true",
                        help="If set, compute Elo separately for each context_order present in the log.")
    args = parser.parse_args()

    ids = [d["id"] for d in config.DEBATERS]

    # Default: pooled (back-compat)
    if not args.filter_order and not args.split_by_order:
        # Old path: read once and pool all matches
        data = json.loads(Path(args.log_json).read_text(encoding="utf-8"))
        W = _win_matrix_from_matches(data.get("matches", []), ids)
        _fit_and_write(ids, W, Path(args.out))
        return

    # Load once for filtering/splitting
    data = json.loads(Path(args.log_json).read_text(encoding="utf-8"))
    matches = data.get("matches", [])

    if args.filter_order:
        sub = [m for m in matches if m.get("context_order") == args.filter_order]
        W = _win_matrix_from_matches(sub, ids)
        suffix = _sanitize(args.filter_order)
        out = Path(args.out)
        if out.suffix:
            out = out.with_name(out.stem + f"_{suffix}" + out.suffix)
        else:
            out = out.with_name(out.name + f"_{suffix}.csv")
        _fit_and_write(ids, W, out)
        return

    if args.split_by_order:
        # stratify by each order found in the log
        orders = sorted({m.get("context_order", "CONCAT_UNSPECIFIED") for m in matches})
        for o in orders:
            sub = [m for m in matches if m.get("context_order","CONCAT_UNSPECIFIED")==o]
            W = _win_matrix_from_matches(sub, ids)
            suffix = _sanitize(o)
            out = Path(args.out)
            if out.suffix:
                out = out.with_name(out.stem + f"_{suffix}" + out.suffix)
            else:
                out = out.with_name(out.name + f"_{suffix}.csv")
            _fit_and_write(ids, W, out)

if __name__ == "__main__":
    main()
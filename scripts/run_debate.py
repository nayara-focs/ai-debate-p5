import argparse, builtins, json
from datetime import datetime
import config
import sys
from math import prod
from pathlib import Path
import hashlib


# package-relative imports
from ai_debate_p5 import run_all_matches
from ai_debate_p5.stats_module import (global_stats, 
                                       compute_average_tokens_per_turn,)

# -------------------------------------------------------------------
# CLI helpers (non-intrusive)
# -------------------------------------------------------------------
def _parse_args():
    ap = argparse.ArgumentParser(
        description="Run the debate tournament (or quick smoke test)."
    )
    ap.add_argument("--out", help="Output JSON path")
    ap.add_argument("--repeats", type=int,
                    help="Override config.REPEATS_PER_PAIR for this run")
    ap.add_argument("--turns", type=int,
                    help="Override config.TURNS_PER_MATCH for this run")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress per-turn console output")
    return ap.parse_args()

class _SilentPrint:
    def __call__(self, *args, **kwargs):
        pass

args = _parse_args()

# runtime overrides ---------------------------------------------------
if args.repeats is not None:
    config.REPEATS_PER_PAIR = args.repeats
if args.turns is not None:
    config.TURNS_PER_MATCH = args.turns
if args.quiet:
    builtins.print = _SilentPrint()


def load_static_context(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()

def main():
    static_context = load_static_context(config.STATIC_CONTEXT_FILE)

    # --- context fingerprint for reproducibility ---

    ctx_path = config.STATIC_CONTEXT_FILE
    # Make sure we have a string path for JSON stats:
    try:
        ctx_path_str = str(ctx_path)  # handles Path objects too
    except Exception:
        ctx_path_str = f"{ctx_path}"

    with open(ctx_path, "rb") as f:
        _ctx_bytes = f.read()

    global_stats["context_path"]   = ctx_path_str
    global_stats["context_bytes"]  = len(_ctx_bytes)
    global_stats["context_sha256"] = hashlib.sha256(_ctx_bytes).hexdigest()
# -----------------------------------------------------------

    total_expected = (
        len(config.DEBATERS) * (len(config.DEBATERS) - 1) * 2
        * config.REPEATS_PER_PAIR
    )
    completed = 0          # progress counter
    dot_wrap_turn  = 60         # wrap line after N dots

    if args.quiet:
        def _bump_match():
            nonlocal completed
            completed += 1

        def _dot_turn():
            _dot_turn.count = getattr(_dot_turn, "count", 0) + 1
            sys.stdout.write("." if (_dot_turn.count % dot_wrap_turn) else ".\n")
            sys.stdout.flush()
    else:
        _bump_match = None
        _dot_turn   = None
   

    print(f"\n [info] This configuration will run {total_expected} matches.\n")

    matches_data = run_all_matches(
        static_context,
        config.INITIAL_TOPIC,
        progress_cb=_bump_match,            # counts matches (no printing)
        progress_turn_cb=_dot_turn,         # prints one dot per turn
        quiet=args.quiet
    )    
    # Compute average tokens per turn and update global stats
    avg_tokens = compute_average_tokens_per_turn()
    global_stats["average_tokens_per_turn"] = avg_tokens
    wins_by_label = global_stats.get("wins_by_label", {})

    # Print global 
    
    
    print("\n📊 Global Statistics:")
    print(f"Total Matches: {global_stats['total_matches']}")
    for label, count in wins_by_label.items():
        print(f"  {label}: {count}")
    print(f"Total Token Usage: {global_stats['total_token_usage']}")
    print(f"Total Turns: {global_stats['total_turns']}")
    print(f"Average Tokens per Turn: {avg_tokens:.2f}")

  
    output_filename = (args.out
        or f"debate_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    output_data    = {
        "matches": matches_data,
        "global_stats": global_stats
    }

# ------------- write main log -------------------------------------
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

# ------------- write stats-only file ------------------------------
    out_path   = Path(output_filename)
    stats_path = out_path.with_name(out_path.stem + "_stats.json")

    with open(stats_path, "w", encoding="utf-8") as f_stats:
        json.dump(global_stats, f_stats, ensure_ascii=False, indent=2)

# ------------- final console lines --------------------------------
    if args.quiet:
        sys.stdout.write(f"\n{completed}/{total_expected} matches done\n")
        sys.stdout.flush()

    print(f"\n🎉 All debates completed! Logs and statistics saved to {output_filename}")
    print(f"Log   → {out_path}")
    print(f"Stats → {stats_path}")



if __name__ == "__main__":
    main()

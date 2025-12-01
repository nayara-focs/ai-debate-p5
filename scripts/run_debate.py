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
    ap.add_argument("--context-order",
                        choices=["random", "p5_first", "fcc_first", "alternate"],
                        default="p5_first",  # default keeps current behaviour
                        help="How to concatenate P5 and FCC context per match."
    )
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed used when --context-order=random."
    )
    ap.add_argument("--ctx-p5", type=str, default=None,
                    help="Optional path to P5 context (overrides --ctx if given)."
    )
    ap.add_argument("--ctx-fcc", type=str, default=None,
                    help="Optional path to FCC context (overrides --ctx if given)."
    )                
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
    # --- context selection (backward compatible) ---
    if args.ctx_p5 and args.ctx_fcc:
        # Two-file mode: load both and concatenate once (load-time only).
        p5_text  = load_static_context(args.ctx_p5)
        fcc_text = load_static_context(args.ctx_fcc)

        if args.context_order == "fcc_first":
            static_context = fcc_text + "\n\n" + p5_text
            _order_mode = "fcc_first"
        else:
            # default keeps historical behaviour (P5 then FCC)
            static_context = p5_text + "\n\n" + fcc_text
            _order_mode = "p5_first"

        _ctx_source = {"p5": str(args.ctx_p5), "fcc": str(args.ctx_fcc)}
        # Fingerprint the exact combined bytes we pass into run_all_matches
        _combined_bytes = static_context.encode("utf-8")

        # Back-compat + richer metadata
        global_stats["context_path"]       = f"{args.ctx_p5} + {args.ctx_fcc} ({_order_mode})"
        global_stats["context_paths"]      = _ctx_source
        global_stats["context_order_mode"] = _order_mode
        global_stats["context_bytes"]      = len(_combined_bytes)
        global_stats["context_sha256"]     = hashlib.sha256(_combined_bytes).hexdigest()

    else:
        # Single-file (legacy) mode: unchanged behaviour
        static_context = load_static_context(config.STATIC_CONTEXT_FILE)
        ctx_path = config.STATIC_CONTEXT_FILE
        try:
            ctx_path_str = str(ctx_path)  # handles Path objects too
        except Exception:
            ctx_path_str = f"{ctx_path}"

        with open(ctx_path, "rb") as f:
            _ctx_bytes = f.read()

        # Human-readable summary, without implying a fixed order
        global_stats["context_path"]       = f"{args.ctx_p5} + {args.ctx_fcc}"
        # Machine-readable sources
        global_stats["context_paths"]      = _ctx_source
        # Reflect the actual run mode you requested (random / p5_first / fcc_first / alternate)
        global_stats["context_order_mode"] = args.context_order
        global_stats["context_order_seed"] = args.seed
        # Fingerprint the exact combined bytes for reproducibility (order here is load-time only)
        global_stats["context_bytes"]      = len(_combined_bytes)
        global_stats["context_sha256"]     = hashlib.sha256(_combined_bytes).hexdigest()

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
        quiet=args.quiet,
        context_order=args.context_order,
        seed=args.seed,
        ctx_p5_text=(p5_text if (args.ctx_p5 and args.ctx_fcc) else None),
        ctx_fcc_text=(fcc_text if (args.ctx_p5 and args.ctx_fcc) else None),
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

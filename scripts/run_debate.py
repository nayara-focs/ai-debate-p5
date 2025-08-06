import argparse, builtins, json
from datetime import datetime
import config
from math import prod

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
    static_context = load_static_context(config.P5_REPORT_FILE)
    total_expected = len(config.DEBATERS) * (len(config.DEBATERS)-1) * 2 * config.REPEATS_PER_PAIR
    print(f"\n [info] This configuration will run {total_expected} matches.\n")

    matches_data = run_all_matches(static_context, config.INITIAL_TOPIC)    
    # Compute average tokens per turn and update global stats
    avg_tokens = compute_average_tokens_per_turn()
    global_stats["average_tokens_per_turn"] = avg_tokens

    # Print global 
    
    
    print("\n📊 Global Statistics:")
    print(f"Total Matches: {global_stats['total_matches']}")
    print(f"Pro-P5 Wins: {global_stats['pro_wins']}")
    print(f"Against-P5 Wins: {global_stats['against_wins']}")
    print(f"Total Token Usage: {global_stats['total_token_usage']}")
    print(f"Total Turns: {global_stats['total_turns']}")
    print(f"Average Tokens per Turn: {avg_tokens:.2f}")

    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # output_filename = f"debate_logs_{timestamp}.json"
    output_filename = (args.out
        or f"debate_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    output_data = {
        "matches": matches_data,
        "global_stats": global_stats
    }
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 All debates completed! Logs and statistics saved to {output_filename}")

if __name__ == "__main__":
    main()

import json
from datetime import datetime
import config
from debate_engine import run_all_matches
from stats_module import global_stats, compute_average_tokens_per_turn

def load_static_context(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()

def main():
    static_context = load_static_context(config.P5_REPORT_FILE)
    matches_data = run_all_matches(config.NUM_MATCHES, static_context, config.INITIAL_TOPIC)
    
    # Compute average tokens per turn and update global stats
    avg_tokens = compute_average_tokens_per_turn()
    global_stats["average_tokens_per_turn"] = avg_tokens

    # Print global stats
    print("\n📊 Global Statistics:")
    print(f"Total Matches: {global_stats['total_matches']}")
    print(f"Pro-P5 Wins: {global_stats['pro_wins']}")
    print(f"Against-P5 Wins: {global_stats['against_wins']}")
    print(f"Total Token Usage: {global_stats['total_token_usage']}")
    print(f"Total Turns: {global_stats['total_turns']}")
    print(f"Average Tokens per Turn: {avg_tokens:.2f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"debate_logs_{timestamp}.json"
    output_data = {
        "matches": matches_data,
        "global_stats": global_stats
    }
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 All debates completed! Logs and statistics saved to {output_filename}")

if __name__ == "__main__":
    main()

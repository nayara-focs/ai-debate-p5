global_stats = {
    "total_matches": 0,
    "pro_wins": 0,
    "against_wins": 0,
    "total_token_usage": 0,   # Sum of completion tokens across all turns
    "total_turns": 0
}

def update_turn_stats(tokens_used):
    global_stats["total_token_usage"] += tokens_used
    global_stats["total_turns"] += 1

def update_match_stats(winner):
    global_stats["total_matches"] += 1
    if winner.lower().find("pro-p5 wins") != -1:
        global_stats["pro_wins"] += 1
    elif winner.lower().find("against-p5 wins") != -1:
        global_stats["against_wins"] += 1

def compute_average_tokens_per_turn():
    if global_stats["total_turns"] > 0:
        return global_stats["total_token_usage"] / global_stats["total_turns"]
    return 0

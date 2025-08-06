global_stats = {
    "total_matches": 0,
    "pro_wins": 0,
    "against_wins": 0,
    "total_turns": 0,
    "total_prompt_tokens": 0,        
    "total_completion_tokens": 0,    
    "total_token_usage": 0,          
}

def update_turn_stats(prompt_tokens: int, completion_tokens: int) -> None:
    """Accumulate prompt + completion counts for one turn."""
    global_stats["total_turns"] += 1
    global_stats["total_prompt_tokens"]     += prompt_tokens
    global_stats["total_completion_tokens"] += completion_tokens
    global_stats["total_token_usage"]= (global_stats["total_prompt_tokens"] + 
                                        global_stats["total_completion_tokens"])

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


from typing import Optional
import re

# Strict parser for a single-line structured verdict
_WINNER_LINE = re.compile(r'^\s*WINNER:\s*(.+?)\s*$', re.IGNORECASE | re.MULTILINE)

# Aggregate counters 
global_stats = {
    "total_matches": 0,
    "total_turns": 0,
    "total_prompt_tokens": 0,
    "total_completion_tokens": 0,
    "total_token_usage": 0,
    "total_judge_calls": 0, 
    "wins_by_label": {},
}

def update_judge_stats(prompt_tokens: int, completion_tokens: int) -> None:
    """Accumulate tokens for a judge call without bumping total_turns."""
    global_stats["total_judge_calls"] += 1
    global_stats["total_prompt_tokens"]     += prompt_tokens
    global_stats["total_completion_tokens"] += completion_tokens
    global_stats["total_token_usage"] = (
        global_stats["total_prompt_tokens"] + global_stats["total_completion_tokens"]
    )
    
def _extract_winner_from_text(verdict_text: Optional[str]) -> Optional[str]:
    """Return the label from a strict 'WINNER: <LABEL>' line; None if absent."""
    if not verdict_text:
        return None
    m = _WINNER_LINE.search(verdict_text)
    return m.group(1).strip() if m else None

def update_turn_stats(prompt_tokens: int, completion_tokens: int) -> None:
    """Accumulate prompt + completion counts for one turn."""
    global_stats["total_turns"] += 1
    global_stats["total_prompt_tokens"] += prompt_tokens
    global_stats["total_completion_tokens"] += completion_tokens
    global_stats["total_token_usage"] = (
        global_stats["total_prompt_tokens"] + global_stats["total_completion_tokens"]
    )

def update_match_stats(winner_label=None, verdict_text: Optional[str] = None) -> None:
    """
    Backward-compatible:
      - Old usage: update_match_stats(verdict_text_with_winner_line)
      - New usage: update_match_stats(winner_label=<label>, verdict_text=<full text>)
    Only increments counters when a clear winner is available.
    """
    # If called the old way with a single prose string, treat it as verdict_text
    if verdict_text is None and isinstance(winner_label, str) and (
        "WINNER:" in winner_label or " wins" in winner_label.lower()
    ):
        verdict_text = winner_label
        winner_label = None

    # Prefer the structured label; else parse strict 'WINNER: <LABEL>' line
    winner = winner_label or _extract_winner_from_text(verdict_text)
    if not winner:
        return  # no clear winner → do not mutate match counters

    # Count by label (neutral to naming)
    wb = global_stats["wins_by_label"]
    wb[winner] = wb.get(winner, 0) + 1

    global_stats["total_matches"] += 1

def compute_average_tokens_per_turn() -> float:
    turns = global_stats["total_turns"]
    return (global_stats["total_token_usage"] / turns) if turns > 0 else 0.0
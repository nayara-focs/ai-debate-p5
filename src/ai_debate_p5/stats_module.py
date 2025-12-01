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
    # Labels levels (e.g., "Strategy 1", "Strategy 2")
    "wins_by_label": {},
    # stance-level  (decoupled from labels)
    "wins_by_stance": {"P5": 0, "FCC": 0},
    # how often each per-match mapping is used (JSON-friendly string key)
    # e.g., "Strategy 1->P5 | Strategy 2->FCC": 30
    "stance_assignment_counts": {},
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

def update_match_stats(
    winner_label: Optional[str] = None,
    verdict_text: Optional[str] = None,
    stance_assignment: Optional[dict] = None,
) -> None:
    """
    Update per-match aggregates.

    Backward-compatible usage:
      - Old style: update_match_stats("<verdict text containing WINNER: ...>")
      - New style: update_match_stats(winner_label=<label>, verdict_text=<full text>, stance_assignment={<label>:<stance>,...})

    Only increments counters when a clear winner is available.
    """
    # If called the old way with a single prose string, treat it as verdict_text
    if verdict_text is None and isinstance(winner_label, str) and (
        "WINNER:" in winner_label or " wins" in winner_label.lower()
    ):
        verdict_text = winner_label
        winner_label = None

    # Prefer structured label; else parse strict 'WINNER: <LABEL>' line
    winner = winner_label or _extract_winner_from_text(verdict_text)
    if not winner:
        return  # no clear winner → do not mutate match counters

    # Count by label (neutral to naming)
    wb = global_stats["wins_by_label"]
    wb[winner] = wb.get(winner, 0) + 1

    # Mapping usage + stance-level tally (if mapping provided)
    if stance_assignment:
        # Count mapping usage with a deterministic, JSON-friendly key
        key = " | ".join(f"{k}->{v}" for k, v in sorted(stance_assignment.items()))
        sac = global_stats["stance_assignment_counts"]
        sac[key] = sac.get(key, 0) + 1

        # Attribute winner to stance (decoupled from label/UI)
        stance = stance_assignment.get(winner)
        if stance in ("P5", "FCC"):
            global_stats["wins_by_stance"][stance] += 1

    global_stats["total_matches"] += 1

def compute_average_tokens_per_turn() -> float:
    turns = global_stats["total_turns"]
    return (global_stats["total_token_usage"] / turns) if turns > 0 else 0.0
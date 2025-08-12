import re
import config
from config import client
from ai_debate_p5.stats_module import update_turn_stats, update_judge_stats


WINNER_RE = re.compile(r'^\s*WINNER:\s*(.+?)\s*$', re.IGNORECASE | re.MULTILINE)

def _extract_winner(text: str, allowed_labels):
    m = WINNER_RE.search(text)
    if m:
        cand = m.group(1).strip()
        if cand in allowed_labels:
            return cand
    # legacy fallback: "<LABEL> wins"
    lower = text.lower()
    for lab in allowed_labels:
        if f"{lab.lower()} wins" in lower:
            return lab
    return None

def judge_debate(match_data):
    """
    Evaluate the debate transcript and decide which debater was more persuasive.
    The function builds the debate transcript, constructs a detailed prompt, and then queries the OpenAI API
    to obtain the judge's verdict. It also prints and stores the token usage information.
    """
    # Build the transcript string from match turns.
    transcript_lines = []
    for turn in match_data["turns"]:
        transcript_lines.append(f"Turn {turn['turn_number']} - {turn['speaker']}: {turn['content']}")
    transcript = "\n".join(transcript_lines)
    

    # === Derive the two side labels directly from the first two turns ===
    side1 = match_data["turns"][0]["speaker"]
    side2 = match_data["turns"][1]["speaker"] if len(match_data["turns"]) > 1 else ("Strategy 2" if side1 != "Strategy 2" else "Strategy 1")
    allowed = (side1, side2)

    judge_prompt = (
        "You are an impartial judge. Evaluate the debate strictly based on the transcript below (no external knowledge). "
        "For each side, briefly note one key strength and one key area for improvement. Then decide which side was more persuasive.\n\n"
        f"At the very end, write exactly one line: WINNER: {side1} or WINNER: {side2}. "
        "Do not add any text after that line.\n\n"
        "Debate Transcript:\n" + transcript + "\n\n"
        f"Keep your response concise and end at a natural boundary within {config.MAX_TOKENS_PER_RESPONSE} tokens."
    )


    # ---------- first attempt -------------------------------------------------
    judge_response = client.chat.completions.create(
    model=config.MODEL,
    messages=[
        {"role": "system", "content": "You are an impartial judge evaluating a debate."},
        {"role": "user",   "content": judge_prompt},
    ],
    temperature=config.TEMPERATURE,
    max_tokens=400,
)
    verdict_text = judge_response.choices[0].message.content.strip()
    update_judge_stats(judge_response.usage.prompt_tokens,
                        judge_response.usage.completion_tokens)


    winner = _extract_winner(verdict_text, allowed)
    full_verdict  = verdict_text        

# ---------- fallback reprompt --------------------------------------------
    if winner is None:
        reprompt = client.chat.completions.create(
            model=config.MODEL,
            messages=[{
                "role": "system",
                "content": f"Reply with exactly one line: WINNER: {side1} or WINNER: {side2}"
            }],
            temperature=0,
            max_tokens=10,
        )
        update_judge_stats(reprompt.usage.prompt_tokens, reprompt.usage.completion_tokens)
        short_line = reprompt.choices[0].message.content.strip()
        full_verdict += "\n\n--- reprompt ---\n" + short_line
        winner = _extract_winner(short_line, allowed)

    verdict = full_verdict
    judge_usage = judge_response.usage
    
    print("\n📢 Judge's Evaluation:")
    print(verdict)
    print(f"[Judge Token usage: Prompt tokens: {judge_usage.prompt_tokens}, "
          f"Completion tokens: {judge_usage.completion_tokens}, Total tokens: {judge_usage.total_tokens}]")
    
    match_data["judge_evaluation"] = {
        "verdict": verdict,
        "winner": winner,
        "token_usage": {
            "prompt_tokens": judge_usage.prompt_tokens,
            "completion_tokens": judge_usage.completion_tokens,
            "total_tokens": judge_usage.total_tokens,
        }
    }
    return verdict

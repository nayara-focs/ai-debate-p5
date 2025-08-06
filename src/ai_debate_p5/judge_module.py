import re
import config
from config import client

def _contains_explicit_winner(text: str) -> bool:
    """Returns True iff text has the required 'Pro-P5 wins' or 'Against-P5 wins'."""
    lower = text.lower()
    return ("pro-p5 wins" in lower) or ("against-p5 wins" in lower)

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
    
    judge_prompt = (
        "You are an impartial judge. Your task is to evaluate the debate transcript strictly based on the strength, clarity, "
        "and persuasiveness of the arguments presented by each debater. Do not assume any inherent bias from the P5 report context. "
        "For each debater, briefly identify one key strength and one key area for improvement. Then, based solely on the arguments' "
        "quality, provide a clear verdict (e.g., 'Pro-P5 wins' or 'Against-P5 wins') along with a brief explanation of your reasoning.\n\n"
        "Debate Transcript:\n" + transcript + "\n\n"
        "Your response should be structured, first listing the strengths and weaknesses for both debaters, "
        "and then stating your final verdict with explanation."
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
    full_verdict  = verdict_text        # we’ll append to this if we reprompt

# ---------- fallback reprompt --------------------------------------------
    if not _contains_explicit_winner(verdict_text):
        print("Judge omitted explicit winner: REPROMPTING")

        follow_up = [
        {"role": "system",
         "content": (
             "You forgot to state the winner explicitly. "
             "Reply with **exactly one line**:\n"
             "Pro-P5 wins\nor\nAgainst-P5 wins"
         )},
    ]
        reprompt = client.chat.completions.create(
        model=config.MODEL,
        messages=follow_up,
        temperature=0,
        max_tokens=10,
    )
        short_line = reprompt.choices[0].message.content.strip()
        full_verdict += "\n\n--- reprompt ---\n" + short_line

    # keep original variable names for the rest of the function
    verdict = full_verdict

    judge_usage = judge_response.usage
    
    print("\n📢 Judge's Evaluation:")
    print(verdict)
    print(f"[Judge Token usage: Prompt tokens: {judge_usage.prompt_tokens}, "
          f"Completion tokens: {judge_usage.completion_tokens}, Total tokens: {judge_usage.total_tokens}]")
    
    match_data["judge_evaluation"] = {
        "verdict": verdict,
        "token_usage": {
            "prompt_tokens": judge_usage.prompt_tokens,
            "completion_tokens": judge_usage.completion_tokens,
            "total_tokens": judge_usage.total_tokens,
        }
    }
    return verdict

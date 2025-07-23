# debate_engine.py
import time
import re
import json
import openai
from datetime import datetime
import config
from config import client  # Import the client from config

from .judge_module import judge_debate
from .stats_module import global_stats, update_turn_stats, update_match_stats

def generate_openings(side: str, n_variants: int, static_context, initial_topic):
    """
    Generate multiple opening arguments for the given side.
    Returns a list of dictionaries with keys: 'text' and 'usage'.
    """
    prompt = (
        f"You are a debater taking the {side} side in a debate.\n\n"
        f"Debate topic: {initial_topic}\n\n"
        f"Context:\n{static_context}\n\n"
        "Please write your opening argument."
    )
    openings = []
    for _ in range(n_variants):
        response = client.chat.completions.create(
            model=config.MODEL,
            messages=[
                {"role": "system", "content": "Respond clearly and concisely. Provide an opening argument for the debate."},
                {"role": "user", "content": prompt}
            ],
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS_PER_RESPONSE,
        )
        opening_text = response.choices[0].message.content.strip()
        usage = response.usage
        openings.append({"text": opening_text, "usage": usage})
        time.sleep(1)  # Short delay for pacing
    return openings

def run_debate_match(match_id, static_context, initial_topic):
    """
    Runs one complete debate match.
    Returns the match data dictionary.
    """
    match_data = {
        "match_id": match_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "turns": []
    }
    


    # Determine starting speakers: odd match -> Pro-P5 starts; even match -> Against-P5 starts.
    if match_id % 2 != 0:
        speakers = [("Pro-P5", "🔵"), ("Against-P5", "🔴")]
        pro_count = (match_id + 1) // 2  # For example: match 1 -> 1, match 3 -> 2, etc.
        variant_index = (pro_count - 1) % config.OPENING_VARIANTS
        starting_side = "Pro"
    else:
        speakers = [("Against-P5", "🔴"), ("Pro-P5", "🔵")]
        against_count = match_id // 2  # For example: match 2 -> 1, match 4 -> 2, etc.
        variant_index = (against_count - 1) % config.OPENING_VARIANTS
        starting_side = "Against"

    messages = [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user", "content": f"Debate topic: {initial_topic}\n\nContext:\n{static_context}"}
    ]

    print(f"\nGenerating {config.OPENING_VARIANTS} opening variants for {starting_side} side...")
    opening_options = generate_openings(starting_side, config.OPENING_VARIANTS, static_context, initial_topic)
    selected_variant = opening_options[variant_index]
    selected_opening = selected_variant["text"]
    usage_info = selected_variant["usage"]

    # Print round header then the opening (Turn 1)
    print("\n🔁 Starting Round 1")
    starting_speaker, starting_emoji = speakers[0]
    print(f"\n{starting_emoji} [Opening Token usage: Prompt tokens: {usage_info.prompt_tokens}, Completion tokens: {usage_info.completion_tokens}, Total tokens: {usage_info.total_tokens}]")
    print(f"\n{starting_emoji} {starting_speaker}'s Opening Argument (Turn 1):")
    print(selected_opening)

    messages.append({"role": "assistant", "content": selected_opening})
    match_data["turns"].append({
        "turn_number": 1,
        "speaker": starting_speaker,
        "tokens_used_completion": usage_info.completion_tokens,
        "content": selected_opening
    })
    update_turn_stats(usage_info.completion_tokens)

    # Continue debate for subsequent turns

    for turn in range(2, config.TURNS_PER_MATCH + 1):
        # Print new round header only when entering a new round
        if turn > 2 and (turn % 2 == 1):
            round_number = (turn + 1) // 2
            print(f"\n🔁 Starting Round {round_number}")
        
        current_speaker, emoji = speakers[(turn - 1) % 2]
        print(f"\n{emoji} {current_speaker}'s Turn {turn}")

        messages.append({"role": "user", "content": f"{current_speaker}, please respond to your opponent."})
        response = client.chat.completions.create(
            model=config.MODEL,
            messages=messages,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS_PER_RESPONSE,
        )
        content = response.choices[0].message.content
        usage = response.usage
        cleaned_content = re.sub(r'【.*?†source】', '', content).strip()
        print(f"{emoji} [Prompt tokens: {usage.prompt_tokens}, Completion tokens: {usage.completion_tokens}, Total tokens: {usage.total_tokens}]")
        print(cleaned_content)

        messages.append({"role": "assistant", "content": cleaned_content})
        match_data["turns"].append({
            "turn_number": turn,
            "speaker": current_speaker,
            "tokens_used_completion": usage.completion_tokens,
            "content": cleaned_content
        })
        update_turn_stats(usage.completion_tokens)

        if turn < config.TURNS_PER_MATCH:
            next_speaker, _ = speakers[(turn) % 2]
            messages.append({"role": "user", "content": f"{next_speaker}, please respond to your opponent."})
        
        time.sleep(1)  # Pacing delay

    # Invoke the judge after the debate match is complete
    verdict = judge_debate(match_data, static_context)
    update_match_stats(verdict)
    return match_data

def run_all_matches(num_matches, static_context, initial_topic):
    matches_data = []
    for match_id in range(1, num_matches + 1):
        print(f"\n===========================")
        print(f"🔁 Starting Debate Match {match_id}")
        print(f"===========================")
        match_data = run_debate_match(match_id, static_context, initial_topic)
        matches_data.append(match_data)
        print(f"\n✅ Debate Match {match_id} complete.")
    return matches_data

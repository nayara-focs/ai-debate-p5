# debate_engine.py
import time
import re
import json
import openai
from datetime import datetime
import config
from config import client  # Import the client from 
from .utils_openai import chat_extra_kwargs, supports_logprobs
from itertools import product


from .judge_module import judge_debate
from .stats_module import global_stats, update_turn_stats, update_match_stats

def generate_openings(
        side: str,
        boN: int,
        temperature: float,
        model_name: str,
        static_context,
        initial_topic):
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
    best_draft = None
    best_score = -float("inf")
    best_usage = None

    for _ in range(boN):
        # --- call the chosen model ---
        want_logprobs = supports_logprobs(model_name)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Respond clearly and concisely. "
                        "Provide an opening argument for the debate."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            **chat_extra_kwargs(model_name, temperature),
            logprobs = want_logprobs,
        )
        draft = response.choices[0].message.content.strip()

        # --- compute simple log-prob score (handles both client shapes) ---


        if want_logprobs:
            lp_obj = response.choices[0].logprobs
            token_logps = lp_obj.token_logprobs if hasattr(lp_obj, "token_logprobs") \
                  else [tok.logprob for tok in lp_obj.content]
            score = sum(token_logps)
        else:
            score = 0.0      # fallback when logprobs unavailable


        if score > best_score:
            best_score, best_draft, best_usage = score, draft, response.usage

        time.sleep(0.5)  # small pacing delay

    # return only the best draft
    return {"text": best_draft, "usage": best_usage}

def run_debate_match(match_id,
                     debater_pro: dict,
                     debater_con: dict,
                     static_context,
                     initial_topic,
                     pro_starts: bool):
    """
    Runs one complete debate match.
    Returns the match data dictionary.
    """
    match_data = {
        "match_id": match_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "turns": []
    }
    

    speakers = [("Pro-P5", "🔵"), ("Against-P5", "🔴")] if pro_starts \
         else [("Against-P5", "🔴"), ("Pro-P5", "🔵")]

    debater_map = {
    "Pro-P5":     debater_pro,
    "Against-P5": debater_con,
    }
    starting_speaker, starting_emoji = speakers[0]
    side_label = starting_speaker.split("-")[0]   # "Pro" or "Against"


    messages = [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user", "content": f"Debate topic: {initial_topic}\n\nContext:\n{static_context}"}
    ]

    print(f"\nGenerating opening variants for {side_label} side...")

    d = debater_map[starting_speaker]
    result = generate_openings(
            side=starting_speaker.split("-")[0],   # "Pro" or "Against"
            boN=d["boN"],
            temperature=d["temperature"],
            model_name=d["model"],
            static_context=static_context,
            initial_topic=initial_topic,
        )
    selected_opening = result["text"]
    usage_info       = result["usage"]

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

        # Get the debater's model and temperature
        d = debater_map[current_speaker]
        model_name  = d["model"]
        temperature = d["temperature"]
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            **chat_extra_kwargs(model_name, temperature),
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

def run_all_matches(static_context, initial_topic):
    matches_data = []
    debs = config.DEBATERS
    match_id = 1

    for deb_pro, deb_con in product(debs, debs):
        if deb_pro["id"] == deb_con["id"]:
            continue  # skip self-play

        for rep in range(1, config.REPEATS_PER_PAIR + 1):
            # -------- direction 1: Pro side opens ------------
            print("\n===========================")
            print(f"🔁 Starting Debate Match {match_id} "
                  f"[{deb_pro['id']}-Pro  vs  {deb_con['id']}-Con] "
                  f"(repeat {rep}/{config.REPEATS_PER_PAIR})")
            print("===========================")

            m = run_debate_match(match_id, deb_pro, deb_con,
                                 static_context, initial_topic,
                                 pro_starts=True)
            matches_data.append(m)
            print(f"\n✅ Debate Match {match_id} complete.")
            match_id += 1

            # -------- direction 2: Con side opens ------------
            print("\n===========================")
            print(f"🔁 Starting Debate Match {match_id} "
                  f"[{deb_con['id']}-Pro  vs  {deb_pro['id']}-Con] "
                  f"(repeat {rep}/{config.REPEATS_PER_PAIR})")
            print("===========================")

            m = run_debate_match(match_id, deb_con, deb_pro,
                                 static_context, initial_topic,
                                 pro_starts=False)
            matches_data.append(m)
            print(f"\n✅ Debate Match {match_id} complete.")
            match_id += 1

    return matches_data


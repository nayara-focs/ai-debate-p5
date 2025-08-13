import time
import re
import json
import openai
from datetime import datetime
import config
from config import client, SIDE_A_LABEL, SIDE_B_LABEL,SIDE_STANCE 
from .utils_openai import chat_extra_kwargs, supports_logprobs
from itertools import product


from .judge_module import judge_debate
from .stats_module import global_stats, update_turn_stats, update_match_stats

_END_PUNCT = re.compile(r'[.!?]["”\']?\s*$')

def _trim_to_sentence_boundary(text: str, tail: int = 240) -> str:
    """
    If the output likely ends mid-sentence, trim back to the last .!? within ~tail chars.
    Keeps text unchanged if it already ends at a clean boundary.
    """
    t = (text or "").strip()
    if _END_PUNCT.search(t):
        return t
    cut = max(t.rfind('.'), t.rfind('!'), t.rfind('?'))
    # only trim if the boundary is near the end; else keep as is
    return (t[:cut+1].strip() if cut != -1 and (len(t) - cut) <= tail else t)

def generate_openings(
        side: str,
        boN: int,
        temperature: float,
        model_name: str,
        static_context,
        initial_topic):
    """
    Generate *boN* candidate opening arguments for the given side, then
    return the single draft with the highest summed log-probability.

    This version batches the request: we make ONE ChatCompletion call
    with n=boN completions, so the large static_context is transmitted
    only once.  The return value is a dict with keys:
        • 'text'  - the chosen opening argument (str)
        • 'usage' - the OpenAI usage object for cost tracking
    """
    stance_text = SIDE_STANCE.get(side, "")
    prompt = (
    f"You are a debater advocating for {side}. {stance_text}\n\n"
    f"Debate topic: {initial_topic}\n\n"
    f"Context:\n{static_context}\n\n"
    "Please write your opening argument. Use only the provided context."
)

    want_logprobs = supports_logprobs(model_name)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": config.SYSTEM_PROMPT,
            },
            {"role": "user", "content": prompt},
        ],
        n=boN,                                     # generate boN completions
        **chat_extra_kwargs(model_name, temperature),
        logprobs=want_logprobs,
    )

    best_draft = None
    best_score = -float("inf")

    for choice in response.choices:
        draft = choice.message.content.strip()

        if want_logprobs:
            lp_obj = choice.logprobs
            token_logps = (
                lp_obj.token_logprobs
                if hasattr(lp_obj, "token_logprobs")
                else [tok.logprob for tok in lp_obj.content]
            )
            score = sum(token_logps)
        else:
            score = 0.0

        if score > best_score:
            best_score = score
            best_draft = draft

    best_usage = response.usage   # aggregated stats for all n completions
    return {"text": best_draft, "usage": best_usage}


def run_debate_match(match_id,
                     debater_pro: dict,
                     debater_con: dict,
                     static_context,
                     initial_topic,
                     pro_starts: bool,
                     progress_turn_cb=None,
                     quiet=False):
    """
    Runs one complete debate match.
    Returns the match data dictionary.
    """
    match_data = {
        "match_id": match_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "turns": []
    }
    

    speakers = [(SIDE_A_LABEL, "🔵"), (SIDE_B_LABEL, "🔴")] if pro_starts \
     else [(SIDE_B_LABEL, "🔴"), (SIDE_A_LABEL, "🔵")]

    debater_map = {
    SIDE_A_LABEL: debater_pro,  # Strategy 1
    SIDE_B_LABEL: debater_con,  # Strategy 2
}
    starting_speaker, starting_emoji = speakers[0]
    side_label = starting_speaker


    messages = [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user", "content": f"Debate topic: {initial_topic}\n\nContext:\n{static_context}"}
    ]

    print(f"\nGenerating opening variants for {side_label} side...")

    d = debater_map[starting_speaker]
    result = generate_openings(
            side=starting_speaker,   
            boN=d["boN"],
            temperature=d["temperature"],
            model_name=d["model"],
            static_context=static_context,
            initial_topic=initial_topic,
        )
    selected_opening = result["text"]
    selected_opening = _trim_to_sentence_boundary(selected_opening)

    usage_info       = result["usage"]
    best_completion_tokens = min(config.MAX_TOKENS_PER_RESPONSE,
                             usage_info.completion_tokens // d["boN"])


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
        "tokens_used_prompt": usage_info.prompt_tokens,
        "tokens_used_completion": best_completion_tokens,
        "tokens_used_completion_all": usage_info.completion_tokens,
        "content": selected_opening
    })
    update_turn_stats(usage_info.prompt_tokens, best_completion_tokens)

    if progress_turn_cb and quiet:
        progress_turn_cb()

    # Continue debate for subsequent turns

    for turn in range(2, config.TURNS_PER_MATCH + 1):
        # Print new round header only when entering a new round
        if turn > 2 and (turn % 2 == 1):
            round_number = (turn + 1) // 2
            print(f"\n🔁 Starting Round {round_number}")
        
        current_speaker, emoji = speakers[(turn - 1) % 2]
        print(f"\n{emoji} {current_speaker}'s Turn {turn}")

        stance = SIDE_STANCE.get(current_speaker, "")
        messages.append({
                    "role": "user",
                    "content": (
                    f"You are advocating for {current_speaker}. {stance}\n"
                    "Base your response only on the provided context. Do not include salutations.\n\n"
                    f"{current_speaker}, please respond to your opponent."
                    ),
                    })
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
        cleaned_content = _trim_to_sentence_boundary(cleaned_content)

        print(f"{emoji} [Prompt tokens: {usage.prompt_tokens}, Completion tokens: {usage.completion_tokens}, Total tokens: {usage.total_tokens}]")
        print(cleaned_content)

        messages.append({"role": "assistant", "content": cleaned_content})
        match_data["turns"].append({
            "turn_number": turn,
            "speaker": current_speaker,
            "tokens_used_prompt": usage.prompt_tokens,
            "tokens_used_completion": usage.completion_tokens,
            "content": cleaned_content
        })
        update_turn_stats(usage.prompt_tokens,usage.completion_tokens)

        # per-turn progress dot (quiet mode only)
        if progress_turn_cb and quiet:
            progress_turn_cb()

        if turn < config.TURNS_PER_MATCH:
            next_speaker, _ = speakers[(turn) % 2]
            stance = SIDE_STANCE.get(current_speaker, "")
            messages.append({
                "role": "user",
                "content": (
                f"You are advocating for {current_speaker}. {stance}\n"
                "Base your response only on the provided context. Do not include salutations.\n\n"
                f"{current_speaker}, please respond to your opponent."
                ),
                })
        time.sleep(1)  # Pacing delay

    # Invoke the judge after the debate match is complete
    verdict = judge_debate(match_data)
    winner = match_data.get("judge_evaluation", {}).get("winner")
    match_data["winner"] = winner
    update_match_stats(winner_label=winner, verdict_text=verdict)
    # Legacy fields
    match_data["debater_pro"] = debater_pro["id"]
    match_data["debater_con"] = debater_con["id"]
    # New neutral 
    match_data["side_labels"] = [SIDE_A_LABEL, SIDE_B_LABEL]
    match_data["side_to_debater_id"] = {
    SIDE_A_LABEL: debater_pro["id"],   # Strategy 1 by our convention
    SIDE_B_LABEL: debater_con["id"],   # Strategy 2
    }
    match_data["start_label"] = speakers[0][0]        # who opened
    match_data["winner"] = match_data.get("judge_evaluation", {}).get("winner")

    match_data["verdict"]     = verdict

    return match_data



def run_all_matches(static_context, initial_topic, progress_cb=None, progress_turn_cb=None, quiet=False):
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
                         pro_starts=True,
                         progress_turn_cb=progress_turn_cb,
                         quiet=quiet)
            matches_data.append(m)

            if progress_cb:
                progress_cb()
            if not quiet:
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
                         pro_starts=False,
                         progress_turn_cb=progress_turn_cb,
                         quiet=quiet)
            matches_data.append(m)

            if progress_cb:
                progress_cb()
            if not quiet:
                print(f"\n✅ Debate Match {match_id} complete.")

            match_id += 1

    return matches_data


import config

def chat_extra_kwargs(model_name: str, temperature: float) -> dict:
    """
    Return the correct keyword dict for an OpenAI chat request:
    • o-series models:     max_completion_tokens, temperature=1
    • GPT-series models:   max_tokens,           temperature as given
    """
    if model_name.startswith("o"):          # e.g. "o4-mini"
        return {
            "max_completion_tokens": config.MAX_TOKENS_PER_RESPONSE,
            "temperature": 1,               # o-series ignores other values
        }
    else:
        return {
            "max_tokens": config.MAX_TOKENS_PER_RESPONSE,
            "temperature": temperature,
        }

# def supports_logprobs(model_name: str) -> bool:
#     """Return True iff the model family allows the `logprobs` option."""
#     return not model_name.startswith("o")      # o-series models currently do not support logprobs  

def supports_logprobs(model_name: str) -> bool:
    # Explicit map is safest
    if model_name.startswith("gpt-4o-mini"):
        return True
    if model_name.startswith("gpt-5-mini"):
        return False
    # Default assumption for other models
    return False
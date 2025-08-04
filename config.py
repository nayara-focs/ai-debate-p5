import os
import openai
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# OpenAI API key (make sure this is set in your .env file)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Debate Configuration
# MODEL = "gpt-4o-mini"
MODEL = "gpt-4o-mini-2024-07-18"
TEMPERATURE = 0.7
MAX_TOKENS_PER_RESPONSE = 300
TURNS_PER_MATCH = 2           # Total turns per match, including the opening turn.

# Prompts and Topic
SYSTEM_PROMPT = (
    "You are participating in a structured debate. "
    "Respond clearly and concisely, address your opponent's points directly, "
    "use the context provided, and keep your responses within the token limit."
)
INITIAL_TOPIC = "Should we follow the recommendations of the 2023 Particle Physics P5 Report?"

# File name for the P5 report text
P5_REPORT_FILE: Path = Path(__file__).resolve().parent /"docs"/"p5_report.txt"

# ------------------------------------------------------------------
# Debater roster for Elo experiment (budget-trimmed)
# ------------------------------------------------------------------

DEBATERS = [
    {
        "id": "A",
        "boN": 1,
        "temperature": 0.7,
        "model": "gpt-4o-mini",
    },
    {
        "id": "B",
        "boN": 4,
        "temperature": 0.7,
        "model": "gpt-4o-mini",
    },
    {
        "id": "C",
        "boN": 8,
        "temperature": 0.7,
        "model": "gpt-4o-mini",
    },
    {
        "id": "D",
        "boN": 1,
        "temperature": 1,
        "model": "o3-mini",          # o-series reasoning model
    },
]


REPEATS_PER_PAIR = 1  # how many independent repeats per ordered direction

"""
Ordered pairs  :  n x (n - 1)          # product() excluding self-play
Opener flips   :  x 2                  # Pro-opens, then Con-opens
Repeats        :  x R
---------------------------------------------
Total matches  =  n · (n - 1) · 2 · R

"""
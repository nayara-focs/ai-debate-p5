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
MODEL = "gpt-4o-mini"
# MODEL = "gpt-4o"

TEMPERATURE = 0.7
MAX_TOKENS_PER_RESPONSE = 400
TURNS_PER_MATCH = 6           # Total turns per match, including the opening turn.

# Prompts and Topic
SYSTEM_PROMPT = (
    "You are participating in a structured debate. "
    "Respond clearly and concisely, address your opponent's points directly, and base your answer only on the provided context. "
    "Use a concise, professional tone—no salutations or rhetorical flourishes. "
    "End at a complete sentence; if you are near the token limit, finish the current sentence and stop. "
    "Finish with a one-sentence takeaway if needed. "
    f"Keep your response within the token limit ({MAX_TOKENS_PER_RESPONSE})."
)

INITIAL_TOPIC = (
    "Given the following two official planning documents (US 2023 P5 report and CERN's 2025 FCC Feasibility Study), "
    "what is the most compelling strategy for advancing high-energy physics over the next decade? "
    "Compare scientific reach, timelines, cost/risk, and international coordination. Argue as assigned."
)


# File name 
STATIC_CONTEXT_FILE: Path = Path(__file__).resolve().parent /"docs"/"context_p5_plus_fcc.txt"

SIDE_A_LABEL = "Strategy 1"
SIDE_B_LABEL = "Strategy 2"
ALLOWED_LABELS = (SIDE_A_LABEL, SIDE_B_LABEL)

SIDE_STANCE = {
    SIDE_A_LABEL: "Emphasise the US P5-aligned roadmap.",
    SIDE_B_LABEL: "Emphasise the FCC-first roadmap.",
}

# ------------------------------------------------------------------
# Debater roster for Elo experiment (budget-trimmed)
# ------------------------------------------------------------------

DEBATERS = [
    {
        "id": "A",
        "boN": 1,
        "temperature": 0.3,
        "model": "gpt-4o-mini",
    },
    {
        "id": "B",
        "boN": 1,
        "temperature": 0.7,
        "model": "gpt-4o-mini",
    },
    {
        "id": "C",
        "boN": 1,
        "temperature": 1.0,
        "model": "gpt-4o-mini",
    },
]


REPEATS_PER_PAIR = 5  # how many independent repeats per ordered direction

"""
Ordered pairs  :  n x (n - 1)          # product() excluding self-play
Opener flips   :  x 2                  # Pro-opens, then Con-opens
Repeats        :  x R
---------------------------------------------
Total matches  =  n · (n - 1) · 2 · R

"""
import os
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API key (make sure this is set in your .env file)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Debate Configuration
MODEL = "gpt-4o-mini"              
TEMPERATURE = 0.7
MAX_TOKENS_PER_RESPONSE = 800
NUM_MATCHES = 4               # Number of debate matches to run (for testing)
TURNS_PER_MATCH = 4           # Total turns per match, including the opening turn.
OPENING_VARIANTS = 2          # Number of distinct opening variants to generate

# Prompts and Topic
SYSTEM_PROMPT = (
    "You are participating in a structured debate. "
    "Respond clearly and concisely, address your opponent's points directly, "
    "use the context provided, and keep your responses within the token limit."
)
INITIAL_TOPIC = "Should we follow the recommendations of the 2023 Particle Physics P5 Report?"

# File name for the P5 report text
P5_REPORT_FILE = "p5_report.txt"

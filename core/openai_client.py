# core/openai_client.py
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Get the project root directory (parent of 'core' directory)
project_root = Path(__file__).resolve().parent.parent
env_path = project_root / ".env"

# Load .env file from project root (override=True ensures fresh read)
load_dotenv(dotenv_path=env_path, override=True)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(f"OPENAI_API_KEY not found in {env_path}. Please ensure your .env file contains OPENAI_API_KEY=your_key_here")

client = OpenAI(api_key=api_key)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

def chat_completion(messages):
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.3,
    )
    return response.choices[0].message.content

import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

def _load_env():
    # Load project .env if present; do not fail if missing.
    if ENV_PATH.exists():
        load_dotenv(dotenv_path=ENV_PATH, override=True)
    else:
        load_dotenv(override=True)

def get_client() -> OpenAI:
    _load_env()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Add it to your environment or a .env file in the project root."
        )
    return OpenAI(api_key=api_key)

def chat_completion(messages):
    _load_env()
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
    )
    return response.choices[0].message.content

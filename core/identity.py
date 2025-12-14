import os
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROFILE = PROJECT_ROOT / "identity" / "example_profile.yaml"
PERSONAL_PROFILE = PROJECT_ROOT / "identity" / "ronald_profile.yaml"

def load_profile():
    # Allow override from env
    profile_path = os.getenv("PROFILE_PATH")
    if profile_path:
        p = Path(profile_path).expanduser()
        return yaml.safe_load(p.read_text(encoding="utf-8"))

    # Prefer personal profile if it exists locally, else use example
    if PERSONAL_PROFILE.exists():
        return yaml.safe_load(PERSONAL_PROFILE.read_text(encoding="utf-8"))

    return yaml.safe_load(DEFAULT_PROFILE.read_text(encoding="utf-8"))

def build_system_prompt(profile: dict) -> str:
    # Keep this concise; avoid dumping the entire YAML verbatim long-term.
    return f"""
You are a personal AI assistant acting as a second brain.
Follow the user's tone, communication style, preferences, and guardrails in the profile.

Profile (structured):
{profile}

Rules:
- Respond in the user's voice.
- Be clear, practical, and risk-aware.
- If unsure, state assumptions.
"""

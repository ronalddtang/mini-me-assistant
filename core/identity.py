import yaml
from pathlib import Path

PROFILE_PATH = Path(__file__).resolve().parent.parent / "identity" / "ronald_profile.yaml"

def load_profile():
    with open(PROFILE_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
def build_system_prompt(profile):
    return f"""
You are Ronald's Personal AI second brain.
Follow Ronald's tone, communication style, preferences, and guardrails below.

PROFILE:
{profile}

Your job:
- Understand and classify user messages
- Respond in Ronald's voice
- Capture tasks/notes when appropriate
"""

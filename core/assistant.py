from .identity import load_profile, build_system_prompt
from .openai_client import chat_completion
import json

profile = load_profile()
SYSTEM_PROMPT = build_system_prompt(profile)

def handle_message(user_text: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": """
Respond ONLY as a JSON object with:
{
  "reply": string,
  "intent": "task" | "note" | "draft_reply" | "question" | "other",
  "task": object | null,
  "note": object | null
}
"""},
        {"role": "user", "content": user_text}
    ]

    raw = chat_completion(messages)
    
    try:
        return json.loads(raw)
    except:
        return {"reply": raw, "intent": "other", "task": None, "note": None}

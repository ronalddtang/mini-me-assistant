from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Any, Dict

from .identity import load_profile, build_system_prompt
from .openai_client import chat_completion

profile = load_profile()
SYSTEM_PROMPT = build_system_prompt(profile)

# ---------- Prompt blocks ----------

_INTENT_PROMPT = """
Classify the user's message into exactly ONE intent from this list:
- task
- note
- draft_reply
- question
- other

Rules:
- "task": user wants something done (create, schedule, remember to do, follow up).
- "note": user is providing information to store/remember (facts, context, decisions).
- "draft_reply": user wants a draft response (email/message).
- "question": user is asking for info/explanation and not asking to store/execute.
- "other": anything else.

Return ONLY the intent string, nothing else.
"""

_JSON_INSTRUCTIONS = """
Return RAW JSON ONLY (no markdown, no triple backticks).
Schema:
{
  "reply": string,
  "intent": "task" | "note" | "draft_reply" | "question" | "other",
  "task": object | null,
  "note": object | null
}

If intent is "task", include task with fields:
- title (string)
- description (string)
- due_date (string|null)
- tags (array of strings)

If intent is "note", include note with fields:
- title (string)
- body (string)
- tags (array of strings)

If intent is "draft_reply", put the drafted text in "reply" and leave task/note null.
If intent is "question" or "other", keep task/note null.
"""

_CONVERSATION_PROMPT = """
You are conversing naturally with the user.
Do NOT output JSON.
Be helpful, concise, and aligned to the user's voice and guardrails.
"""

# ---------- Helpers ----------

def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return t


def _classify_intent(user_text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": _INTENT_PROMPT},
        {"role": "user", "content": user_text},
    ]

    raw = chat_completion(messages).strip().lower()
    raw = raw.split()[0] if raw else "other"

    allowed = {"task", "note", "draft_reply", "question", "other"}
    return raw if raw in allowed else "other"


def _handle_action(user_text: str, intent: str) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": _JSON_INSTRUCTIONS},
        {"role": "user", "content": f"INTENT_HINT: {intent}\n\nUSER_MESSAGE: {user_text}"},
    ]

    raw = chat_completion(messages)
    raw = _strip_code_fences(raw)

    try:
        data = json.loads(raw)
    except JSONDecodeError:
        return {"reply": raw, "intent": "other", "task": None, "note": None}

    return {
        "reply": data.get("reply", ""),
        "intent": data.get("intent", intent),
        "task": data.get("task"),
        "note": data.get("note"),
    }


def _handle_conversation(user_text: str, intent: str) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": _CONVERSATION_PROMPT},
        {"role": "user", "content": user_text},
    ]

    reply = chat_completion(messages).strip()
    return {"reply": reply, "intent": intent, "task": None, "note": None}


# ---------- Public API ----------

def handle_message(user_text: str) -> Dict[str, Any]:
    intent = _classify_intent(user_text)

    if intent in {"task", "note", "draft_reply"}:
        return _handle_action(user_text, intent)

    return _handle_conversation(user_text, intent)

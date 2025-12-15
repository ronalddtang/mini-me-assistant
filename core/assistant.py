from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Any, Dict, Optional

from .identity import load_profile, build_system_prompt
from .openai_client import chat_completion as default_chat_completion
from .memory import get_memory_manager

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


def _classify_intent(user_text: str, memory_context: Optional[str] = None, memory_manager=None) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": _INTENT_PROMPT},
    ]
    
    # Add memory context if available
    if memory_context:
        messages.append({"role": "system", "content": f"Context:\n{memory_context}"})
    
    messages.append({"role": "user", "content": user_text})

    # Use Memori's chat completion if available, otherwise use default
    if memory_manager and hasattr(memory_manager, 'chat_completion'):
        raw = memory_manager.chat_completion(messages).strip().lower()
    else:
        raw = default_chat_completion(messages).strip().lower()
    raw = raw.split()[0] if raw else "other"

    allowed = {"task", "note", "draft_reply", "question", "other"}
    return raw if raw in allowed else "other"


def _handle_action(user_text: str, intent: str, memory_context: Optional[str] = None, memory_manager=None) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": _JSON_INSTRUCTIONS},
    ]
    
    # Add memory context if available
    if memory_context:
        messages.append({"role": "system", "content": f"Context:\n{memory_context}"})
    
    messages.append({"role": "user", "content": f"INTENT_HINT: {intent}\n\nUSER_MESSAGE: {user_text}"})

    # Use Memori's chat completion if available, otherwise use default
    if memory_manager and hasattr(memory_manager, 'chat_completion'):
        raw = memory_manager.chat_completion(messages)
    else:
        raw = default_chat_completion(messages)
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


def _handle_conversation(user_text: str, intent: str, memory_context: Optional[str] = None, memory_manager=None) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": _CONVERSATION_PROMPT},
    ]
    
    # Add memory context if available
    if memory_context:
        messages.append({"role": "system", "content": f"Context:\n{memory_context}"})
    
    messages.append({"role": "user", "content": user_text})

    # Use Memori's chat completion if available, otherwise use default
    if memory_manager and hasattr(memory_manager, 'chat_completion'):
        reply = memory_manager.chat_completion(messages).strip()
    else:
        reply = default_chat_completion(messages).strip()
    return {"reply": reply, "intent": intent, "task": None, "note": None}


# ---------- Public API ----------

def handle_message(user_text: str, agent_id: str = "main_assistant") -> Dict[str, Any]:
    """
    Handle a user message with memory integration.
    
    Args:
        user_text: The user's message
        agent_id: Unique identifier for the agent/namespace (for memory isolation).
                  Use different IDs for different agents (e.g., "main_assistant", "email_agent")
        
    Returns:
        Dictionary with reply, intent, task, and note
    """
    # Get memory manager for this agent namespace
    memory_manager = None
    memory_context = None
    
    try:
        memory_manager = get_memory_manager(agent_id)
        memory_context = memory_manager.build_memory_context(user_text)
    except Exception as e:
        # If memory fails, continue without it
        print(f"Warning: Memory not available: {e}")
        memory_context = None
        memory_manager = None
    
    # Classify intent with memory context
    intent = _classify_intent(user_text, memory_context, memory_manager)
    
    # Handle based on intent
    if intent in {"task", "note", "draft_reply"}:
        result = _handle_action(user_text, intent, memory_context, memory_manager)
    else:
        result = _handle_conversation(user_text, intent, memory_context, memory_manager)
    
    # Store conversation and important information in memory
    if memory_manager:
        try:
            # Always store the conversation
            memory_manager.add_conversation(user_text, result.get("reply", ""))
            
            # Store notes and tasks explicitly
            if result.get("note"):
                note = result["note"]
                note_content = f"{note.get('title', 'Note')}: {note.get('body', '')}"
                memory_manager.store_memory(
                    content=note_content,
                    memory_type="note",
                    metadata={"tags": note.get("tags", [])}
                )
            
            if result.get("task"):
                task = result["task"]
                task_content = f"Task: {task.get('title', '')} - {task.get('description', '')}"
                memory_manager.store_memory(
                    content=task_content,
                    memory_type="task",
                    metadata={
                        "due_date": task.get("due_date"),
                        "tags": task.get("tags", [])
                    }
                )
            
            # Memori automatically captures conversations, so no need for manual storage
            # The preference detection is handled automatically by Memori
        except Exception as e:
            print(f"Warning: Failed to store in memory: {e}")
    
    return result

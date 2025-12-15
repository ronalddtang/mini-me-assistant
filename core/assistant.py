from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Any, Dict, Optional

from .identity import load_profile, build_system_prompt
from .openai_client import chat_completion as default_chat_completion
from .memory import get_memory_manager
from .email_agent import EmailAgent, EmailAgentError

profile = load_profile()
SYSTEM_PROMPT = build_system_prompt(profile)
_EMAIL_AGENT: Optional[EmailAgent] = None
_EMAIL_AGENT_ERROR: Optional[Exception] = None
_EMAIL_KEYWORDS = {
    "email",
    "emails",
    "inbox",
    "mailbox",
    "gmail",
    "draft",
    "drafts",
    "reply",
    "replies",
    "unread",
}

# ---------- Prompt blocks ----------

_INTENT_PROMPT = """
Classify the user's message into exactly ONE intent from this list:
- task
- note
- draft_reply
- question
- other
- email

Rules:
- "task": user wants something done (create, schedule, remember to do, follow up).
- "note": user is providing information to store/remember (facts, context, decisions).
- "draft_reply": user wants a draft response (email/message).
- "question": user is asking for info/explanation and not asking to store/execute.
- "other": anything else.
- "email": user wants inbox summaries, email context, or drafting/sending help.

Return ONLY the intent string, nothing else.
"""

_JSON_INSTRUCTIONS = """
Return RAW JSON ONLY (no markdown, no triple backticks).
Schema:
{
  "reply": string,
  "intent": "task" | "note" | "draft_reply" | "question" | "other" | "email",
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

_EMAIL_ROUTING_PROMPT = """
You convert user requests into email agent commands. Respond with RAW JSON only:
{
  "action": "summarize_inbox" | "summarize_thread" | "draft_reply" | "send_draft" | "status",
  "query": string | null,
  "instructions": string | null,
  "confirmation": "send_now" | "needs_confirmation" | null
}

Guidance:
- Use "summarize_inbox" to check the inbox or unread mail.
- Use "summarize_thread" when they reference a sender/topic; include keywords in query.
- Use "draft_reply" when they want a reply; include query to locate the thread and write instructions describing style/goals.
- Use "send_draft" only if they explicitly approve sending the pending draft; set confirmation to "send_now" when approval is clear.
- Use "status" if they ask about pending drafts or email tasks.
- Default confirmation to "needs_confirmation" unless they explicitly request sending now.
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

    allowed = {"task", "note", "draft_reply", "question", "other", "email"}
    return raw if raw in allowed else "other"


def _get_email_agent() -> EmailAgent:
    global _EMAIL_AGENT, _EMAIL_AGENT_ERROR
    if _EMAIL_AGENT is None and _EMAIL_AGENT_ERROR is None:
        try:
            _EMAIL_AGENT = EmailAgent()
        except Exception as exc:
            _EMAIL_AGENT_ERROR = exc
    if _EMAIL_AGENT is None:
        raise EmailAgentError(f"Email agent unavailable: {_EMAIL_AGENT_ERROR}")
    return _EMAIL_AGENT


def _parse_email_command(user_text: str, memory_context: Optional[str] = None, memory_manager=None) -> Dict[str, Any]:
    messages = [{"role": "system", "content": _EMAIL_ROUTING_PROMPT}]
    if memory_context:
        messages.append({"role": "system", "content": f"Context:\n{memory_context}"})
    messages.append({"role": "user", "content": user_text})

    if memory_manager and hasattr(memory_manager, "chat_completion"):
        raw = memory_manager.chat_completion(messages)
    else:
        raw = default_chat_completion(messages)
    raw = _strip_code_fences(raw)
    try:
        return json.loads(raw)
    except JSONDecodeError:
        return {"action": "summarize_inbox", "query": None, "instructions": None, "confirmation": None}


def _handle_email(user_text: str, memory_context: Optional[str] = None, memory_manager=None) -> Dict[str, Any]:
    try:
        agent = _get_email_agent()
    except EmailAgentError as exc:
        return {"reply": str(exc), "intent": "email", "task": None, "note": None}

    command = _parse_email_command(user_text, memory_context, memory_manager)
    action = command.get("action", "summarize_inbox")
    query = (command.get("query") or "").strip()
    instructions = (command.get("instructions") or "").strip()
    confirmation = command.get("confirmation")

    try:
        if action == "summarize_thread":
            if not query:
                raise EmailAgentError("Please specify which conversation to summarize.")
            reply_text = agent.summarize_thread(query)
        elif action == "draft_reply":
            if not query:
                raise EmailAgentError("I need a keyword or sender to find that conversation.")
            result = agent.draft_reply(query, instructions or "Keep it concise and helpful.")
            reply_text = f"{result['message']}\n\nPreview:\n{result['preview']}"
        elif action == "send_draft":
            if not agent.has_pending_draft():
                raise EmailAgentError("There's no draft waiting to send.")
            if confirmation != "send_now":
                reply_text = "I have the draft ready. Say 'yes, send it' when you're sure."
            else:
                reply_text = agent.send_pending_draft()
        elif action == "status":
            if agent.has_pending_draft():
                reply_text = "There's a draft waiting for approval. Tell me when to send it."
            else:
                reply_text = "No drafts are pending approval right now."
        else:
            reply_text = agent.summarize_inbox()
    except EmailAgentError as exc:
        reply_text = str(exc)
    except Exception as exc:
        reply_text = f"Sorry, the email agent ran into an issue: {exc}"

    return {"reply": reply_text, "intent": "email", "task": None, "note": None}


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
    if intent != "email":
        normalized = user_text.lower()
        if any(keyword in normalized for keyword in _EMAIL_KEYWORDS):
            intent = "email"
    
    # Handle based on intent
    if intent == "email":
        result = _handle_email(user_text, memory_context, memory_manager)
    elif intent in {"task", "note", "draft_reply"}:
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

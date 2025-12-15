"""Gmail-backed email agent managed by Emily.

Provides inbox summaries, thread summaries, safe draft creation, and
send-on-confirmation controls so the main assistant can delegate email
work.
"""
from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from .openai_client import chat_completion

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOKEN_PATH = PROJECT_ROOT / "gmail_token.json"
DEFAULT_CREDENTIALS_PATH = PROJECT_ROOT / "gmail_credentials.json"

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.send",
]


class EmailAgentError(RuntimeError):
    """Raised when an email action cannot be completed."""


@dataclass
class EmailPromptConfig:
    inbox_summary: str = (
        "You summarize the user's inbox. Highlight urgent senders, dates, and"
        " action items. Be concise and note unanswered questions."
    )
    thread_summary: str = (
        "Summarize this email thread with key decisions, blockers, and next"
        " steps so the user can quickly respond."
    )
    draft_reply: str = (
        "Write a clear, polite reply that matches the user's voice. Mention any"
        " commitments or follow-ups from the instructions, and keep it concise."
    )


class EmailAgent:
    """Encapsulates Gmail access and LLM-powered workflows."""

    def __init__(
        self,
        token_path: Optional[Path] = None,
        credentials_path: Optional[Path] = None,
        prompts: Optional[EmailPromptConfig] = None,
    ):
        self.token_path = Path(os.getenv("GMAIL_TOKEN_PATH", token_path or DEFAULT_TOKEN_PATH))
        self.credentials_path = Path(
            os.getenv("GMAIL_CREDENTIALS_PATH", credentials_path or DEFAULT_CREDENTIALS_PATH)
        )
        self.sender_address = os.getenv("EMAIL_SENDER_ADDRESS")
        self.prompts = prompts or EmailPromptConfig()
        self._service = None
        self.pending_draft: Optional[Dict[str, Any]] = None

    # ---------- Public API ----------

    def summarize_inbox(self, limit: Optional[int] = None) -> str:
        limit = limit or int(os.getenv("EMAIL_SUMMARY_LIMIT", "10"))
        try:
            service = self._get_service()
            response = service.users().messages().list(
                userId="me", labelIds=["INBOX"], maxResults=limit, q="label:inbox"
            ).execute()
            refs = response.get("messages", [])
            if not refs:
                return "Your inbox is clear right now."

            messages = [self._fetch_message(ref["id"]) for ref in refs]
            content = self._format_messages_for_summary(messages)
            return self._run_prompt(self.prompts.inbox_summary, content)
        except HttpError as exc:
            raise EmailAgentError(f"Gmail error: {exc}") from exc

    def summarize_thread(self, query: str) -> str:
        thread = self._find_thread(query)
        if not thread:
            raise EmailAgentError("I couldn't find a matching conversation in your inbox.")
        formatted = self._format_thread(thread)
        return self._run_prompt(self.prompts.thread_summary, formatted)

    def draft_reply(self, query: str, instructions: str) -> Dict[str, Any]:
        thread = self._find_thread(query)
        if not thread:
            raise EmailAgentError("I couldn't locate that thread. Try a different keyword or sender.")

        formatted = self._format_thread(thread)
        prompt = (
            f"{self.prompts.draft_reply}\n\nTHREAD:\n{formatted}\n\n"
            f"INSTRUCTIONS:\n{instructions or 'Reply briefly.'}"
        )
        reply_text = self._run_prompt("You craft helpful email replies.", prompt)

        latest = thread["messages"][-1]
        email = self._build_reply(latest, reply_text)
        raw = base64.urlsafe_b64encode(email.as_bytes()).decode()

        draft_id = str(uuid4())
        self.pending_draft = {
            "id": draft_id,
            "raw": raw,
            "preview": reply_text,
            "subject": self._get_header(latest, "Subject") or "(no subject)",
        }
        return {
            "draft_id": draft_id,
            "message": "Here's a draft. Let me know if you'd like me to send it.",
            "preview": reply_text,
            "requires_confirmation": True,
        }

    def send_pending_draft(self) -> str:
        if not self.pending_draft:
            raise EmailAgentError("There's no draft waiting for approval.")
        try:
            service = self._get_service()
            body = {"raw": self.pending_draft["raw"]}
            service.users().messages().send(userId="me", body=body).execute()
        except HttpError as exc:
            raise EmailAgentError(f"Failed to send the email: {exc}") from exc

        subject = self.pending_draft.get("subject", "email")
        self.pending_draft = None
        return f"Done — I sent your reply about '{subject}'."

    def has_pending_draft(self) -> bool:
        return self.pending_draft is not None

    # ---------- Gmail helpers ----------

    def _get_service(self):
        if self._service is None:
            self._service = self._build_service()
        return self._service

    def _build_service(self):
        creds = None
        if self.token_path.exists():
            creds = Credentials.from_authorized_user_file(str(self.token_path), SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not self.credentials_path.exists():
                    raise EmailAgentError(
                        "Gmail credentials not found. Set GMAIL_CREDENTIALS_PATH or place the"
                        f" file at {self.credentials_path}."
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self.credentials_path), SCOPES
                )
                creds = flow.run_local_server(port=0)
            self.token_path.write_text(creds.to_json())
        return build("gmail", "v1", credentials=creds, cache_discovery=False)

    def _fetch_message(self, message_id: str) -> Dict[str, Any]:
        service = self._get_service()
        return service.users().messages().get(userId="me", id=message_id, format="full").execute()

    def _find_thread(self, query: str) -> Optional[Dict[str, Any]]:
        try:
            service = self._get_service()
            result = service.users().messages().list(userId="me", q=query, maxResults=1).execute()
            refs = result.get("messages", [])
            if not refs:
                return None
            thread_id = refs[0]["threadId"]
            return service.users().threads().get(userId="me", id=thread_id, format="full").execute()
        except HttpError as exc:
            raise EmailAgentError(f"Gmail error while searching: {exc}") from exc

    # ---------- Formatting ----------

    def _format_messages_for_summary(self, messages: List[Dict[str, Any]]) -> str:
        chunks = []
        for message in messages:
            sender = self._get_header(message, "From") or "Unknown sender"
            subject = self._get_header(message, "Subject") or "(no subject)"
            snippet = message.get("snippet", "")
            chunks.append(f"From: {sender}\nSubject: {subject}\nSnippet: {snippet}\n---")
        return "\n".join(chunks)

    def _format_thread(self, thread: Dict[str, Any]) -> str:
        entries = []
        for message in thread.get("messages", []):
            sender = self._get_header(message, "From")
            date = self._get_header(message, "Date")
            body = self._extract_plain_text(message)
            entries.append(f"[{date}] {sender}:\n{body}\n")
        return "\n".join(entries)

    def _run_prompt(self, system_prompt: str, user_content: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        return chat_completion(messages).strip()

    def _build_reply(self, latest_message: Dict[str, Any], reply_text: str) -> EmailMessage:
        if not self.sender_address:
            raise EmailAgentError("EMAIL_SENDER_ADDRESS is not set in your environment.")
        reply = EmailMessage()
        reply["To"] = self._extract_reply_to(latest_message)
        reply["From"] = self.sender_address
        subject = self._get_header(latest_message, "Subject") or "(no subject)"
        reply["Subject"] = subject if subject.lower().startswith("re:") else f"Re: {subject}"
        msg_id = self._get_header(latest_message, "Message-ID")
        if msg_id:
            reply["In-Reply-To"] = msg_id
            reply["References"] = msg_id
        reply.set_content(reply_text)
        return reply

    # ---------- Utility ----------

    def _get_header(self, message: Dict[str, Any], name: str) -> str:
        headers = message.get("payload", {}).get("headers", [])
        for header in headers:
            if header.get("name", "").lower() == name.lower():
                return header.get("value", "")
        return ""

    def _extract_plain_text(self, message: Dict[str, Any]) -> str:
        payload = message.get("payload", {})
        data = payload.get("body", {}).get("data")
        if data:
            return base64.urlsafe_b64decode(data).decode(errors="ignore")
        for part in payload.get("parts", []) or []:
            mime = part.get("mimeType")
            if mime == "text/plain":
                part_data = part.get("body", {}).get("data")
                if part_data:
                    return base64.urlsafe_b64decode(part_data).decode(errors="ignore")
        return message.get("snippet", "")

    def _extract_reply_to(self, message: Dict[str, Any]) -> str:
        reply_to = self._get_header(message, "Reply-To")
        return reply_to or self._get_header(message, "From")

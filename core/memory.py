import asyncio
import inspect
import os
import re
from pathlib import Path
from typing import Optional, List, Dict, Any
from uuid import uuid4

from dotenv import load_dotenv

# Try to import memori and sqlalchemy, but make it optional
try:
    from memori import Memori
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker
    from openai import OpenAI
    MEMORI_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    # Handle cases where memori or its dependencies aren't available
    MEMORI_AVAILABLE = False
    Memori = None  # type: ignore
    create_engine = None  # type: ignore
    text = None  # type: ignore
    sessionmaker = None  # type: ignore
    OpenAI = None  # type: ignore
    _import_error = str(e)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
MEMORY_DB_PATH = PROJECT_ROOT / "memory.db"

_STOPWORDS = {
    "i", "me", "my", "you", "your", "the", "and", "but", "for",
    "are", "about", "that", "this", "with", "have", "what", "when",
    "who", "how", "why", "which", "favorite", "favourite"
}

def _load_env():
    """Load environment variables from .env file if present."""
    if ENV_PATH.exists():
        load_dotenv(dotenv_path=ENV_PATH, override=True)
    else:
        load_dotenv(override=True)


class MemoryManager:
    """Manages short-term and long-term memory using Memori AI."""
    
    def __init__(self, agent_id: str, namespace: Optional[str] = None):
        """
        Initialize memory manager for a specific agent/namespace.
        
        Args:
            agent_id: Unique identifier for the agent (e.g., "main_assistant", "email_agent")
            namespace: Optional namespace for organizing memories (defaults to agent_id)
        """
        if not MEMORI_AVAILABLE:
            error_msg = "Memori is not installed or has missing dependencies."
            if '_import_error' in globals():
                error_msg += f" Error: {_import_error}"
            error_msg += " Install with: pip install memori sqlalchemy"
            raise ImportError(error_msg)
        
        _load_env()
        
        self.agent_id = agent_id
        self.namespace = namespace or agent_id
        self.entity_id: Optional[int] = None
        self.process_record_id: Optional[int] = None
        
        # Get OpenAI API key (required by Memori)
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Required for Memori AI memory functionality."
            )
        
        # Database connection string (default to SQLite in project root)
        # Use absolute path for SQLite to avoid issues
        # Only use env var if it's explicitly set and not the default PostgreSQL template
        env_db_connection = os.getenv("MEMORI_DB_CONNECTION")
        if env_db_connection and not env_db_connection.startswith("postgresql://user:password"):
            # User has explicitly set a custom database connection
            db_connection = env_db_connection
        else:
            # Default to SQLite
            db_connection = f"sqlite:///{MEMORY_DB_PATH.absolute()}"
        
        # Ensure the SQLite database file directory exists
        MEMORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        # Create SQLAlchemy engine with SQLite-specific settings
        # Use check_same_thread=False for SQLite to allow multiple threads
        # Use autocommit=False and autoflush=True for better transaction control
        engine = create_engine(
            db_connection,
            connect_args={"check_same_thread": False} if "sqlite" in db_connection else {},
            echo=False,  # Set to True for SQL debugging
            pool_pre_ping=True,  # Verify connections before using
        )
        Session = sessionmaker(bind=engine, autocommit=False, autoflush=True)
        self.Session = Session  # Store session factory
        self.session = Session()  # Store session for reuse
        
        # Initialize Memori with the database session
        # Memori should use the session we provide, not try to create its own connection
        try:
            self.memori = Memori(conn=self.session)
        except Exception as e:
            print(f"Debug: Memori initialization error: {e}")
            raise
        
        # Create a single OpenAI client that Memori wraps so that calls are tracked
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.memori.openai.register(self.openai_client)
        
        # Set attribution (entity_id = agent, process_id = namespace)
        self.memori.attribution(entity_id=self.agent_id, process_id=self.namespace)
        
        # Build the storage schema
        # This might be where Memori tries to connect - wrap in try/except
        try:
            self.memori.config.storage.build()
        except Exception as e:
            # If build fails, it might be because schema already exists or connection issue
            print(f"Debug: Storage build warning: {e}")
            # Try to continue anyway - schema might already exist
        
        # Cache numeric IDs so we can read/write memories directly
        self.entity_id = self._get_or_create_record_id("memori_entity", self.agent_id)
        self.process_record_id = self._get_or_create_record_id("memori_process", self.namespace)
    
    def get_relevant_memories(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories for a given query.
        Memori v3 automatically retrieves memories when using its OpenAI wrapper.
        This method is kept for compatibility but Memori handles retrieval automatically.
        
        Args:
            query: The user's message or query
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of relevant memory dictionaries (empty for v3, as it's handled automatically)
        """
        # Memori v3 handles memory retrieval automatically through its OpenAI integration
        # The memories are injected into the context automatically
        return []
    
    def store_memory(self, content: str, memory_type: str = "fact", metadata: Optional[Dict] = None):
        """
        Store a memory.
        In Memori v3, memories are automatically captured through OpenAI calls.
        This method is kept for compatibility but may not be needed.
        
        Args:
            content: The content to remember
            memory_type: Type of memory (fact, preference, rule, summary, etc.)
            metadata: Optional metadata to attach to the memory
        """
        # Memori v3 automatically captures conversations through its OpenAI integration
        # Manual storage may not be needed, but we keep this for explicit storage if needed
        pass
    
    def add_conversation(self, user_message: str, assistant_reply: str):
        """
        Add a conversation turn to memory.
        In Memori v3, conversations are automatically captured through OpenAI calls.
        This method is kept for compatibility.
        
        Args:
            user_message: The user's message
            assistant_reply: The assistant's reply
        """
        # Memori v3 automatically captures conversations when using its OpenAI wrapper
        # No manual storage needed
        pass
    
    def get_conversation_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recent conversation history.
        
        Args:
            limit: Maximum number of conversation turns to retrieve
            
        Returns:
            List of conversation dictionaries
        """
        try:
            # Try different methods based on what's available
            if hasattr(self.memori, 'retrieve_memories'):
                conversations = self.memori.retrieve_memories("conversation")
            elif hasattr(self.memori, 'search'):
                conversations = self.memori.search("conversation", limit=limit)
            else:
                conversations = []
            return conversations if conversations else []
        except Exception as e:
            print(f"Warning: Failed to retrieve conversation history: {e}")
            return []
    
    def build_memory_context(self, user_message: str) -> str:
        """
        Build a short context string with the most relevant stored facts.
        
        Args:
            user_message: The current user message
            
        Returns:
            Context string suitable for injecting into the LLM prompt
        """
        if not user_message or not self.entity_id:
            return ""
        
        try:
            facts = self._fetch_recent_facts(limit=25)
        except Exception as exc:
            print(f"Warning: Failed to fetch memories: {exc}")
            return ""
        
        if not facts:
            return ""
        
        keywords = {
            token for token in re.findall(r"\w+", user_message.lower())
            if len(token) > 2 and token not in _STOPWORDS
        }
        if not keywords:
            keywords = set(re.findall(r"\w+", user_message.lower()))
        
        scored: List[tuple[float, str]] = []
        for idx, fact in enumerate(facts):
            lowered = fact.lower()
            score = sum(1 for kw in keywords if kw in lowered)
            score += max(0.0, (len(facts) - idx) * 0.01)
            scored.append((score, fact))
        
        scored.sort(key=lambda item: item[0], reverse=True)
        relevant = [fact for score, fact in scored if score > 0][:5]
        if not relevant:
            relevant = facts[:3]
        
        if not relevant:
            return ""
        
        context_lines = ["Relevant memories:"]
        for fact in relevant:
            context_lines.append(f"- {fact}")
        
        context = "\n".join(context_lines)
        return context[:1500]
    
    def chat_completion(self, messages, model: str = "gpt-4o-mini", temperature: float = 0.3):
        """
        Make a chat completion using the registered OpenAI client.
        Memori automatically captures and retrieves memories through its registration.
        
        Args:
            messages: List of message dictionaries
            model: Model to use
            temperature: Temperature setting
            
        Returns:
            Response content string
        """
        try:
            # Ensure session is in a clean state before making the call
            # Rollback any pending transactions to avoid conflicts
            try:
                if self.session.is_active:
                    self.session.rollback()
            except Exception:
                # If rollback fails, the session might be in a bad state
                # Close it and create a new one
                try:
                    self.session.close()
                except Exception:
                    pass
                self.session = self.Session()
                # Re-initialize Memori with the new session
                self.memori = Memori(conn=self.session)
                self.memori.openai.register(self.openai_client)
                self.memori.attribution(entity_id=self.agent_id, process_id=self.namespace)
            
            # Use the registered OpenAI client directly
            # Memori automatically captures conversations through its registration
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            response = self._ensure_response_ready(response)
            
            # After the call, ensure session is clean for next operation
            try:
                if self.session.is_active:
                    self.session.commit()
            except Exception:
                # If commit fails, rollback to clean state
                try:
                    self.session.rollback()
                except Exception:
                    pass
            
            return response.choices[0].message.content
        except Exception as e:
            # If there's a session error, try to recover
            error_msg = str(e)
            if "commit" in error_msg.lower() or "transaction" in error_msg.lower():
                # Session transaction error - recover by creating new session
                try:
                    self.session.close()
                except Exception:
                    pass
                self.session = self.Session()
                self.memori = Memori(conn=self.session)
                self.memori.openai.register(self.openai_client)
                self.memori.attribution(entity_id=self.agent_id, process_id=self.namespace)
                
                # Retry the call
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature
                )
                response = self._ensure_response_ready(response)
                return response.choices[0].message.content
            else:
                # Re-raise if it's not a session error
                raise
    
    # ----- Internal helpers -----
    
    def _ensure_response_ready(self, response):
        """Make sure the response is resolved even if the client returned a coroutine."""
        if inspect.iscoroutine(response):
            return self._run_coroutine_sync(response)
        return response
    
    def _run_coroutine_sync(self, coro):
        """Run a coroutine to completion using a dedicated event loop."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def _fetch_recent_facts(self, limit: int = 20) -> List[str]:
        """
        Fetch recent Memori facts for the current entity.
        """
        if text is None or not self.entity_id:
            return []
        
        engine = self.session.bind
        if engine is None:
            return []
        
        stmt = text(
            "SELECT content FROM memori_entity_fact "
            "WHERE entity_id = :entity_id "
            "ORDER BY date_last_time DESC "
            "LIMIT :limit"
        )
        params = {"entity_id": self.entity_id, "limit": limit}
        try:
            with engine.begin() as conn:
                rows = conn.execute(stmt, params).fetchall()
        except Exception as exc:
            print(f"Warning: Failed to query stored facts: {exc}")
            return []
        
        return [row[0] for row in rows if row and row[0]]

    def _get_or_create_record_id(self, table: str, external_id: str) -> Optional[int]:
        """
        Resolve the numeric ID for Memori tables like memori_entity/process.
        """
        if text is None or not external_id:
            return None
        
        if table not in {"memori_entity", "memori_process"}:
            raise ValueError(f"Unsupported table lookup: {table}")
        
        engine = self.session.bind
        if engine is None:
            return None
        
        select_stmt = text(
            f"SELECT id FROM {table} WHERE external_id = :external_id"
        )
        insert_stmt = text(
            f"INSERT INTO {table} (uuid, external_id) VALUES (:uuid, :external_id)"
        )
        
        with engine.begin() as conn:
            row = conn.execute(select_stmt, {"external_id": external_id}).fetchone()
            if row:
                return row[0]
            conn.execute(insert_stmt, {"uuid": str(uuid4()), "external_id": external_id})
            row = conn.execute(select_stmt, {"external_id": external_id}).fetchone()
            return row[0] if row else None
    
    def cleanup(self):
        """Clean up resources if needed."""
        # Memori handles its own cleanup, but we can add custom cleanup here if needed
        pass


# Global memory managers cache (keyed by agent_id)
_memory_managers: Dict[str, MemoryManager] = {}


def get_memory_manager(agent_id: str, namespace: Optional[str] = None) -> MemoryManager:
    """
    Get or create a memory manager for an agent/namespace.
    
    Args:
        agent_id: Unique identifier for the agent (e.g., "main_assistant", "email_agent")
        namespace: Optional namespace (defaults to agent_id)
        
    Returns:
        MemoryManager instance for the agent
    """
    cache_key = f"{agent_id}:{namespace or agent_id}"
    
    if cache_key not in _memory_managers:
        _memory_managers[cache_key] = MemoryManager(agent_id, namespace)
    
    return _memory_managers[cache_key]

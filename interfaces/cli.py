import os
from rich.prompt import Prompt
from rich.console import Console
from core.assistant import handle_message

console = Console()

def run_cli():
    # Get agent ID from environment or use default
    # Since you're the only user, we use a single agent namespace
    agent_id = os.getenv("AGENT_ID", "main_assistant")
    
    console.print("[bold cyan]Mini-Me Assistant CLI[/bold cyan] (type 'exit' to quit)\n")
    console.print(f"[dim]Agent: {agent_id} (set AGENT_ID env var to change)[/dim]\n")

    while True:
        user_text = Prompt.ask("[bold green]You[/bold green]")
        if user_text.strip().lower() in ("exit", "quit"):
            break

        result = handle_message(user_text, agent_id=agent_id)
        console.print(f"[bold blue]Assistant:[/bold blue] {result.get('reply', '')}")

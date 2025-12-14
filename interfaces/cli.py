from rich.prompt import Prompt
from rich.console import Console
from core.assistant import handle_message

console = Console()

def run_cli():
    console.print("[bold cyan]Mini-Me Assistant CLI[/bold cyan] (type 'exit' to quit)\n")

    while True:
        user_text = Prompt.ask("[bold green]You[/bold green]")
        if user_text.strip().lower() in ("exit", "quit"):
            break

        result = handle_message(user_text)
        console.print(f"[bold blue]Assistant:[/bold blue] {result.get('reply', '')}")

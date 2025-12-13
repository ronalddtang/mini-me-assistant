from core.assistant import handle_message
from rich.prompt import Prompt
from rich.console import Console

console = Console()

def main():
    console.print("[bold cyan]Mini-Me Assistant CLI[/bold cyan]")
    while True:
        user_text = Prompt.ask("[bold green]You[/bold green]")
        if user_text.lower() in ["exit", "quit"]:
            break

        result = handle_message(user_text)
        console.print(f"[bold blue]Assistant:[/bold blue] {result['reply']}")
        console.print(f"[grey50]Intent: {result['intent']}[/grey50]")

if __name__ == "__main__":
    main()

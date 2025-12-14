import argparse

def main():
    parser = argparse.ArgumentParser(prog="mini-me-assistant")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("cli", help="Run the interactive CLI assistant")
    sub.add_parser("telegram", help="Run Telegram bot mode")

    args = parser.parse_args()

    if args.command == "cli":
        from interfaces.cli import run_cli
        run_cli()
    elif args.command == "telegram":
        from interfaces.telegram import run_telegram_bot
        run_telegram_bot()
    else:
        raise SystemExit(f"Unknown command: {args.command}")

if __name__ == "__main__":
    main()

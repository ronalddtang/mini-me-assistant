import os
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from core.assistant import handle_message

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

def _load_env():
    """Load environment variables from .env file if present."""
    if ENV_PATH.exists():
        load_dotenv(dotenv_path=ENV_PATH, override=True)
    else:
        load_dotenv(override=True)

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    await update.message.reply_text(
        "Hello! I'm your Mini-Me Assistant. Send me a message and I'll help you out!\n\n"
        "Use /help to see available commands."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text(
        "Available commands:\n"
        "/start - Start the bot\n"
        "/help - Show this help message\n\n"
        "Just send me a message and I'll respond as your assistant!"
    )


async def handle_user_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming text messages."""
    user_text = update.message.text
    
    if not user_text or not user_text.strip():
        await update.message.reply_text("Please send me a message with some text.")
        return
    
    # Use main assistant agent (since you're the only user)
    # For future multi-agent support, you could route based on command or context
    agent_id = "main_assistant"
    
    # Show typing indicator
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    try:
        # Process the message using the assistant with agent-specific memory.
        # Run in a thread executor so we don't block the event loop or mix sync/async calls.
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, handle_message, user_text, agent_id
        )
        reply = result.get('reply', '')
        
        if not reply:
            reply = "I received your message, but I don't have a response right now."
        
        # Send the reply
        await update.message.reply_text(reply)
        
        # Optionally log the intent and other metadata
        intent = result.get('intent', 'unknown')
        if intent in ['task', 'note']:
            logger.info(f"Processed {intent} from user {update.effective_user.id}")
    
    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        await update.message.reply_text(
            "Sorry, I encountered an error processing your message. Please try again."
        )


def run_telegram_bot():
    """Start the Telegram bot."""
    # Load environment variables from .env file
    _load_env()
    
    # Get bot token from environment
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        raise ValueError(
            "TELEGRAM_BOT_TOKEN is not set. "
            "Get a token from @BotFather on Telegram and add it to your environment or .env file."
        )
    
    # Create the Application
    application = Application.builder().token(bot_token).build()
    
    # Register handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_user_message))
    
    # Start the bot
    logger.info("Starting Telegram bot...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

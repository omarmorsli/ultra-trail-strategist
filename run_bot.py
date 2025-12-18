import os
import sys

from dotenv import load_dotenv

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from ultra_trail_strategist.api.telegram_bot import CrewBot


def main():
    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN")

    if not token:
        print("❌ Error: TELEGRAM_BOT_TOKEN not found in .env")
        print("1. Create a bot via @BotFather on Telegram.")
        print("2. Add TELEGRAM_BOT_TOKEN=your_token to .env")
        return

    print("🤖 Starting Crew Bot...")
    bot = CrewBot(token)
    bot.run()


if __name__ == "__main__":
    main()

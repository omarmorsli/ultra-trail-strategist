import logging
import os

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

from ultra_trail_strategist.state_manager import RaceStateManager

logger = logging.getLogger(__name__)


class CrewBot:
    def __init__(self, token: str):
        self.token = token
        self.state_manager = RaceStateManager()  # Shared access to live_race_state.json

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.effective_chat:
            return
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="👋 Hello Crew! I'm the Ultra-Trail Strategist Bot.\n\n"
            "Commands:\n"
            "/status - Get current race status and ETA for next checkpoint.\n"
            "/eta - Same as status.",
        )

    async def status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            state = self.state_manager.get_state()
            current_idx = state.current_segment_index
            splits = state.actual_splits

            if not splits and current_idx == 0:
                msg = "🏁 Race hasn't started yet (or no check-ins)."
            else:
                last_idx = max(splits.keys()) if splits else -1
                last_time = splits.get(last_idx, 0)
                msg = (
                    f"🏃 **Race Status Update**\n"
                    f"Last Check-in at Segment {last_idx + 1}: {last_time:.0f} mins elapsed.\n"
                    f"Current Segment Index: {current_idx}\n"
                    # f"Next ETA: [Need Full Plan Access]"  <- TODO: Persist plan summary in state?
                )

            if update.effective_chat:
                await context.bot.send_message(chat_id=update.effective_chat.id, text=msg)

        except Exception as e:
            logger.error(f"Bot Error: {e}")
            if update.effective_chat:
                await context.bot.send_message(
                    chat_id=update.effective_chat.id, text="⚠️ System Error reading race state."
                )

    def run(self):
        if not self.token:
            logger.error("No Telegram Token provided.")
            return

        application = ApplicationBuilder().token(self.token).build()

        start_handler = CommandHandler("start", self.start)
        status_handler = CommandHandler("status", self.status)
        eta_handler = CommandHandler("eta", self.status)

        application.add_handler(start_handler)
        application.add_handler(status_handler)
        application.add_handler(eta_handler)

        logger.info("🤖 Bot is polling...")
        application.run_polling()


if __name__ == "__main__":
    # For testing direct run
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if token:
        bot = CrewBot(token)
        bot.run()
    else:
        print("Please set TELEGRAM_BOT_TOKEN env var.")

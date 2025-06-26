import os
import sqlite3
import datetime
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from langchain_openai import ChatOpenAI 
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory 
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key and Telegram Bot Token as environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
DB_PATH = 'mental_health_bot.db'

# Initialize LangChain Chat LLM
# Using ChatOpenAI for better integration with chat history and messaging
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.7)

# Per-user chat memory
user_histories = {}

# Function to get or create a user's message history
def get_user_history(user_id):
    """
    Retrieves or creates a ChatMessageHistory instance for a given user ID.
    This ensures each user has their own persistent conversation history.
    """
    if user_id not in user_histories:
        user_histories[user_id] = ChatMessageHistory()
    return user_histories[user_id]

# Function to get a conversation runnable for a user
# The get_session_history parameter of RunnableWithMessageHistory expects a callable,
# not an instance of ChatMessageHistory. This callable should take a session_id
# and return the appropriate ChatMessageHistory object.
def get_conversation():
    """
    Returns a RunnableWithMessageHistory instance that wraps the LLM
    with a per-session chat history.
    """
    return RunnableWithMessageHistory(
        llm,
        lambda session_id: get_user_history(session_id)
    )

def init_db():
    """
    Initializes the SQLite database with tables for users, chat history, and mood tracking.
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        # User table to store Telegram user information
        c.execute('''CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            username TEXT,
            first_name TEXT,
            last_name TEXT
        )''')
        # Chat history table to store user messages and bot responses
        c.execute('''CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            message TEXT,
            response TEXT,
            timestamp TEXT,
            FOREIGN KEY(user_id) REFERENCES users(user_id)
        )''')
        # Mood tracking table to store user reported moods
        c.execute('''CREATE TABLE IF NOT EXISTS moods (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            mood TEXT,
            timestamp TEXT,
            FOREIGN KEY(user_id) REFERENCES users(user_id)
        )''')
        conn.commit()
    except sqlite3.Error as e:
        print(f"Database initialization error: {e}")
    finally:
        if conn:
            conn.close()

def save_user(user):
    """
    Saves or updates user information in the database.
    Uses INSERT OR IGNORE to avoid duplicating existing users.
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''INSERT OR IGNORE INTO users (user_id, username, first_name, last_name) VALUES (?, ?, ?, ?)''',
                  (user.id, user.username, user.first_name, user.last_name))
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error saving user: {e}")
    finally:
        if conn:
            conn.close()

def save_chat(user_id, message, response):
    """
    Saves a user's message and the bot's response to the chat history table.
    Includes a timestamp for record-keeping.
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''INSERT INTO chat_history (user_id, message, response, timestamp) VALUES (?, ?, ?, ?)''',
                  (user_id, message, response, datetime.datetime.now().isoformat()))
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error saving chat history: {e}")
    finally:
        if conn:
            conn.close()

def save_mood(user_id, mood):
    """
    Saves a user's reported mood to the moods table.
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''INSERT INTO moods (user_id, mood, timestamp) VALUES (?, ?, ?)''',
                  (user_id, mood, datetime.datetime.now().isoformat()))
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error saving mood: {e}")
    finally:
        if conn:
            conn.close()

def get_last_chats(user_id, limit=5):
    """
    Retrieves the last 'limit' number of chat entries for a given user.
    Returns them in chronological order.
    """
    conn = None
    rows = []
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''SELECT message, response, timestamp FROM chat_history WHERE user_id=? ORDER BY id DESC LIMIT ?''', (user_id, limit))
        rows = c.fetchall()
    except sqlite3.Error as e:
        print(f"Error fetching last chats: {e}")
    finally:
        if conn:
            conn.close()
    return rows[::-1]  # reverse to show oldest first

def get_user_stats(user_id):
    """
    Calculates and returns statistics about a user's chat activity,
    including total messages, days active, and average messages per day.
    """
    conn = None
    total_msgs = 0
    days_active = 0
    avg_per_day = 0.0
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''SELECT COUNT(*) FROM chat_history WHERE user_id=?''', (user_id,))
        total_msgs = c.fetchone()[0]
        c.execute('''SELECT MIN(timestamp), MAX(timestamp) FROM chat_history WHERE user_id=?''', (user_id,))
        minmax = c.fetchone()
        
        if minmax[0] and minmax[1]:
            d1 = datetime.datetime.fromisoformat(minmax[0])
            d2 = datetime.datetime.fromisoformat(minmax[1])
            days_active = max((d2 - d1).days + 1, 1) # Ensure at least 1 day if there are messages
            avg_per_day = total_msgs / days_active if days_active > 0 else 0.0
    except sqlite3.Error as e:
        print(f"Error fetching user stats: {e}")
    finally:
        if conn:
            conn.close()
    return total_msgs, days_active, avg_per_day

def get_mood_stats(user_id):
    """
    Retrieves mood frequency and recent mood entries for a given user.
    """
    conn = None
    mood_counts = []
    recent = []
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''SELECT mood, COUNT(*) FROM moods WHERE user_id=? GROUP BY mood''', (user_id,))
        mood_counts = c.fetchall()
        c.execute('''SELECT mood, timestamp FROM moods WHERE user_id=? ORDER BY id DESC LIMIT 5''', (user_id,))
        recent = c.fetchall()
    except sqlite3.Error as e:
        print(f"Error fetching mood stats: {e}")
    finally:
        if conn:
            conn.close()
    return mood_counts, recent[::-1] # Reverse recent to show oldest first

def start_message():
    """
    Returns the introductory message for the bot.
    """
    return ("Hello! I'm MindFriend, your mental health companion 🤗\n"
            "I'm here to listen and chat with you about anything on your mind. "
            "You can talk to me about your feelings, worries, or anything else.\n"
            "How are you feeling today?")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handler for the /start command.
    Saves user info and sends the welcome message.
    """
    user = update.effective_user
    save_user(user)
    await update.message.reply_text(start_message())

async def history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handler for the /history command.
    Retrieves and displays the last 5 chat interactions for the user.
    """
    user = update.effective_user
    save_user(user)
    chats = get_last_chats(user.id, 5)
    if not chats:
        await update.message.reply_text("No conversation history found.")
        return
    msg = "Here are your last 5 conversations:\n"
    for i, (user_msg, bot_resp, ts) in enumerate(chats, 1):
        msg += f"\n{i}. You: {user_msg}\n  MindFriend: {bot_resp}\n  Time: {ts[:19]}"
    await update.message.reply_text(msg)

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handler for the /stats command.
    Displays overall chat statistics for the user.
    """
    user = update.effective_user
    save_user(user)
    total_msgs, days_active, avg_per_day = get_user_stats(user.id)
    msg = (f"📊 Your Stats:\n\n"
           f"Total messages: {total_msgs}\n"
           f"Days active: {days_active}\n"
           f"Average messages per day: {avg_per_day:.2f}")
    await update.message.reply_text(msg)

async def mood(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handler for the /mood command.
    Allows users to record their current mood.
    Example usage: /mood happy
    """
    user = update.effective_user
    save_user(user)
    if context.args:
        mood_value = ' '.join(context.args).strip()
        if mood_value: # Ensure mood_value is not empty after stripping
            save_mood(user.id, mood_value)
            await update.message.reply_text(f"Mood '{mood_value}' recorded. Thank you for sharing!")
        else:
            await update.message.reply_text("It seems you didn't provide a mood. Please specify your mood after the command, e.g., /mood happy")
    else:
        await update.message.reply_text("Please provide your mood after the command, e.g., /mood happy")

async def moodstats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handler for the /moodstats command.
    Displays the user's mood frequency and recent mood entries.
    """
    user = update.effective_user
    save_user(user)
    mood_counts, recent = get_mood_stats(user.id)
    if not mood_counts and not recent:
        await update.message.reply_text("No mood records found. Use /mood <your mood> to log one!")
        return
    
    msg = "📝 Your Mood Stats:\n\n"
    if recent:
        msg += "Recent moods:\n"
        for mood_entry, ts in recent:
            msg += f"- {mood_entry} at {ts[:19]}\n"
    else:
        msg += "No recent moods recorded.\n"

    if mood_counts:
        msg += "\nMood frequency:\n"
        for mood_entry, count in mood_counts:
            msg += f"- {mood_entry}: {count}\n"
    else:
        msg += "No mood frequency data available.\n"

    await update.message.reply_text(msg)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Main message handler for all text messages that are not commands.
    Uses LangChain to generate a response and saves the interaction.
    """
    user = update.effective_user
    user_message = update.message.text
    save_user(user)

    # Use LangChain to generate a supportive response with per-user memory
    conversation = get_conversation()
    
    try:
        llm_response = conversation.invoke(
            {"input": f"You are a supportive, empathetic and funny friend. Respond kindly to: {user_message}"},
            config={"configurable": {"session_id": user.id}} # Pass user.id as session_id for history tracking
        )
        
        # Extract the content from the AIMessage object
        if hasattr(llm_response, 'content'):
            response_text = llm_response.content
        else:
            response_text = str(llm_response) # Fallback in case response object structure changes
    except Exception as e:
        response_text = "I'm sorry, I couldn't process that right now. Please try again later."
        print(f"Error generating LLM response: {e}")
    
    save_chat(user.id, user_message, response_text)
    await update.message.reply_text(response_text)

# Main function to run the bot
def main():
    """
    Main function to initialize the database, build the Telegram application,
    add command and message handlers, and start polling for updates.
    """
    init_db()
    
    # Ensure TELEGRAM_BOT_TOKEN is set
    if not TELEGRAM_BOT_TOKEN:
        print("Error: TELEGRAM_BOT_TOKEN environment variable not set.")
        print("Please set it in your .env file or environment variables.")
        return

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Register command handlers
    app.add_handler(CommandHandler('start', start))
    app.add_handler(CommandHandler('history', history))
    app.add_handler(CommandHandler('stats', stats))
    app.add_handler(CommandHandler('mood', mood))
    app.add_handler(CommandHandler('moodstats', moodstats))
    
    # Register general message handler for non-command text messages
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    print("Bot is running...")
    # Start the bot. This method will block until the bot is stopped.
    app.run_polling()

if __name__ == '__main__':
    main()
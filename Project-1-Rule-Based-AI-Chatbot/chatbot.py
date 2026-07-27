
import string
from datetime import datetime



# ---------------------------------------------------
# Rule-Based AI Chatbot
# DecodeLabs - Artificial Intelligence Project 1
# ---------------------------------------------------

BOT_NAME = "LogicBot"


# Commands that will close the chatbot
EXIT_COMMANDS = {
    "bye",
    "goodbye",
    "exit",
    "quit",
    "stop"
}


# The chatbot's knowledge base
# Key = expected user message
# Value = chatbot response
RESPONSES = {
    # Greeting intent
    "hello": "Hello! How can I help you?",
    "hi": "Hi! How can I assist you today?",
    "hey": "Hey! What would you like to know?",

    # Well-being intent
    "how are you": "I am working perfectly. Thank you for asking!",
    "how are you doing": "I am doing great. How can I help you?",

    # Identity intent
    "what is your name": f"My name is {BOT_NAME}.",
    "who are you": f"I am {BOT_NAME}, a rule-based AI chatbot.",

    # Capability intent
    "what can you do": (
        "I can answer predefined questions about AI, Python, "
        "and this chatbot project."
    ),
    "help": (
        "You can greet me or ask: 'What is AI?', "
        "'What is Python?', or 'What can you do?'"
    ),

    # AI intent
    "what is ai": (
        "Artificial Intelligence allows machines to perform tasks "
        "that normally require human intelligence."
    ),
    "define artificial intelligence": (
        "Artificial Intelligence is the simulation of human "
        "intelligence by computer systems."
    ),

    # Python intent
    "what is python": (
        "Python is a popular programming language used for "
        "AI, data science, automation, and web development."
    ),

    # Gratitude intent
    "thank you": "You are welcome!",
    "thanks": "You're welcome! I am happy to help."
}


# Default answer for an unknown message
FALLBACK_RESPONSE = (
    "Sorry, I do not understand that yet. "
    "Type 'help' to see what you can ask."
)


def sanitize_input(raw_input: str) -> str:
    """
    Convert input to lowercase, remove punctuation,
    and remove unnecessary spaces.
    """

    clean_input = raw_input.lower().strip()

    clean_input = clean_input.translate(
        str.maketrans("", "", string.punctuation)
    )

    return " ".join(clean_input.split())


def get_response(user_input: str) -> str:
    """
    Return a response using exact matching first,
    followed by simple keyword matching.
    """

    # First check exact dictionary matches
    exact_response = RESPONSES.get(user_input)

    if exact_response:
        return exact_response

    words = set(user_input.split())

    # Greeting intent
    if words.intersection({"hello", "hi", "hey"}):
        return "Hello! How can I help you?"

    # AI intent
    if "ai" in words or "artificial intelligence" in user_input:
        return (
            "Artificial Intelligence allows machines to perform "
            "tasks that normally require human intelligence."
        )

    # Python intent
    if "python" in words:
        return (
            "Python is a popular programming language used for "
            "AI, data science, automation, and web development."
        )

    # Name intent
    if "name" in words:
        return f"My name is {BOT_NAME}."

    # Capability/help intent
    if words.intersection({"help", "do", "capabilities"}):
        if "help" in words or "what can you do" in user_input:
            return (
                "I can answer basic questions about AI, Python, "
                "and this chatbot project."
            )

    # Gratitude intent
    if words.intersection({"thanks", "thank"}):
        return "You are welcome!"

    return FALLBACK_RESPONSE


def save_message(sender: str, message: str) -> None:
    """Save a chatbot message to the conversation history file."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        with open("chat_history.txt", "a", encoding="utf-8") as file:
            file.write(f"[{timestamp}] {sender}: {message}\n")

    except OSError as error:
        print(f"Unable to save chat history: {error}")

def run_chatbot() -> None:
    """Start and continuously run the chatbot."""

    welcome_message = "Hello! I am your rule-based AI chatbot."

    print("=" * 50)
    print(f"{BOT_NAME}: {welcome_message}")
    print(f"{BOT_NAME}: Type 'help' for guidance or 'exit' to stop.")
    print("=" * 50)

    save_message(BOT_NAME, welcome_message)

    while True:
        raw_input = input("\nYou: ")
        user_input = sanitize_input(raw_input)

        if not user_input:
            empty_response = "Please enter a message."
            print(f"{BOT_NAME}: {empty_response}")
            save_message(BOT_NAME, empty_response)
            continue

        save_message("User", raw_input.strip())

        if user_input in EXIT_COMMANDS:
            goodbye_message = "Goodbye! Have a great day."
            print(f"{BOT_NAME}: {goodbye_message}")
            save_message(BOT_NAME, goodbye_message)
            break

        response = get_response(user_input)

        print(f"{BOT_NAME}: {response}")
        save_message(BOT_NAME, response)


# Run the program only when this file is executed directly
if __name__ == "__main__":
    run_chatbot()

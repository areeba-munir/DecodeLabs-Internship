# Rule-Based AI Chatbot

## Project Overview

This project was developed as Project 1 of the DecodeLabs Artificial Intelligence Internship 2026.

It is a Python-based rule-driven chatbot that responds to predefined user inputs using dictionaries, control flow, keyword matching, and basic text processing.

## Features

- Handles greetings and basic questions
- Answers predefined questions about AI and Python
- Converts user input to lowercase
- Removes punctuation and unnecessary spaces
- Supports exact and keyword-based matching
- Provides a fallback response for unknown questions
- Runs continuously until an exit command is entered
- Supports commands such as `exit`, `quit`, `bye`, and `stop`

## Technologies Used

- Python
- Dictionaries
- Functions
- Conditional statements
- While loops
- String processing
- File handling

## How It Works

1. The user enters a message.
2. The chatbot cleans and normalises the input.
3. It first checks for an exact predefined response.
4. It then checks for matching keywords.
5. If no rule matches, it returns a fallback response.
6. The conversation continues until the user enters an exit command.

## Example Conversation

```text
LogicBot: Hello! I am your rule-based AI chatbot.
LogicBot: Type 'help' for guidance or 'exit' to stop.

You: Hello
LogicBot: Hello! How can I help you?

You: Can you explain AI?
LogicBot: Artificial Intelligence allows machines to perform tasks that normally require human intelligence.

You: Tell me about football
LogicBot: Sorry, I do not understand that yet.

You: Exit
LogicBot: Goodbye! Have a great day.
"""
Configuration file for Google Gemini Flash 2.5 AI Chatbot.
Store your API credentials and model settings here.
"""
import os

# Google Gemini API Configuration
# Set your API key as an environment variable or replace the placeholder below
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyAvvb-7rrxDmyPystaOkZ0UGB-HYilNvMc")
MODEL_NAME = "gemini-2.5-flash"

# Application Settings
MAX_HISTORY_LENGTH = 50  # Maximum conversation turns to keep
TEMPERATURE = 0.7  # AI creativity level (0.0 - 1.0)
MAX_OUTPUT_TOKENS = 2048  # Maximum response length

# Sentiment Analysis Settings
SENTIMENT_CATEGORIES = ["positive", "negative", "neutral"]
EMOTION_CATEGORIES = [
    "happy", "sad", "angry", "confused", "excited", 
    "anxious", "surprised", "neutral", "frustrated", "hopeful"
]

# Analytics Settings
TREND_WINDOW_SIZE = 5  # Number of messages to consider for trend analysis

"""
Utility Module - Helper functions for the AI Chatbot.
Provides formatting, validation, and common utilities.
"""

import json
import re
from typing import Dict, Any, List, Optional
from datetime import datetime


def format_timestamp(dt: datetime = None) -> str:
    """
    Format a datetime object to a readable string.
    
    Args:
        dt: Datetime object (defaults to now)
        
    Returns:
        Formatted timestamp string
    """
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def clean_text(text: str) -> str:
    """
    Clean and normalize text input.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = " ".join(text.split())
    # Remove control characters
    text = "".join(char for char in text if ord(char) >= 32 or char in "\n\t")
    return text.strip()


def validate_message(message: str, min_length: int = 1, max_length: int = 10000) -> tuple:
    """
    Validate a user message.
    
    Args:
        message: Message to validate
        min_length: Minimum allowed length
        max_length: Maximum allowed length
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not message:
        return False, "Message cannot be empty."
    
    cleaned = clean_text(message)
    
    if len(cleaned) < min_length:
        return False, f"Message must be at least {min_length} character(s)."
    
    if len(cleaned) > max_length:
        return False, f"Message cannot exceed {max_length} characters."
    
    return True, None


def format_sentiment_badge(sentiment: str, confidence: float = None) -> str:
    """
    Create a formatted badge for sentiment display.
    
    Args:
        sentiment: Sentiment category
        confidence: Optional confidence score
        
    Returns:
        Formatted badge string
    """
    badges = {
        "positive": "✅ Positive",
        "negative": "❌ Negative",
        "neutral": "➖ Neutral"
    }
    
    badge = badges.get(sentiment.lower(), f"❓ {sentiment}")
    
    if confidence is not None:
        badge += f" ({confidence:.0%})"
    
    return badge


def format_emotion_badge(emotion: str, intensity: str = None) -> str:
    """
    Create a formatted badge for emotion display.
    
    Args:
        emotion: Emotion category
        intensity: Optional intensity level
        
    Returns:
        Formatted badge string
    """
    icons = {
        "happy": "😊",
        "sad": "😢",
        "angry": "😠",
        "confused": "😕",
        "excited": "🎉",
        "anxious": "😰",
        "surprised": "😮",
        "neutral": "😐",
        "frustrated": "😤",
        "hopeful": "🌟"
    }
    
    icon = icons.get(emotion.lower(), "❓")
    badge = f"{icon} {emotion.capitalize()}"
    
    if intensity:
        badge += f" ({intensity})"
    
    return badge


def create_progress_bar(value: float, width: int = 20, 
                        fill_char: str = "█", empty_char: str = "░") -> str:
    """
    Create an ASCII progress bar.
    
    Args:
        value: Value between 0 and 1
        width: Width of the bar
        fill_char: Character for filled portion
        empty_char: Character for empty portion
        
    Returns:
        Progress bar string
    """
    value = max(0, min(1, value))  # Clamp to 0-1
    filled = int(value * width)
    empty = width - filled
    
    return f"[{fill_char * filled}{empty_char * empty}] {value:.0%}"


def format_conversation_for_display(history: List[Dict]) -> str:
    """
    Format conversation history for display.
    
    Args:
        history: List of message dictionaries
        
    Returns:
        Formatted conversation string
    """
    lines = []
    
    for msg in history:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        timestamp = msg.get("timestamp", "")
        
        if role == "SYSTEM":
            continue  # Skip system messages
        
        # Format role indicator
        if role == "USER":
            prefix = "👤 You"
        elif role == "ASSISTANT":
            prefix = "🤖 AI"
        else:
            prefix = f"❓ {role}"
        
        # Add sentiment info if available
        sentiment = msg.get("sentiment")
        if sentiment:
            sentiment_str = format_sentiment_badge(
                sentiment.get("sentiment", ""),
                sentiment.get("confidence")
            )
            emotion_str = format_emotion_badge(
                sentiment.get("emotion", ""),
                sentiment.get("emotion_intensity")
            )
            lines.append(f"{prefix}: {content}")
            lines.append(f"   {sentiment_str} | {emotion_str}")
        else:
            lines.append(f"{prefix}: {content}")
        
        lines.append("")
    
    return "\n".join(lines)


def extract_json_from_text(text: str) -> Optional[Dict]:
    """
    Extract JSON object from text that may contain other content.
    
    Args:
        text: Text that may contain JSON
        
    Returns:
        Extracted dictionary or None
    """
    # Try to find JSON block
    text = text.strip()
    
    # Remove markdown code blocks
    if "```json" in text:
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
    elif "```" in text:
        match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
    
    # Find JSON boundaries
    start = text.find('{')
    end = text.rfind('}')
    
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    
    return None


def safe_get(dictionary: Dict, *keys, default=None):
    """
    Safely get a nested value from a dictionary.
    
    Args:
        dictionary: The dictionary to search
        *keys: Keys to traverse
        default: Default value if not found
        
    Returns:
        The value or default
    """
    result = dictionary
    
    for key in keys:
        if isinstance(result, dict):
            result = result.get(key, default)
        else:
            return default
    
    return result


def merge_dicts(*dicts: Dict) -> Dict:
    """
    Merge multiple dictionaries.
    
    Args:
        *dicts: Dictionaries to merge
        
    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result


class ConversationLogger:
    """
    Logger for conversation history.
    Can save and load conversations from files.
    """
    
    def __init__(self, log_dir: str = "./logs"):
        """Initialize logger with log directory."""
        self.log_dir = log_dir
    
    def save_conversation(self, history: List[Dict], filename: str = None) -> str:
        """
        Save conversation to a JSON file.
        
        Args:
            history: Conversation history
            filename: Optional filename
            
        Returns:
            Path to saved file
        """
        import os
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.json"
        
        filepath = os.path.join(self.log_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": format_timestamp(),
                "messages": history
            }, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def load_conversation(self, filename: str) -> List[Dict]:
        """
        Load conversation from a JSON file.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            Conversation history
        """
        import os
        
        filepath = os.path.join(self.log_dir, filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data.get("messages", [])
    
    def list_conversations(self) -> List[str]:
        """List all saved conversations."""
        import os
        
        if not os.path.exists(self.log_dir):
            return []
        
        return [f for f in os.listdir(self.log_dir) if f.endswith('.json')]


def print_colored(text: str, color: str = "white"):
    """
    Print colored text to console (if supported).
    Falls back to regular print if colors not supported.
    
    Args:
        text: Text to print
        color: Color name
    """
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "purple": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m"
    }
    
    try:
        color_code = colors.get(color.lower(), colors["white"])
        print(f"{color_code}{text}{colors['reset']}")
    except:
        print(text)


def print_banner():
    """Print application banner."""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║     🤖 AI CHATBOT - Powered by Google Gemini Flash 2.5    ║
║                                                           ║
║     Features:                                             ║
║     • Intelligent Conversations                           ║
║     • Real-time Sentiment Analysis                        ║
║     • Emotion Detection                                   ║
║     • Conversation Analytics                              ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_help():
    """Print help information."""
    help_text = """
Available Commands:
─────────────────────────────────────────────────────
  /help      - Show this help message
  /stats     - Show conversation statistics
  /summary   - Generate conversation summary
  /keywords  - Extract keywords from conversation
  /trends    - Show sentiment trend analysis
  /graph     - Display ASCII mood graph
  /profile   - Show emotion profile
  /report    - Generate full analytics report
  /reset     - Start a new conversation
  /save      - Save conversation to file
  /exit      - Exit the chatbot
─────────────────────────────────────────────────────
Just type your message to chat with the AI!
"""
    print(help_text)

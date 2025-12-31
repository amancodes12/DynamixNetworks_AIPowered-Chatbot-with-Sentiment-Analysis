"""
Unit tests for Chatbot Module.
Tests conversation management, history tracking, and AI-driven responses.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatbot import (
    Chatbot, SmartChatbot, Message, ConversationRole, 
    ConversationState
)


class TestMessage:
    """Test suite for Message data class."""
    
    def test_creation(self):
        """Test message creation."""
        msg = Message(
            role=ConversationRole.USER,
            content="Hello, AI!"
        )
        
        assert msg.role == ConversationRole.USER
        assert msg.content == "Hello, AI!"
        assert msg.timestamp is not None
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        msg = Message(
            role=ConversationRole.ASSISTANT,
            content="Hello! How can I help?"
        )
        
        d = msg.to_dict()
        
        assert d["role"] == "assistant"
        assert d["content"] == "Hello! How can I help?"
        assert "timestamp" in d


class TestConversationState:
    """Test suite for ConversationState class."""
    
    def test_initial_state(self):
        """Test initial state values."""
        state = ConversationState()
        
        assert state.mood == "neutral"
        assert state.engagement_level == "normal"
        assert state.sentiment_trend == "stable"
    
    def test_update_from_sentiment(self):
        """Test state update from sentiment result."""
        from sentiment import SentimentResult
        
        state = ConversationState()
        
        sentiment = SentimentResult(
            message="I'm so excited!",
            sentiment="positive",
            confidence=0.9,
            emotion="excited",
            emotion_intensity="high",
            reasoning="User is very enthusiastic"
        )
        
        state.update(sentiment)
        
        assert state.mood == "positive"
        assert state.last_emotion == "excited"
        assert state.engagement_level == "high"
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        state = ConversationState()
        d = state.to_dict()
        
        assert "mood" in d
        assert "engagement_level" in d
        assert "sentiment_trend" in d


class TestChatbot:
    """Test suite for Chatbot class."""
    
    @pytest.fixture
    def mock_ai_client(self):
        """Create a mock AI client."""
        client = Mock()
        client.analyze_sentiment.return_value = {
            "sentiment": "neutral",
            "confidence": 0.7,
            "emotion": "neutral",
            "emotion_intensity": "medium",
            "reasoning": "Standard message"
        }
        client.generate_reply.return_value = "Hello! How can I assist you today?"
        client.summarize_conversation.return_value = {
            "summary": "Test conversation",
            "key_points": ["test"],
            "overall_tone": "neutral"
        }
        client.extract_keywords.return_value = {
            "keywords": ["test"],
            "themes": ["testing"]
        }
        return client
    
    @pytest.fixture
    def chatbot(self, mock_ai_client):
        """Create a chatbot with mocked AI client."""
        return Chatbot(ai_client=mock_ai_client)
    
    def test_initialization(self, chatbot):
        """Test chatbot initialization."""
        assert chatbot.history is not None
        assert len(chatbot.history) == 1  # System message
        assert chatbot.history[0].role == ConversationRole.SYSTEM
    
    def test_chat_response(self, chatbot):
        """Test basic chat functionality."""
        result = chatbot.chat("Hello!")
        
        assert "response" in result
        assert "sentiment" in result
        assert "state" in result
        assert result["response"] == "Hello! How can I assist you today?"
    
    def test_chat_updates_history(self, chatbot):
        """Test that chat updates conversation history."""
        initial_length = len(chatbot.history)
        
        chatbot.chat("Test message")
        
        # Should add user message and assistant response
        assert len(chatbot.history) == initial_length + 2
    
    def test_chat_analyzes_sentiment(self, chatbot, mock_ai_client):
        """Test that chat triggers sentiment analysis."""
        chatbot.chat("I'm feeling happy today!")
        
        mock_ai_client.analyze_sentiment.assert_called()
    
    def test_get_history(self, chatbot):
        """Test getting conversation history."""
        chatbot.chat("Test message")
        
        history = chatbot.get_history()
        
        assert isinstance(history, list)
        assert len(history) > 0
        assert all(isinstance(h, dict) for h in history)
    
    def test_get_user_messages(self, chatbot):
        """Test getting only user messages."""
        chatbot.chat("Message 1")
        chatbot.chat("Message 2")
        
        user_messages = chatbot.get_user_messages()
        
        assert len(user_messages) == 2
        assert "Message 1" in user_messages
        assert "Message 2" in user_messages
    
    def test_get_sentiment_history(self, chatbot):
        """Test getting sentiment analysis history."""
        chatbot.chat("Test 1")
        chatbot.chat("Test 2")
        
        sentiment_history = chatbot.get_sentiment_history()
        
        assert len(sentiment_history) == 2
    
    def test_get_conversation_summary(self, chatbot, mock_ai_client):
        """Test getting AI-generated summary."""
        chatbot.chat("Test message")
        
        summary = chatbot.get_conversation_summary()
        
        assert "summary" in summary
        mock_ai_client.summarize_conversation.assert_called()
    
    def test_get_keywords(self, chatbot, mock_ai_client):
        """Test getting AI-extracted keywords."""
        chatbot.chat("Test message")
        
        keywords = chatbot.get_keywords()
        
        assert "keywords" in keywords
        mock_ai_client.extract_keywords.assert_called()
    
    def test_get_statistics(self, chatbot):
        """Test getting conversation statistics."""
        chatbot.chat("Test 1")
        chatbot.chat("Test 2")
        
        stats = chatbot.get_statistics()
        
        assert stats["user_messages"] == 2
        assert stats["assistant_messages"] == 2
        assert "sentiment_stats" in stats
    
    def test_reset(self, chatbot):
        """Test conversation reset."""
        chatbot.chat("Test message")
        initial_history_length = len(chatbot.history)
        
        chatbot.reset()
        
        # Should only have system message
        assert len(chatbot.history) == 1
        assert chatbot.history[0].role == ConversationRole.SYSTEM
    
    def test_set_personality(self, chatbot):
        """Test setting chatbot personality."""
        new_personality = "You are a pirate who speaks in pirate language."
        
        chatbot.set_personality(new_personality)
        
        assert chatbot.system_prompt == new_personality
        assert chatbot.history[0].content == new_personality


class TestSmartChatbot:
    """Test suite for SmartChatbot class (enhanced features)."""
    
    @pytest.fixture
    def mock_ai_client(self):
        """Create a mock AI client."""
        client = Mock()
        client.analyze_sentiment.return_value = {
            "sentiment": "positive",
            "confidence": 0.8,
            "emotion": "happy",
            "emotion_intensity": "high",
            "reasoning": "User is happy"
        }
        client.generate_reply.return_value = "That's great to hear!"
        return client
    
    @pytest.fixture
    def smart_chatbot(self, mock_ai_client):
        """Create a SmartChatbot with mocked AI client."""
        return SmartChatbot(ai_client=mock_ai_client)
    
    def test_mood_shift_detection_improving(self, smart_chatbot, mock_ai_client):
        """Test detection of improving mood."""
        # First message - negative
        mock_ai_client.analyze_sentiment.return_value = {
            "sentiment": "negative",
            "confidence": 0.7,
            "emotion": "sad",
            "emotion_intensity": "medium",
            "reasoning": "User seems sad"
        }
        smart_chatbot.chat("I'm feeling down")
        
        # Second message - positive
        mock_ai_client.analyze_sentiment.return_value = {
            "sentiment": "positive",
            "confidence": 0.8,
            "emotion": "happy",
            "emotion_intensity": "high",
            "reasoning": "User is happy now"
        }
        result = smart_chatbot.chat("Actually, I feel better now!")
        
        assert result["mood_shift_detected"] is not None
        assert result["mood_shift_detected"]["direction"] == "improving"
    
    def test_mood_shift_detection_declining(self, smart_chatbot, mock_ai_client):
        """Test detection of declining mood."""
        # First message - positive
        mock_ai_client.analyze_sentiment.return_value = {
            "sentiment": "positive",
            "confidence": 0.8,
            "emotion": "happy",
            "emotion_intensity": "high",
            "reasoning": "User is happy"
        }
        smart_chatbot.chat("I'm having a great day!")
        
        # Second message - negative
        mock_ai_client.analyze_sentiment.return_value = {
            "sentiment": "negative",
            "confidence": 0.7,
            "emotion": "sad",
            "emotion_intensity": "medium",
            "reasoning": "User is sad now"
        }
        result = smart_chatbot.chat("But something bad just happened...")
        
        assert result["mood_shift_detected"] is not None
        assert result["mood_shift_detected"]["direction"] == "declining"
    
    def test_ui_hints_generation(self, smart_chatbot, mock_ai_client):
        """Test UI hints are generated."""
        mock_ai_client.analyze_sentiment.return_value = {
            "sentiment": "positive",
            "confidence": 0.9,
            "emotion": "excited",
            "emotion_intensity": "high",
            "reasoning": "User is excited"
        }
        
        result = smart_chatbot.chat("This is amazing!")
        
        assert "ui_hints" in result
        assert "suggested_color" in result["ui_hints"]
        assert "emotion_icon" in result["ui_hints"]
        assert "intensity_level" in result["ui_hints"]
    
    def test_ui_hints_colors(self, smart_chatbot, mock_ai_client):
        """Test UI hint colors for different emotions."""
        emotions_and_expected = [
            ("happy", "#4CAF50"),
            ("sad", "#2196F3"),
            ("angry", "#F44336"),
            ("excited", "#FF9800")
        ]
        
        for emotion, expected_color in emotions_and_expected:
            mock_ai_client.analyze_sentiment.return_value = {
                "sentiment": "positive",
                "confidence": 0.8,
                "emotion": emotion,
                "emotion_intensity": "high",
                "reasoning": f"User is {emotion}"
            }
            
            result = smart_chatbot.chat(f"Test {emotion}")
            
            assert result["ui_hints"]["suggested_color"] == expected_color


class TestHistoryManagement:
    """Test suite for conversation history management."""
    
    @pytest.fixture
    def mock_ai_client(self):
        """Create a mock AI client."""
        client = Mock()
        client.analyze_sentiment.return_value = {
            "sentiment": "neutral",
            "confidence": 0.5,
            "emotion": "neutral",
            "emotion_intensity": "low",
            "reasoning": "test"
        }
        client.generate_reply.return_value = "Test response"
        return client
    
    @pytest.fixture
    def chatbot(self, mock_ai_client):
        """Create a chatbot with mocked AI client."""
        return Chatbot(ai_client=mock_ai_client)
    
    def test_history_order(self, chatbot):
        """Test that history maintains correct order."""
        chatbot.chat("First message")
        chatbot.chat("Second message")
        chatbot.chat("Third message")
        
        user_messages = chatbot.get_user_messages()
        
        assert user_messages[0] == "First message"
        assert user_messages[1] == "Second message"
        assert user_messages[2] == "Third message"
    
    def test_history_contains_both_roles(self, chatbot):
        """Test that history contains both user and assistant messages."""
        chatbot.chat("Test message")
        
        history = chatbot.get_history()
        
        roles = [h["role"] for h in history]
        
        assert "user" in roles
        assert "assistant" in roles
    
    def test_history_preserves_sentiment(self, chatbot):
        """Test that history preserves sentiment analysis."""
        chatbot.chat("I'm happy!")
        
        history = chatbot.get_history()
        
        # Find user message
        user_msg = next(h for h in history if h["role"] == "user")
        
        assert user_msg["sentiment"] is not None
        assert "sentiment" in user_msg["sentiment"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

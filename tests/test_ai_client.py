"""
Unit tests for AI Client Module.
Tests the GeminiAIClient wrapper class with mocked API responses.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_client import GeminiAIClient, AIClientError


class MockResponse:
    """Mock response object for Gemini API."""
    def __init__(self, text):
        self.text = text


class TestGeminiAIClient:
    """Test suite for GeminiAIClient class."""
    
    @pytest.fixture
    def mock_client(self):
        """Create a client with mocked API."""
        with patch('ai_client.genai.Client') as mock_genai:
            client = GeminiAIClient(api_key="test_key", model_name="test_model")
            client.client = mock_genai.return_value
            return client
    
    def test_initialization(self):
        """Test client initialization with custom parameters."""
        with patch('ai_client.genai.Client'):
            client = GeminiAIClient(api_key="custom_key", model_name="custom_model")
            assert client.api_key == "custom_key"
            assert client.model_name == "custom_model"
    
    def test_generate_content_success(self, mock_client):
        """Test successful content generation."""
        expected_response = "Hello! How can I help you today?"
        mock_client.client.models.generate_content.return_value = MockResponse(expected_response)
        
        result = mock_client._generate_content("Hello")
        
        assert result == expected_response
        mock_client.client.models.generate_content.assert_called_once()
    
    def test_generate_content_error(self, mock_client):
        """Test content generation error handling."""
        mock_client.client.models.generate_content.side_effect = Exception("API Error")
        
        with pytest.raises(AIClientError) as excinfo:
            mock_client._generate_content("Hello")
        
        assert "Failed to generate content" in str(excinfo.value)
    
    def test_generate_reply(self, mock_client):
        """Test generate_reply method."""
        expected_response = "I'd be happy to help you with that!"
        mock_client.client.models.generate_content.return_value = MockResponse(expected_response)
        
        result = mock_client.generate_reply(
            user_message="Can you help me?",
            conversation_history=[{"role": "user", "content": "Hello"}],
            current_mood="positive",
            sentiment_context="User seems happy"
        )
        
        assert result == expected_response
    
    def test_analyze_sentiment_valid_json(self, mock_client):
        """Test sentiment analysis with valid JSON response."""
        json_response = json.dumps({
            "sentiment": "positive",
            "confidence": 0.85,
            "emotion": "happy",
            "emotion_intensity": "high",
            "reasoning": "User expressed gratitude"
        })
        mock_client.client.models.generate_content.return_value = MockResponse(json_response)
        
        result = mock_client.analyze_sentiment("Thank you so much!")
        
        assert result["sentiment"] == "positive"
        assert result["confidence"] == 0.85
        assert result["emotion"] == "happy"
        assert result["emotion_intensity"] == "high"
    
    def test_analyze_sentiment_invalid_json(self, mock_client):
        """Test sentiment analysis with invalid JSON response."""
        mock_client.client.models.generate_content.return_value = MockResponse("Not valid JSON")
        
        result = mock_client.analyze_sentiment("Hello")
        
        # Should return defaults
        assert result["sentiment"] == "neutral"
        assert result["confidence"] == 0.5
        assert result["emotion"] == "neutral"
    
    def test_summarize_conversation(self, mock_client):
        """Test conversation summarization."""
        json_response = json.dumps({
            "summary": "User asked about weather",
            "key_points": ["Weather inquiry", "Local area"],
            "overall_tone": "curious",
            "insights": "User is planning outdoor activities"
        })
        mock_client.client.models.generate_content.return_value = MockResponse(json_response)
        
        history = [
            {"role": "user", "content": "What's the weather like?"},
            {"role": "assistant", "content": "It looks sunny today!"}
        ]
        
        result = mock_client.summarize_conversation(history)
        
        assert "summary" in result
        assert "key_points" in result
    
    def test_summarize_conversation_empty_history(self, mock_client):
        """Test summarization with empty history."""
        result = mock_client.summarize_conversation([])
        
        assert result["summary"] == "No conversation to summarize."
    
    def test_extract_keywords(self, mock_client):
        """Test keyword extraction."""
        json_response = json.dumps({
            "keywords": ["python", "programming", "help"],
            "themes": ["coding", "learning"],
            "frequency_analysis": "User focused on Python programming"
        })
        mock_client.client.models.generate_content.return_value = MockResponse(json_response)
        
        history = [{"role": "user", "content": "I need help with Python programming"}]
        
        result = mock_client.extract_keywords(history)
        
        assert "keywords" in result
        assert "python" in result["keywords"]
    
    def test_generate_trend_analysis(self, mock_client):
        """Test trend analysis generation."""
        json_response = json.dumps({
            "trend": "improving",
            "direction": "positive",
            "mood_shifts": ["Started neutral, became happy"],
            "analysis": "User mood improved over conversation"
        })
        mock_client.client.models.generate_content.return_value = MockResponse(json_response)
        
        sentiment_history = [
            {"sentiment": "neutral", "emotion": "neutral"},
            {"sentiment": "positive", "emotion": "happy"}
        ]
        
        result = mock_client.generate_trend_analysis(sentiment_history)
        
        assert result["trend"] == "improving"
        assert result["direction"] == "positive"
    
    def test_generate_ascii_mood_graph(self, mock_client):
        """Test ASCII mood graph generation."""
        expected_graph = """
        Positive  |  * * *
        Neutral   |* 
        Negative  |
                   1 2 3 4 5
        """
        mock_client.client.models.generate_content.return_value = MockResponse(expected_graph)
        
        sentiment_history = [{"sentiment": "neutral"}, {"sentiment": "positive"}]
        
        result = mock_client.generate_ascii_mood_graph(sentiment_history)
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_generate_emotion_profile(self, mock_client):
        """Test emotion profile generation."""
        profile = "User shows predominantly positive emotions with occasional neutral moments."
        mock_client.client.models.generate_content.return_value = MockResponse(profile)
        
        sentiment_history = [
            {"sentiment": "positive", "emotion": "happy"},
            {"sentiment": "neutral", "emotion": "neutral"}
        ]
        
        result = mock_client.generate_emotion_profile(sentiment_history)
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_extract_json_with_markdown(self, mock_client):
        """Test JSON extraction from markdown code blocks."""
        markdown_response = '```json\n{"key": "value"}\n```'
        
        result = mock_client._extract_json(markdown_response)
        
        assert result == '{"key": "value"}'
    
    def test_extract_json_plain(self, mock_client):
        """Test JSON extraction from plain text."""
        text_response = 'Here is the result: {"key": "value"} end.'
        
        result = mock_client._extract_json(text_response)
        
        assert result == '{"key": "value"}'


class TestPromptConstruction:
    """Test suite for prompt construction logic."""
    
    @pytest.fixture
    def client(self):
        """Create a client for testing."""
        with patch('ai_client.genai.Client'):
            return GeminiAIClient(api_key="test_key")
    
    def test_reply_prompt_contains_user_message(self, client):
        """Test that reply prompt contains the user message."""
        client.client.models.generate_content.return_value = MockResponse("Test")
        
        client.generate_reply(user_message="Test message")
        
        call_args = client.client.models.generate_content.call_args
        prompt = call_args[1]['contents']
        assert "Test message" in prompt
    
    def test_reply_prompt_contains_history(self, client):
        """Test that reply prompt contains conversation history."""
        client.client.models.generate_content.return_value = MockResponse("Test")
        
        history = [
            {"role": "user", "content": "Previous message"},
            {"role": "assistant", "content": "Previous response"}
        ]
        
        client.generate_reply(user_message="New message", conversation_history=history)
        
        call_args = client.client.models.generate_content.call_args
        prompt = call_args[1]['contents']
        assert "Previous message" in prompt
        assert "Previous response" in prompt
    
    def test_sentiment_prompt_structure(self, client):
        """Test sentiment analysis prompt structure."""
        client.client.models.generate_content.return_value = MockResponse('{"sentiment":"neutral"}')
        
        client.analyze_sentiment("Test message")
        
        call_args = client.client.models.generate_content.call_args
        prompt = call_args[1]['contents']
        
        # Check required fields are mentioned in prompt
        assert "sentiment" in prompt.lower()
        assert "emotion" in prompt.lower()
        assert "confidence" in prompt.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

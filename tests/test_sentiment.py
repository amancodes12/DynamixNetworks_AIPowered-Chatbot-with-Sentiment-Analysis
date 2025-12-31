"""
Unit tests for Sentiment Analysis Module.
Tests sentiment analysis functionality with mocked AI responses.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentiment import SentimentResult, SentimentAnalyzer, SentimentPipeline


class TestSentimentResult:
    """Test suite for SentimentResult data class."""
    
    def test_creation(self):
        """Test SentimentResult creation."""
        result = SentimentResult(
            message="Hello",
            sentiment="positive",
            confidence=0.9,
            emotion="happy",
            emotion_intensity="high",
            reasoning="User greeted positively"
        )
        
        assert result.message == "Hello"
        assert result.sentiment == "positive"
        assert result.confidence == 0.9
        assert result.emotion == "happy"
        assert result.emotion_intensity == "high"
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = SentimentResult(
            message="Test",
            sentiment="neutral",
            confidence=0.5,
            emotion="neutral",
            emotion_intensity="low",
            reasoning="Neutral message"
        )
        
        d = result.to_dict()
        
        assert d["message"] == "Test"
        assert d["sentiment"] == "neutral"
        assert "timestamp" in d
    
    def test_str_representation(self):
        """Test string representation."""
        result = SentimentResult(
            message="Test",
            sentiment="positive",
            confidence=0.85,
            emotion="happy",
            emotion_intensity="high",
            reasoning="Test reasoning"
        )
        
        s = str(result)
        
        assert "POSITIVE" in s
        assert "85%" in s
        assert "happy" in s


class TestSentimentAnalyzer:
    """Test suite for SentimentAnalyzer class."""
    
    @pytest.fixture
    def mock_ai_client(self):
        """Create a mock AI client."""
        client = Mock()
        client.analyze_sentiment.return_value = {
            "sentiment": "positive",
            "confidence": 0.8,
            "emotion": "happy",
            "emotion_intensity": "medium",
            "reasoning": "Positive message detected"
        }
        return client
    
    @pytest.fixture
    def analyzer(self, mock_ai_client):
        """Create an analyzer with mocked client."""
        return SentimentAnalyzer(ai_client=mock_ai_client)
    
    def test_analyze_single_message(self, analyzer):
        """Test analyzing a single message."""
        result = analyzer.analyze("I'm feeling great today!")
        
        assert isinstance(result, SentimentResult)
        assert result.sentiment == "positive"
        assert result.confidence == 0.8
    
    def test_analyze_stores_history(self, analyzer):
        """Test that analysis results are stored in history."""
        analyzer.analyze("Message 1")
        analyzer.analyze("Message 2")
        analyzer.analyze("Message 3")
        
        assert len(analyzer.history) == 3
    
    def test_analyze_batch(self, analyzer):
        """Test batch analysis."""
        messages = ["Hello", "How are you?", "Goodbye"]
        results = analyzer.analyze_batch(messages)
        
        assert len(results) == 3
        assert all(isinstance(r, SentimentResult) for r in results)
    
    def test_get_dominant_sentiment(self, analyzer, mock_ai_client):
        """Test getting dominant sentiment."""
        # Configure different sentiments
        mock_ai_client.analyze_sentiment.side_effect = [
            {"sentiment": "positive", "confidence": 0.8, "emotion": "happy", 
             "emotion_intensity": "medium", "reasoning": "test"},
            {"sentiment": "positive", "confidence": 0.7, "emotion": "happy",
             "emotion_intensity": "medium", "reasoning": "test"},
            {"sentiment": "negative", "confidence": 0.6, "emotion": "sad",
             "emotion_intensity": "low", "reasoning": "test"}
        ]
        
        analyzer.analyze("Test 1")
        analyzer.analyze("Test 2")
        analyzer.analyze("Test 3")
        
        dominant = analyzer.get_dominant_sentiment()
        assert dominant == "positive"
    
    def test_get_dominant_emotion(self, analyzer, mock_ai_client):
        """Test getting dominant emotion."""
        mock_ai_client.analyze_sentiment.side_effect = [
            {"sentiment": "positive", "confidence": 0.8, "emotion": "happy",
             "emotion_intensity": "high", "reasoning": "test"},
            {"sentiment": "positive", "confidence": 0.8, "emotion": "excited",
             "emotion_intensity": "high", "reasoning": "test"},
            {"sentiment": "positive", "confidence": 0.8, "emotion": "happy",
             "emotion_intensity": "medium", "reasoning": "test"}
        ]
        
        analyzer.analyze("Test 1")
        analyzer.analyze("Test 2")
        analyzer.analyze("Test 3")
        
        dominant = analyzer.get_dominant_emotion()
        assert dominant == "happy"
    
    def test_get_average_confidence(self, analyzer, mock_ai_client):
        """Test average confidence calculation."""
        mock_ai_client.analyze_sentiment.side_effect = [
            {"sentiment": "positive", "confidence": 0.8, "emotion": "happy",
             "emotion_intensity": "medium", "reasoning": "test"},
            {"sentiment": "positive", "confidence": 0.6, "emotion": "happy",
             "emotion_intensity": "medium", "reasoning": "test"}
        ]
        
        analyzer.analyze("Test 1")
        analyzer.analyze("Test 2")
        
        avg = analyzer.get_average_confidence()
        assert avg == 0.7
    
    def test_get_sentiment_distribution(self, analyzer, mock_ai_client):
        """Test sentiment distribution calculation."""
        mock_ai_client.analyze_sentiment.side_effect = [
            {"sentiment": "positive", "confidence": 0.8, "emotion": "happy",
             "emotion_intensity": "medium", "reasoning": "test"},
            {"sentiment": "negative", "confidence": 0.7, "emotion": "sad",
             "emotion_intensity": "medium", "reasoning": "test"},
            {"sentiment": "positive", "confidence": 0.9, "emotion": "excited",
             "emotion_intensity": "high", "reasoning": "test"}
        ]
        
        analyzer.analyze("Test 1")
        analyzer.analyze("Test 2")
        analyzer.analyze("Test 3")
        
        dist = analyzer.get_sentiment_distribution()
        
        assert dist["positive"] == 2
        assert dist["negative"] == 1
        assert dist["neutral"] == 0
    
    def test_get_recent_mood(self, analyzer, mock_ai_client):
        """Test getting recent mood."""
        mock_ai_client.analyze_sentiment.side_effect = [
            {"sentiment": "positive", "confidence": 0.8, "emotion": "happy",
             "emotion_intensity": "medium", "reasoning": "test"},
            {"sentiment": "negative", "confidence": 0.7, "emotion": "sad",
             "emotion_intensity": "medium", "reasoning": "test"},
            {"sentiment": "negative", "confidence": 0.6, "emotion": "sad",
             "emotion_intensity": "low", "reasoning": "test"},
            {"sentiment": "negative", "confidence": 0.8, "emotion": "angry",
             "emotion_intensity": "high", "reasoning": "test"}
        ]
        
        for i in range(4):
            analyzer.analyze(f"Test {i}")
        
        # Most recent 3 messages should be negative
        recent = analyzer.get_recent_mood(n=3)
        assert recent == "negative"
    
    def test_clear_history(self, analyzer):
        """Test clearing history."""
        analyzer.analyze("Test")
        assert len(analyzer.history) == 1
        
        analyzer.clear_history()
        assert len(analyzer.history) == 0
    
    def test_get_summary_stats(self, analyzer, mock_ai_client):
        """Test getting summary statistics."""
        mock_ai_client.analyze_sentiment.return_value = {
            "sentiment": "positive",
            "confidence": 0.8,
            "emotion": "happy",
            "emotion_intensity": "medium",
            "reasoning": "test"
        }
        
        analyzer.analyze("Test 1")
        analyzer.analyze("Test 2")
        
        stats = analyzer.get_summary_stats()
        
        assert stats["total_messages"] == 2
        assert "dominant_sentiment" in stats
        assert "dominant_emotion" in stats
        assert "sentiment_distribution" in stats


class TestSentimentPipeline:
    """Test suite for SentimentPipeline class."""
    
    @pytest.fixture
    def mock_ai_client(self):
        """Create a mock AI client."""
        client = Mock()
        client.analyze_sentiment.return_value = {
            "sentiment": "positive",
            "confidence": 0.85,
            "emotion": "happy",
            "emotion_intensity": "high",
            "reasoning": "User expressed joy"
        }
        return client
    
    @pytest.fixture
    def pipeline(self, mock_ai_client):
        """Create a pipeline with mocked client."""
        return SentimentPipeline(ai_client=mock_ai_client)
    
    def test_process_message(self, pipeline):
        """Test processing a message through the pipeline."""
        result = pipeline.process_message("I'm so happy!")
        
        assert "current_analysis" in result
        assert "stats" in result
        assert "mood_context" in result
    
    def test_get_full_analysis(self, pipeline):
        """Test getting full analysis."""
        pipeline.process_message("Test 1")
        pipeline.process_message("Test 2")
        
        analysis = pipeline.get_full_analysis()
        
        assert "history" in analysis
        assert "statistics" in analysis
        assert len(analysis["history"]) == 2
    
    def test_reset(self, pipeline):
        """Test pipeline reset."""
        pipeline.process_message("Test")
        pipeline.reset()
        
        analysis = pipeline.get_full_analysis()
        assert len(analysis["history"]) == 0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def mock_ai_client(self):
        """Create a mock AI client with default response."""
        client = Mock()
        client.analyze_sentiment.return_value = {
            "sentiment": "neutral",
            "confidence": 0.5,
            "emotion": "neutral",
            "emotion_intensity": "low",
            "reasoning": "Default response"
        }
        return client
    
    def test_empty_history_dominant_sentiment(self, mock_ai_client):
        """Test getting dominant sentiment with empty history."""
        analyzer = SentimentAnalyzer(ai_client=mock_ai_client)
        
        assert analyzer.get_dominant_sentiment() is None
    
    def test_empty_history_average_confidence(self, mock_ai_client):
        """Test average confidence with empty history."""
        analyzer = SentimentAnalyzer(ai_client=mock_ai_client)
        
        assert analyzer.get_average_confidence() == 0.0
    
    def test_single_message_recent_mood(self, mock_ai_client):
        """Test recent mood with single message."""
        analyzer = SentimentAnalyzer(ai_client=mock_ai_client)
        analyzer.analyze("Test")
        
        # Should handle gracefully
        mood = analyzer.get_recent_mood(n=5)
        assert mood == "neutral"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

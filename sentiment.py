"""
Sentiment Analysis Module - AI-Powered Sentiment & Emotion Classification.
All analysis is performed using Google Gemini Flash 2.5 API.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from ai_client import GeminiAIClient


@dataclass
class SentimentResult:
    """Data class for sentiment analysis results."""
    message: str
    sentiment: str  # positive, negative, neutral
    confidence: float
    emotion: str
    emotion_intensity: str
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message": self.message,
            "sentiment": self.sentiment,
            "confidence": self.confidence,
            "emotion": self.emotion,
            "emotion_intensity": self.emotion_intensity,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat()
        }
    
    def __str__(self) -> str:
        """Human-readable representation."""
        return (
            f"Sentiment: {self.sentiment.upper()} ({self.confidence:.0%} confident)\n"
            f"Emotion: {self.emotion} ({self.emotion_intensity} intensity)\n"
            f"Reasoning: {self.reasoning}"
        )


class SentimentAnalyzer:
    """
    AI-powered sentiment analyzer using Google Gemini Flash 2.5.
    Analyzes text for sentiment, emotion, and provides reasoning.
    """
    
    def __init__(self, ai_client: GeminiAIClient = None):
        """Initialize the sentiment analyzer."""
        self.ai_client = ai_client or GeminiAIClient()
        self.history: List[SentimentResult] = []
    
    def analyze(self, message: str) -> SentimentResult:
        """
        Analyze a single message for sentiment and emotion.
        
        Args:
            message: Text to analyze
            
        Returns:
            SentimentResult with full analysis
        """
        # Get AI analysis
        analysis = self.ai_client.analyze_sentiment(message)
        
        # Create result object
        result = SentimentResult(
            message=message,
            sentiment=analysis.get("sentiment", "neutral"),
            confidence=float(analysis.get("confidence", 0.5)),
            emotion=analysis.get("emotion", "neutral"),
            emotion_intensity=analysis.get("emotion_intensity", "medium"),
            reasoning=analysis.get("reasoning", "Analysis completed.")
        )
        
        # Store in history
        self.history.append(result)
        
        return result
    
    def analyze_batch(self, messages: List[str]) -> List[SentimentResult]:
        """
        Analyze multiple messages.
        
        Args:
            messages: List of texts to analyze
            
        Returns:
            List of SentimentResult objects
        """
        return [self.analyze(msg) for msg in messages]
    
    def get_history(self) -> List[SentimentResult]:
        """Get all sentiment analysis history."""
        return self.history
    
    def get_history_dicts(self) -> List[Dict[str, Any]]:
        """Get history as list of dictionaries."""
        return [result.to_dict() for result in self.history]
    
    def get_dominant_sentiment(self) -> Optional[str]:
        """
        Get the most common sentiment from history.
        
        Returns:
            Most frequent sentiment or None if no history
        """
        if not self.history:
            return None
        
        sentiment_counts = {}
        for result in self.history:
            sentiment_counts[result.sentiment] = sentiment_counts.get(result.sentiment, 0) + 1
        
        return max(sentiment_counts, key=sentiment_counts.get)
    
    def get_dominant_emotion(self) -> Optional[str]:
        """
        Get the most common emotion from history.
        
        Returns:
            Most frequent emotion or None if no history
        """
        if not self.history:
            return None
        
        emotion_counts = {}
        for result in self.history:
            emotion_counts[result.emotion] = emotion_counts.get(result.emotion, 0) + 1
        
        return max(emotion_counts, key=emotion_counts.get)
    
    def get_average_confidence(self) -> float:
        """Get average confidence score from history."""
        if not self.history:
            return 0.0
        
        return sum(r.confidence for r in self.history) / len(self.history)
    
    def get_sentiment_distribution(self) -> Dict[str, int]:
        """Get count of each sentiment type."""
        distribution = {"positive": 0, "negative": 0, "neutral": 0}
        
        for result in self.history:
            if result.sentiment in distribution:
                distribution[result.sentiment] += 1
        
        return distribution
    
    def get_emotion_distribution(self) -> Dict[str, int]:
        """Get count of each emotion type."""
        distribution = {}
        
        for result in self.history:
            distribution[result.emotion] = distribution.get(result.emotion, 0) + 1
        
        return distribution
    
    def get_recent_mood(self, n: int = 3) -> str:
        """
        Get the mood based on recent messages.
        
        Args:
            n: Number of recent messages to consider
            
        Returns:
            Dominant recent sentiment
        """
        if not self.history:
            return "neutral"
        
        recent = self.history[-n:]
        sentiment_counts = {}
        
        for result in recent:
            sentiment_counts[result.sentiment] = sentiment_counts.get(result.sentiment, 0) + 1
        
        return max(sentiment_counts, key=sentiment_counts.get)
    
    def clear_history(self):
        """Clear all sentiment history."""
        self.history = []
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get comprehensive summary statistics."""
        return {
            "total_messages": len(self.history),
            "dominant_sentiment": self.get_dominant_sentiment(),
            "dominant_emotion": self.get_dominant_emotion(),
            "average_confidence": self.get_average_confidence(),
            "sentiment_distribution": self.get_sentiment_distribution(),
            "emotion_distribution": self.get_emotion_distribution(),
            "recent_mood": self.get_recent_mood()
        }


class SentimentPipeline:
    """
    Complete sentiment analysis pipeline.
    Combines real-time analysis with trend tracking and insights.
    """
    
    def __init__(self, ai_client: GeminiAIClient = None):
        """Initialize the sentiment pipeline."""
        self.ai_client = ai_client or GeminiAIClient()
        self.analyzer = SentimentAnalyzer(self.ai_client)
    
    def process_message(self, message: str) -> Dict[str, Any]:
        """
        Process a message through the full sentiment pipeline.
        
        Args:
            message: Text to process
            
        Returns:
            Complete analysis results
        """
        # Analyze sentiment
        result = self.analyzer.analyze(message)
        
        # Get current stats
        stats = self.analyzer.get_summary_stats()
        
        return {
            "current_analysis": result.to_dict(),
            "stats": stats,
            "mood_context": self._generate_mood_context(result, stats)
        }
    
    def _generate_mood_context(self, result: SentimentResult, stats: Dict) -> str:
        """Generate a mood context string for chatbot use."""
        recent_mood = stats.get("recent_mood", "neutral")
        dominant = stats.get("dominant_sentiment", "neutral")
        
        context = f"Current: {result.sentiment} ({result.emotion})"
        
        if recent_mood != result.sentiment:
            context += f" | Shift detected from {recent_mood}"
        
        if stats.get("total_messages", 0) > 3:
            context += f" | Overall trend: {dominant}"
        
        return context
    
    def get_full_analysis(self) -> Dict[str, Any]:
        """Get complete analysis of all processed messages."""
        return {
            "history": self.analyzer.get_history_dicts(),
            "statistics": self.analyzer.get_summary_stats(),
            "distributions": {
                "sentiment": self.analyzer.get_sentiment_distribution(),
                "emotion": self.analyzer.get_emotion_distribution()
            }
        }
    
    def reset(self):
        """Reset the pipeline."""
        self.analyzer.clear_history()

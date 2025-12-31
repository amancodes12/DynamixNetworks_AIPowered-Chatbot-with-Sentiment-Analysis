"""
AI Client Module - Wrapper for Google Gemini Flash 2.5 API.
All AI operations are centralized here for clean architecture.
"""

import json
from typing import Dict, List, Any, Optional
from google import genai
from google.genai import types

from config import GEMINI_API_KEY, MODEL_NAME, TEMPERATURE, MAX_OUTPUT_TOKENS


class GeminiAIClient:
    """
    Wrapper class for Google Gemini Flash 2.5 API.
    Provides methods for chat, sentiment analysis, summarization, and analytics.
    """
    
    def __init__(self, api_key: str = None, model_name: str = None):
        """Initialize the Gemini AI client."""
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model_name or MODEL_NAME
        
        # Initialize the client
        self.client = genai.Client(api_key=self.api_key)
        
    def _generate_content(self, prompt: str, temperature: float = None) -> str:
        """
        Core method to generate content from Gemini API.
        
        Args:
            prompt: The input prompt for the AI
            temperature: Creativity level (0.0 - 1.0)
            
        Returns:
            Generated text response
        """
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature or TEMPERATURE,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                )
            )
            return response.text
        except Exception as e:
            raise AIClientError(f"Failed to generate content: {str(e)}")
    
    def generate_reply(self, user_message: str, conversation_history: List[Dict] = None,
                       current_mood: str = None, sentiment_context: str = None) -> str:
        """
        Generate an AI-powered chatbot reply.
        
        Args:
            user_message: The user's input message
            conversation_history: List of previous messages
            current_mood: Current conversation mood
            sentiment_context: Recent sentiment analysis results
            
        Returns:
            AI-generated response
        """
        history_text = ""
        if conversation_history:
            history_text = "\n".join([
                f"{msg['role'].upper()}: {msg['content']}" 
                for msg in conversation_history[-10:]  # Last 10 messages for context
            ])
        
        prompt = f"""You are an intelligent, empathetic AI assistant. Your responses should be:
- Natural and conversational
- Contextually aware of the conversation history
- Emotionally intelligent based on the user's mood
- Helpful and engaging

CONVERSATION HISTORY:
{history_text if history_text else "No previous conversation."}

CURRENT MOOD CONTEXT: {current_mood if current_mood else "Not determined yet."}
RECENT SENTIMENT: {sentiment_context if sentiment_context else "Not analyzed yet."}

USER MESSAGE: {user_message}

Generate a thoughtful, contextually appropriate response. If the user seems upset, be more supportive. 
If they're happy, share in their enthusiasm. Adapt your tone based on the emotional context.

YOUR RESPONSE:"""

        return self._generate_content(prompt)
    
    def analyze_sentiment(self, message: str) -> Dict[str, Any]:
        """
        Analyze sentiment and emotion of a message using AI.
        
        Args:
            message: Text to analyze
            
        Returns:
            Dictionary with sentiment, emotion, confidence, and reasoning
        """
        prompt = f"""Analyze the following message for sentiment and emotion.

MESSAGE: "{message}"

Provide your analysis in the following JSON format (respond ONLY with valid JSON):
{{
    "sentiment": "positive|negative|neutral",
    "confidence": 0.0-1.0,
    "emotion": "happy|sad|angry|confused|excited|anxious|surprised|neutral|frustrated|hopeful",
    "emotion_intensity": "low|medium|high",
    "reasoning": "Brief explanation of why you classified it this way"
}}

ANALYSIS:"""

        response = self._generate_content(prompt, temperature=0.3)
        
        try:
            # Extract JSON from response
            json_str = self._extract_json(response)
            result = json.loads(json_str)
            
            # Validate and set defaults
            result.setdefault("sentiment", "neutral")
            result.setdefault("confidence", 0.5)
            result.setdefault("emotion", "neutral")
            result.setdefault("emotion_intensity", "medium")
            result.setdefault("reasoning", "Analysis completed.")
            
            return result
        except json.JSONDecodeError:
            return {
                "sentiment": "neutral",
                "confidence": 0.5,
                "emotion": "neutral",
                "emotion_intensity": "medium",
                "reasoning": "Unable to parse AI response, defaulting to neutral."
            }
    
    def summarize_conversation(self, conversation_history: List[Dict]) -> Dict[str, Any]:
        """
        Generate an AI-powered summary of the entire conversation.
        
        Args:
            conversation_history: Full conversation history
            
        Returns:
            Dictionary with summary, key points, and insights
        """
        if not conversation_history:
            return {
                "summary": "No conversation to summarize.",
                "key_points": [],
                "overall_tone": "neutral",
                "insights": "No insights available."
            }
        
        history_text = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}" 
            for msg in conversation_history
        ])
        
        prompt = f"""Analyze and summarize the following conversation comprehensively.

CONVERSATION:
{history_text}

Provide your analysis in the following JSON format (respond ONLY with valid JSON):
{{
    "summary": "A comprehensive 2-3 sentence summary of what was discussed",
    "key_points": ["List", "of", "main", "topics", "discussed"],
    "overall_tone": "The dominant emotional tone of the conversation",
    "user_mood_journey": "How the user's mood evolved during the conversation",
    "insights": "Any notable patterns or insights about the conversation",
    "recommendation": "Suggestion for how the conversation could continue"
}}

ANALYSIS:"""

        response = self._generate_content(prompt, temperature=0.4)
        
        try:
            json_str = self._extract_json(response)
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {
                "summary": response,
                "key_points": [],
                "overall_tone": "neutral",
                "insights": "Analysis completed."
            }
    
    def extract_keywords(self, conversation_history: List[Dict]) -> Dict[str, Any]:
        """
        Extract keywords and themes from conversation using AI.
        
        Args:
            conversation_history: Full conversation history
            
        Returns:
            Dictionary with keywords, themes, and frequency insights
        """
        if not conversation_history:
            return {
                "keywords": [],
                "themes": [],
                "frequency_analysis": "No data to analyze."
            }
        
        history_text = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}" 
            for msg in conversation_history
        ])
        
        prompt = f"""Extract key information from the following conversation.

CONVERSATION:
{history_text}

Provide your analysis in the following JSON format (respond ONLY with valid JSON):
{{
    "keywords": ["list", "of", "important", "keywords"],
    "themes": ["main", "themes", "discussed"],
    "entities": ["people", "places", "things", "mentioned"],
    "questions_asked": ["questions", "the", "user", "asked"],
    "topics_of_interest": ["topics", "user", "seemed", "interested", "in"],
    "frequency_analysis": "Brief analysis of recurring terms or ideas"
}}

ANALYSIS:"""

        response = self._generate_content(prompt, temperature=0.3)
        
        try:
            json_str = self._extract_json(response)
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {
                "keywords": [],
                "themes": [],
                "frequency_analysis": response
            }
    
    def generate_trend_analysis(self, sentiment_history: List[Dict]) -> Dict[str, Any]:
        """
        Analyze sentiment trends over the conversation.
        
        Args:
            sentiment_history: List of sentiment analysis results
            
        Returns:
            Dictionary with trend analysis and insights
        """
        if not sentiment_history:
            return {
                "trend": "stable",
                "direction": "neutral",
                "analysis": "No sentiment data to analyze."
            }
        
        sentiment_text = "\n".join([
            f"Message {i+1}: Sentiment={s.get('sentiment', 'N/A')}, "
            f"Emotion={s.get('emotion', 'N/A')}, "
            f"Intensity={s.get('emotion_intensity', 'N/A')}"
            for i, s in enumerate(sentiment_history)
        ])
        
        prompt = f"""Analyze the sentiment trend from the following conversation data.

SENTIMENT HISTORY:
{sentiment_text}

Provide your analysis in the following JSON format (respond ONLY with valid JSON):
{{
    "trend": "improving|declining|stable|volatile",
    "direction": "positive|negative|neutral",
    "mood_shifts": ["List of significant mood changes"],
    "emotional_peaks": ["Notable emotional high/low points"],
    "analysis": "Detailed analysis of how the user's emotions evolved",
    "prediction": "Likely emotional direction if conversation continues"
}}

ANALYSIS:"""

        response = self._generate_content(prompt, temperature=0.4)
        
        try:
            json_str = self._extract_json(response)
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {
                "trend": "stable",
                "direction": "neutral",
                "analysis": response
            }
    
    def generate_ascii_mood_graph(self, sentiment_history: List[Dict]) -> str:
        """
        Generate an AI-created ASCII art mood graph.
        
        Args:
            sentiment_history: List of sentiment analysis results
            
        Returns:
            ASCII art representation of mood over time
        """
        if not sentiment_history:
            return "📊 No conversation data yet.\n\nStart chatting to see your mood graph!"
        
        # Build detailed sentiment data
        sentiment_text = "\n".join([
            f"Message {i+1}: {s.get('sentiment', 'neutral')} ({s.get('emotion', 'neutral')}, confidence: {s.get('confidence', 0.5):.0%})"
            for i, s in enumerate(sentiment_history)
        ])
        
        # Count sentiments for context
        pos_count = sum(1 for s in sentiment_history if s.get('sentiment') == 'positive')
        neg_count = sum(1 for s in sentiment_history if s.get('sentiment') == 'negative')
        neu_count = sum(1 for s in sentiment_history if s.get('sentiment') == 'neutral')
        
        prompt = f"""Create an ASCII art mood graph for this conversation data.

SENTIMENT DATA ({len(sentiment_history)} messages):
{sentiment_text}

SUMMARY: {pos_count} positive, {neu_count} neutral, {neg_count} negative

Create an ASCII visualization with these requirements:
1. Show a graph with messages on X-axis (1 to {len(sentiment_history)})
2. Y-axis shows mood level: Positive (top), Neutral (middle), Negative (bottom)
3. Plot each message's sentiment as a point or bar
4. Use these characters: █ ▓ ░ │ ─ ┼ ● ○ ▲ ▼
5. Add a simple legend at the bottom
6. Keep it clean and readable (max 55 chars wide)

Example format:
Positive  │ ●     ●   ●
Neutral   │   ●         ●
Negative  │     ●
          └─────────────────
            1 2 3 4 5 6 7

Now create the graph for the actual data:"""

        try:
            result = self._generate_content(prompt, temperature=0.5)
            return result if result else self._fallback_mood_graph(sentiment_history)
        except Exception as e:
            return self._fallback_mood_graph(sentiment_history)
    
    def _fallback_mood_graph(self, sentiment_history: List[Dict]) -> str:
        """Generate a simple fallback mood graph if AI fails."""
        if not sentiment_history:
            return "No data available."
        
        lines = []
        lines.append("📊 Mood Graph")
        lines.append("=" * 50)
        lines.append("")
        
        # Simple text-based visualization
        sentiment_map = {'positive': '▲ Positive', 'neutral': '● Neutral', 'negative': '▼ Negative'}
        
        for i, s in enumerate(sentiment_history):
            sentiment = s.get('sentiment', 'neutral')
            emotion = s.get('emotion', 'neutral')
            symbol = {'positive': '😊', 'neutral': '😐', 'negative': '😢'}.get(sentiment, '😐')
            bar_char = {'positive': '█', 'neutral': '▓', 'negative': '░'}.get(sentiment, '▓')
            bar_len = {'positive': 20, 'neutral': 12, 'negative': 5}.get(sentiment, 10)
            
            lines.append(f"Msg {i+1}: {symbol} {bar_char * bar_len} ({sentiment}, {emotion})")
        
        lines.append("")
        lines.append("Legend: 😊 Positive | 😐 Neutral | 😢 Negative")
        
        return "\n".join(lines)
    
    def generate_emotion_profile(self, sentiment_history: List[Dict]) -> str:
        """
        Generate an AI-created emotion profile summary.
        
        Args:
            sentiment_history: List of sentiment analysis results
            
        Returns:
            Detailed emotion profile as text
        """
        if not sentiment_history:
            return "No data available for emotion profile."
        
        emotions = [s.get('emotion', 'neutral') for s in sentiment_history]
        sentiments = [s.get('sentiment', 'neutral') for s in sentiment_history]
        
        prompt = f"""Create a detailed emotion profile based on this conversation data.

EMOTIONS DETECTED: {emotions}
SENTIMENTS: {sentiments}
TOTAL MESSAGES: {len(sentiment_history)}

Generate a comprehensive emotion profile that includes:
1. Dominant emotion and its frequency
2. Emotional range (variety of emotions shown)
3. Emotional stability assessment
4. Personality insights based on emotional patterns
5. Communication style observations
6. Recommendations for engagement

Make it insightful and professional.

EMOTION PROFILE:"""

        return self._generate_content(prompt, temperature=0.6)
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from a text response."""
        # Try to find JSON block
        text = text.strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        # Find JSON boundaries
        start = text.find('{')
        end = text.rfind('}') + 1
        
        if start != -1 and end > start:
            return text[start:end]
        
        return text


class AIClientError(Exception):
    """Custom exception for AI Client errors."""
    pass

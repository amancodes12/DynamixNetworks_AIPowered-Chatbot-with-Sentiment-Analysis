"""
AI Chatbot Application - Main Entry Point.
A production-grade AI chatbot powered by Google Gemini Flash 2.5.
"""

import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chatbot import SmartChatbot
from analytics import ConversationAnalytics, ReportGenerator
from utils import (
    print_banner, print_help, print_colored, 
    format_sentiment_badge, format_emotion_badge,
    format_conversation_for_display, ConversationLogger,
    validate_message, clean_text
)
from config import GEMINI_API_KEY


class ChatbotApplication:
    """
    Main application class for the AI Chatbot.
    Handles user interaction, command processing, and analytics display.
    """
    
    def __init__(self):
        """Initialize the application."""
        self.chatbot = None
        self.analytics = None
        self.report_generator = None
        self.logger = ConversationLogger()
        self.running = False
    
    def initialize(self):
        """Initialize all components."""
        # Check API key
        if GEMINI_API_KEY == "<YOUR_GEMINI_API_KEY>":
            print_colored("\n⚠️  WARNING: Please set your Gemini API key in config.py", "yellow")
            print_colored("   Get your API key from: https://aistudio.google.com/app/apikey\n", "yellow")
            
            # Ask for API key
            api_key = input("Enter your Gemini API key (or press Enter to exit): ").strip()
            if not api_key:
                print_colored("Exiting...", "red")
                return False
            
            # Update config temporarily
            from ai_client import GeminiAIClient
            self.chatbot = SmartChatbot(ai_client=GeminiAIClient(api_key=api_key))
        else:
            self.chatbot = SmartChatbot()
        
        self.analytics = ConversationAnalytics(self.chatbot.ai_client)
        self.report_generator = ReportGenerator(self.analytics)
        
        return True
    
    def run(self):
        """Run the main application loop."""
        print_banner()
        
        if not self.initialize():
            return
        
        print_colored("Type /help to see available commands", "cyan")
        print_colored("Start chatting by typing your message!\n", "green")
        
        self.running = True
        
        while self.running:
            try:
                # Get user input
                user_input = input("👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Check for commands
                if user_input.startswith("/"):
                    self._handle_command(user_input)
                    continue
                
                # Validate message
                is_valid, error = validate_message(user_input)
                if not is_valid:
                    print_colored(f"❌ {error}", "red")
                    continue
                
                # Process message
                self._process_message(user_input)
                
            except KeyboardInterrupt:
                print("\n")
                self._handle_command("/exit")
            except Exception as e:
                print_colored(f"\n❌ Error: {str(e)}", "red")
                print_colored("Try again or type /help for commands.\n", "yellow")
    
    def _process_message(self, message: str):
        """Process a user message and display the response."""
        print()  # Empty line for spacing
        
        try:
            # Get chatbot response
            result = self.chatbot.chat(message)
            
            # Display AI response
            print_colored(f"🤖 AI: {result['response']}\n", "white")
            
            # Display sentiment info
            sentiment = result.get("sentiment", {})
            sentiment_badge = format_sentiment_badge(
                sentiment.get("sentiment", "neutral"),
                sentiment.get("confidence", 0.5)
            )
            emotion_badge = format_emotion_badge(
                sentiment.get("emotion", "neutral"),
                sentiment.get("emotion_intensity", "medium")
            )
            
            print_colored(f"   {sentiment_badge} | {emotion_badge}", "cyan")
            
            # Display mood shift if detected
            mood_shift = result.get("mood_shift_detected")
            if mood_shift:
                direction = mood_shift.get("direction", "")
                if direction == "improving":
                    print_colored(f"   📈 Mood improving: {mood_shift.get('from')} → {mood_shift.get('to')}", "green")
                elif direction == "declining":
                    print_colored(f"   📉 Mood declining: {mood_shift.get('from')} → {mood_shift.get('to')}", "red")
            
            print()  # Empty line after response
            
        except Exception as e:
            print_colored(f"❌ Error generating response: {str(e)}", "red")
            print_colored("Please check your API key and try again.\n", "yellow")
    
    def _handle_command(self, command: str):
        """Handle special commands."""
        cmd = command.lower().split()[0]
        
        if cmd == "/help":
            print_help()
        
        elif cmd == "/exit" or cmd == "/quit":
            self._exit()
        
        elif cmd == "/stats":
            self._show_stats()
        
        elif cmd == "/summary":
            self._show_summary()
        
        elif cmd == "/keywords":
            self._show_keywords()
        
        elif cmd == "/trends":
            self._show_trends()
        
        elif cmd == "/graph":
            self._show_graph()
        
        elif cmd == "/profile":
            self._show_profile()
        
        elif cmd == "/report":
            self._show_full_report()
        
        elif cmd == "/reset":
            self._reset_conversation()
        
        elif cmd == "/save":
            self._save_conversation()
        
        elif cmd == "/history":
            self._show_history()
        
        else:
            print_colored(f"Unknown command: {command}", "red")
            print_colored("Type /help to see available commands.\n", "yellow")
    
    def _show_stats(self):
        """Display conversation statistics."""
        stats = self.chatbot.get_statistics()
        
        print("\n" + "=" * 50)
        print("📊 CONVERSATION STATISTICS")
        print("=" * 50)
        
        print(f"\nMessages:")
        print(f"  Total: {stats.get('total_messages', 0)}")
        print(f"  User: {stats.get('user_messages', 0)}")
        print(f"  AI: {stats.get('assistant_messages', 0)}")
        
        sentiment_stats = stats.get("sentiment_stats", {})
        print(f"\nSentiment Analysis:")
        print(f"  Dominant: {sentiment_stats.get('dominant_sentiment', 'N/A')}")
        print(f"  Dominant Emotion: {sentiment_stats.get('dominant_emotion', 'N/A')}")
        print(f"  Avg Confidence: {sentiment_stats.get('average_confidence', 0):.1%}")
        
        dist = sentiment_stats.get("sentiment_distribution", {})
        if dist:
            print(f"\n  Distribution:")
            for sentiment, count in dist.items():
                print(f"    {sentiment.capitalize()}: {count}")
        
        state = stats.get("conversation_state", {})
        print(f"\nCurrent State:")
        print(f"  Mood: {state.get('mood', 'N/A')}")
        print(f"  Engagement: {state.get('engagement_level', 'N/A')}")
        
        print("=" * 50 + "\n")
    
    def _show_summary(self):
        """Display AI-generated conversation summary."""
        print("\n⏳ Generating conversation summary...")
        
        try:
            summary = self.chatbot.get_conversation_summary()
            
            print("\n" + "=" * 50)
            print("📝 CONVERSATION SUMMARY")
            print("=" * 50)
            
            print(f"\nSummary: {summary.get('summary', 'N/A')}")
            print(f"\nOverall Tone: {summary.get('overall_tone', 'N/A')}")
            print(f"\nMood Journey: {summary.get('user_mood_journey', 'N/A')}")
            
            key_points = summary.get("key_points", [])
            if key_points:
                print("\nKey Points:")
                for point in key_points:
                    print(f"  • {point}")
            
            print(f"\nInsights: {summary.get('insights', 'N/A')}")
            print(f"\nRecommendation: {summary.get('recommendation', 'N/A')}")
            
            print("=" * 50 + "\n")
            
        except Exception as e:
            print_colored(f"❌ Error generating summary: {str(e)}\n", "red")
    
    def _show_keywords(self):
        """Display AI-extracted keywords."""
        print("\n⏳ Extracting keywords...")
        
        try:
            keywords = self.chatbot.get_keywords()
            
            print("\n" + "=" * 50)
            print("🔑 KEYWORDS & THEMES")
            print("=" * 50)
            
            kw_list = keywords.get("keywords", [])
            if kw_list:
                print(f"\nKeywords: {', '.join(kw_list)}")
            
            themes = keywords.get("themes", [])
            if themes:
                print(f"\nThemes: {', '.join(themes)}")
            
            entities = keywords.get("entities", [])
            if entities:
                print(f"\nEntities: {', '.join(entities)}")
            
            topics = keywords.get("topics_of_interest", [])
            if topics:
                print(f"\nTopics of Interest: {', '.join(topics)}")
            
            questions = keywords.get("questions_asked", [])
            if questions:
                print("\nQuestions Asked:")
                for q in questions:
                    print(f"  ? {q}")
            
            print(f"\nAnalysis: {keywords.get('frequency_analysis', 'N/A')}")
            
            print("=" * 50 + "\n")
            
        except Exception as e:
            print_colored(f"❌ Error extracting keywords: {str(e)}\n", "red")
    
    def _show_trends(self):
        """Display sentiment trend analysis."""
        print("\n⏳ Analyzing trends...")
        
        try:
            sentiment_history = self.chatbot.get_sentiment_history()
            trends = self.analytics.analyze_trends(sentiment_history)
            
            print("\n" + "=" * 50)
            print("📈 SENTIMENT TRENDS")
            print("=" * 50)
            
            print(f"\nTrend: {trends.trend}")
            print(f"Direction: {trends.direction}")
            print(f"\nAnalysis: {trends.analysis}")
            print(f"\nPrediction: {trends.prediction}")
            
            if trends.mood_shifts:
                print("\nMood Shifts:")
                for shift in trends.mood_shifts:
                    print(f"  ↔ {shift}")
            
            if trends.emotional_peaks:
                print("\nEmotional Peaks:")
                for peak in trends.emotional_peaks:
                    print(f"  ⚡ {peak}")
            
            print("=" * 50 + "\n")
            
        except Exception as e:
            print_colored(f"❌ Error analyzing trends: {str(e)}\n", "red")
    
    def _show_graph(self):
        """Display ASCII mood graph."""
        print("\n⏳ Generating mood graph...")
        
        try:
            sentiment_history = self.chatbot.get_sentiment_history()
            graph = self.analytics.generate_mood_graph(sentiment_history)
            
            print("\n" + "=" * 60)
            print("📊 MOOD VISUALIZATION")
            print("=" * 60)
            print()
            print(graph)
            print()
            print("=" * 60 + "\n")
            
        except Exception as e:
            print_colored(f"❌ Error generating graph: {str(e)}\n", "red")
    
    def _show_profile(self):
        """Display emotion profile."""
        print("\n⏳ Generating emotion profile...")
        
        try:
            sentiment_history = self.chatbot.get_sentiment_history()
            profile = self.analytics.generate_emotion_profile(sentiment_history)
            
            print("\n" + "=" * 60)
            print("🎭 EMOTION PROFILE")
            print("=" * 60)
            print()
            print(profile)
            print()
            print("=" * 60 + "\n")
            
        except Exception as e:
            print_colored(f"❌ Error generating profile: {str(e)}\n", "red")
    
    def _show_full_report(self):
        """Display full analytics report."""
        print("\n⏳ Generating comprehensive report...")
        
        try:
            conversation_history = self.chatbot.get_history()
            sentiment_history = self.chatbot.get_sentiment_history()
            
            report = self.report_generator.generate_text_report(
                conversation_history,
                sentiment_history
            )
            
            print("\n" + report + "\n")
            
        except Exception as e:
            print_colored(f"❌ Error generating report: {str(e)}\n", "red")
    
    def _reset_conversation(self):
        """Reset the conversation."""
        confirm = input("Are you sure you want to reset? (y/n): ").strip().lower()
        
        if confirm == 'y':
            self.chatbot.reset()
            print_colored("\n✅ Conversation reset. Start fresh!\n", "green")
        else:
            print_colored("Reset cancelled.\n", "yellow")
    
    def _save_conversation(self):
        """Save the conversation to a file."""
        try:
            history = self.chatbot.get_history()
            filepath = self.logger.save_conversation(history)
            print_colored(f"\n✅ Conversation saved to: {filepath}\n", "green")
        except Exception as e:
            print_colored(f"❌ Error saving conversation: {str(e)}\n", "red")
    
    def _show_history(self):
        """Display conversation history."""
        history = self.chatbot.get_history()
        
        if len(history) <= 1:  # Only system message
            print_colored("\nNo conversation history yet.\n", "yellow")
            return
        
        print("\n" + "=" * 50)
        print("📜 CONVERSATION HISTORY")
        print("=" * 50)
        print()
        print(format_conversation_for_display(history))
        print("=" * 50 + "\n")
    
    def _exit(self):
        """Exit the application."""
        print()
        
        # Offer to save
        stats = self.chatbot.get_statistics()
        if stats.get("total_messages", 0) > 0:
            save = input("Would you like to save this conversation? (y/n): ").strip().lower()
            if save == 'y':
                self._save_conversation()
            
            # Show final summary
            print("\n⏳ Generating final summary...")
            try:
                summary = self.chatbot.get_conversation_summary()
                print_colored("\n📝 Final Summary:", "cyan")
                print(f"   {summary.get('summary', 'Conversation ended.')}\n")
            except:
                pass
        
        print_colored("👋 Thank you for chatting! Goodbye!\n", "green")
        self.running = False


def main():
    """Main entry point."""
    app = ChatbotApplication()
    app.run()


if __name__ == "__main__":
    main()

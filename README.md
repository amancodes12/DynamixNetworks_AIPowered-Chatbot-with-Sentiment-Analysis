# 🤖 AI Chatbot - Powered by Google Gemini Flash 2.5

A production-grade, **100% AI-powered** chatbot application with real-time sentiment analysis, emotion classification, conversation summarization, and trend analysis. All text processing is done through **Google Gemini Flash 2.5 API** - no rule-based sentiment analysis.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Gemini](https://img.shields.io/badge/Google-Gemini%202.5%20Flash-orange.svg)
![Flask](https://img.shields.io/badge/Flask-Web%20Interface-green.svg)
![Tests](https://img.shields.io/badge/Tests-60%20Passed-brightgreen.svg)

---

## 📋 Table of Contents

1. [How to Run](#-how-to-run)
2. [Chosen Technologies](#-chosen-technologies)
3. [Sentiment Logic Explanation](#-sentiment-logic-explanation)
4. [Tier 2 Implementation Status](#-tier-2-implementation-status)
5. [Tests](#-tests)
6. [Bonus Features & Innovations](#-bonus-features--innovations)
7. [Example Output](#-example-output)
8. [Project Structure](#-project-structure)

---

## 🚀 How to Run

### Prerequisites
- Python 3.8 or higher
- Google Gemini API key ([Get one free here](https://aistudio.google.com/app/apikey))

### Quick Start (Windows)

```powershell
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/ai-chatbot-gemini.git
cd ai-chatbot-gemini

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API key in config.py
# Open config.py and replace the API key value

# 5. Run the Web Interface (Recommended)
python web_app.py
# Open browser at: http://127.0.0.1:5000

# OR Run the CLI version
python app.py
```

### Quick Start (Linux/Mac)

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/ai-chatbot-gemini.git
cd ai-chatbot-gemini

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API key
export GEMINI_API_KEY="your-api-key-here"
# OR edit config.py directly

# 5. Run the Web Interface
python web_app.py

# OR Run CLI version
python app.py
```

### API Key Configuration

Edit `config.py`:
```python
GEMINI_API_KEY = "your-actual-api-key-here"
MODEL_NAME = "gemini-2.5-flash"
```

---

## 🛠 Chosen Technologies

| Technology | Purpose | Why Chosen |
|------------|---------|------------|
| **Python 3.8+** | Core Language | Industry standard for AI/ML, excellent library ecosystem |
| **Google Gemini Flash 2.5** | AI Engine | State-of-the-art LLM, fast responses, excellent reasoning |
| **Flask** | Web Framework | Lightweight, easy to use, perfect for API development |
| **google-genai** | Gemini SDK | Official Google client for Gemini API |
| **pytest** | Testing | Industry standard for Python testing |
| **HTML/CSS/JavaScript** | Frontend | Modern, responsive web interface |

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE                           │
│            (Web UI / CLI Application)                        │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                   APPLICATION LAYER                          │
│     app.py (CLI) / web_app.py (Flask Web Server)            │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                  BUSINESS LOGIC LAYER                        │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐    │
│  │  chatbot.py  │ │ sentiment.py │ │   analytics.py   │    │
│  │              │ │              │ │                  │    │
│  │ • History    │ │ • Analysis   │ │ • Trends         │    │
│  │ • State      │ │ • Emotions   │ │ • Reports        │    │
│  │ • Context    │ │ • Pipeline   │ │ • Graphs         │    │
│  └──────────────┘ └──────────────┘ └──────────────────┘    │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    AI CLIENT LAYER                           │
│                    (ai_client.py)                            │
│                                                              │
│  GeminiAIClient - Wrapper for all AI operations:            │
│  • generate_reply()      • analyze_sentiment()              │
│  • summarize_conversation() • extract_keywords()            │
│  • generate_trend_analysis() • generate_ascii_mood_graph()  │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              GOOGLE GEMINI FLASH 2.5 API                     │
│           (All AI Processing Happens Here)                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧠 Sentiment Logic Explanation

### Overview

**All sentiment analysis is performed by Google Gemini Flash 2.5 AI** - there is NO rule-based processing, NO TextBlob, NO NLTK. The AI analyzes each message in real-time and provides:

1. **Sentiment Classification**: positive / negative / neutral
2. **Confidence Score**: 0.0 to 1.0 (how certain the AI is)
3. **Emotion Tag**: happy, sad, angry, confused, excited, anxious, surprised, neutral, frustrated, hopeful
4. **Emotion Intensity**: low / medium / high
5. **Reasoning**: AI's explanation for the classification

### How It Works

```
User Message: "Your service disappoints me"
                    │
                    ▼
        ┌───────────────────────┐
        │   GEMINI FLASH 2.5    │
        │                       │
        │  Prompt Engineering:  │
        │  "Analyze sentiment   │
        │   and emotion..."     │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │   AI RESPONSE (JSON)  │
        │                       │
        │  sentiment: "negative"│
        │  confidence: 0.92     │
        │  emotion: "frustrated"│
        │  intensity: "high"    │
        │  reasoning: "User     │
        │  expressed            │
        │  disappointment..."   │
        └───────────────────────┘
```

### Sentiment Analysis Pipeline

1. **Message Received** → User input is captured
2. **AI Analysis** → Sent to Gemini with structured prompt requesting JSON response
3. **Parse Response** → Extract sentiment, emotion, confidence, and reasoning
4. **Update State** → Conversation state updated with mood tracking
5. **Generate Reply** → AI generates contextual response based on detected sentiment
6. **Store History** → All analysis stored for trend tracking

### Key Code (ai_client.py)

```python
def analyze_sentiment(self, message: str) -> Dict[str, Any]:
    prompt = f"""Analyze the following message for sentiment and emotion.

    MESSAGE: "{message}"

    Provide your analysis in JSON format:
    {{
        "sentiment": "positive|negative|neutral",
        "confidence": 0.0-1.0,
        "emotion": "happy|sad|angry|confused|excited|...",
        "emotion_intensity": "low|medium|high",
        "reasoning": "Brief explanation"
    }}"""
    
    response = self._generate_content(prompt)
    return json.loads(response)
```

### Conversation-Level Analysis

At conversation end, the AI provides:
- **Overall Sentiment**: Weighted analysis of all messages
- **Mood Trend**: How emotions evolved (improving/declining/stable)
- **Key Themes**: Topics discussed
- **Summary**: Comprehensive conversation summary

---

## ✅ Tier 2 Implementation Status

| Feature | Status | Description |
|---------|--------|-------------|
| **AI-Driven Chatbot** | ✅ Complete | Full conversation with Gemini Flash 2.5 |
| **Sentiment Analysis** | ✅ Complete | Real-time positive/negative/neutral |
| **Emotion Classification** | ✅ Complete | 10 emotion categories with intensity |
| **Confidence Scores** | ✅ Complete | 0-100% confidence for each analysis |
| **Reasoning Explanation** | ✅ Complete | AI explains each classification |
| **Conversation Summary** | ✅ Complete | AI-generated summary at any point |
| **Mood Trend Analysis** | ✅ Complete | Tracks emotional journey |
| **Keyword Extraction** | ✅ Complete | AI identifies key topics |
| **ASCII Mood Graph** | ✅ Complete | Visual mood representation |
| **Emotion Profile** | ✅ Complete | Personality insights |
| **Web Interface** | ✅ Complete | Modern, responsive UI |
| **CLI Interface** | ✅ Complete | Terminal-based interaction |
| **Unit Tests** | ✅ Complete | 60 tests with mocking |
| **Mood Shift Detection** | ✅ Complete | Detects emotional changes |
| **Contextual Responses** | ✅ Complete | AI adapts to user mood |

**All Tier 2 features are fully implemented.**

---

## 🧪 Tests

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test file
pytest tests/test_sentiment.py -v
```

### Test Results

```
======================== 60 passed in 1.56s ========================

tests/test_ai_client.py    - 17 tests ✅
tests/test_sentiment.py    - 18 tests ✅
tests/test_chatbot.py      - 25 tests ✅
```

### Test Coverage

| Module | Tests | Coverage |
|--------|-------|----------|
| **ai_client.py** | API wrapper, prompt construction, JSON parsing, error handling | ✅ |
| **sentiment.py** | Analysis pipeline, history management, statistics, distributions | ✅ |
| **chatbot.py** | Conversation flow, state management, mood detection, history | ✅ |

### Sample Test

```python
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
```

---

## 🌟 Bonus Features & Innovations

### 1. 🎨 Modern Web Interface
- Real-time sentiment badges on each message
- Emoji-based emotion indicators
- Live statistics dashboard
- Sentiment distribution chart
- Dark theme with gradient design

### 2. 📈 AI-Generated Mood Graph
- ASCII art visualization of emotional journey
- Generated entirely by AI, not hardcoded
- Fallback mechanism for reliability

### 3. 🔄 Mood Shift Detection
- Automatically detects when user mood changes
- Notifies: "📈 Mood improving: negative → positive"
- Adapts chatbot responses accordingly

### 4. 🎭 Emotion Profile
- AI-generated personality insights
- Communication style analysis
- Engagement recommendations

### 5. 💾 Conversation Persistence
- Save conversations to JSON files
- Load and continue previous chats
- Full history with sentiment data

### 6. 🎯 UI Hints System
- Color suggestions based on emotion
- Emoji recommendations
- Intensity level indicators

### 7. 🔧 Robust Error Handling
- Graceful fallbacks for API failures
- JSON parsing with error recovery
- User-friendly error messages

### 8. 📊 Comprehensive Analytics
- Full JSON report export
- Text-based formatted reports
- Statistical distributions

---

## 📝 Example Output

### Example Conversation

```
╔═══════════════════════════════════════════════════════════╗
║     🤖 AI CHATBOT - Powered by Google Gemini Flash 2.5    ║
╚═══════════════════════════════════════════════════════════╝

👤 User: "Your service disappoints me"

🤖 AI: "I'm truly sorry to hear that you're disappointed with our 
   service. Your feedback is important, and I want to make sure 
   your concern is addressed. Could you tell me more about what 
   went wrong so I can help resolve this?"

   ❌ Negative (92% confident) | 😤 Frustrated (high)
   
   Reasoning: "User explicitly expressed disappointment, indicating 
   dissatisfaction with the service. The word 'disappoints' is a 
   strong negative sentiment indicator."

─────────────────────────────────────────────────────────────

👤 User: "Last experience was better"

🤖 AI: "I appreciate you sharing that comparison. It's helpful to 
   know that you've had better experiences before - that gives us 
   a benchmark to work toward. What made that previous experience 
   stand out for you?"

   ✅ Positive (78% confident) | 🌟 Hopeful (medium)
   📈 Mood improving: negative → positive
   
   Reasoning: "User is comparing favorably to a past experience, 
   suggesting they have positive memories and expectations."

═════════════════════════════════════════════════════════════
                    FINAL OUTPUT
═════════════════════════════════════════════════════════════

📊 CONVERSATION SUMMARY
─────────────────────────────────────────────────────────────
Overall Sentiment: NEGATIVE - General dissatisfaction

Analysis: The conversation began with the user expressing 
disappointment with the service. While the second message 
showed a slight improvement in tone by referencing positive 
past experiences, the overall sentiment remains negative due 
to the initial strong expression of dissatisfaction.

Mood Journey: Started frustrated → Became slightly hopeful

Key Themes: Service quality, Past experiences, Expectations

Recommendation: Continue with empathetic responses and focus 
on understanding specific pain points to address concerns.
═════════════════════════════════════════════════════════════
```

### Web Interface Screenshot

```
┌─────────────────────────────────────────────────────────────┐
│  🤖 AI Chatbot          [New Chat] [Full Report]           │
├───────────────────────────────────┬─────────────────────────┤
│                                   │  😤 Current Mood        │
│  👤 Your service disappoints me   │  Frustrated             │
│     ❌ Negative (92%) | 😤        │  Negative • 92%         │
│                                   ├─────────────────────────┤
│  🤖 I'm truly sorry to hear...    │  📊 Statistics          │
│                                   │  Messages: 2            │
│  👤 Last experience was better    │  Confidence: 85%        │
│     ✅ Positive (78%) | 🌟        ├─────────────────────────┤
│     📈 Mood improving             │  Sentiment Distribution │
│                                   │  Positive  ████░░ 50%   │
│  🤖 I appreciate you sharing...   │  Neutral   ░░░░░░  0%   │
│                                   │  Negative  ████░░ 50%   │
│                                   ├─────────────────────────┤
│  [Type your message...    ] [➤]   │  [Summary] [Keywords]   │
│                                   │  [Trends]  [Mood Graph] │
└───────────────────────────────────┴─────────────────────────┘
```

---

## 📁 Project Structure

```
ai_chatbot_gemini/
│
├── app.py                 # CLI application entry point
├── web_app.py             # Flask web application
├── ai_client.py           # Gemini API wrapper (all AI operations)
├── chatbot.py             # Conversation management
├── sentiment.py           # Sentiment analysis pipeline
├── analytics.py           # Trend analysis & reporting
├── utils.py               # Utility functions
├── config.py              # Configuration settings
│
├── templates/
│   └── index.html         # Web interface template
│
├── tests/
│   ├── __init__.py
│   ├── test_ai_client.py  # AI client tests (17 tests)
│   ├── test_sentiment.py  # Sentiment tests (18 tests)
│   └── test_chatbot.py    # Chatbot tests (25 tests)
│
├── requirements.txt       # Python dependencies
├── environment_setup.sh   # Linux/Mac setup script
├── environment_setup.bat  # Windows setup script
├── .gitignore            # Git ignore file
└── README.md             # This file
```

---

## 📄 License

MIT License - Feel free to use, modify, and distribute.

---

## 👤 Author

**Gourvi Chawla**

---

## 🙏 Acknowledgments

- Google Gemini Team for the amazing AI API
- Flask community for the web framework
- All open-source contributors

---

**Built with ❤️ using Google Gemini Flash 2.5**

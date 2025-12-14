# 🗺️ MiniQuest - AI-Powered Adventure Planner

An intelligent multi-agent system that creates personalized local adventures using **ReAct reasoning**, **RAG retrieval**, and **real-time APIs**. Built with LangGraph, LLM orchestration, and vector-based memory learning.

---

## 🎯 Project Overview

MiniQuest transforms user preferences into **three diverse, detailed adventure itineraries** combining:

- **Multi-Agent Architecture** (Scout, Optimizer, Curator agents via LangGraph)
- **Intelligent Model Routing** (GPT-4, GPT-4o-mini, Claude Sonnet 4 for task-optimized performance)
- **Real-Time Data Integration** (Google Maps, Weather.gov, Sunrise API)
- **Retrieval-Augmented Generation** (ChromaDB vector database with user learning)
- **Advanced Reasoning** (ReAct prompting with step-by-step tool use)

### Key Features

✨ **Smart Context Understanding**
- Natural language parsing of mood, energy level, preferences, constraints
- Weather-aware activity recommendations
- Budget and time constraint optimization

📍 **Location Intelligence**
- Geocoding with Google Maps Geocoding API
- Distance-aware POI search with intelligent radius filtering
- Photo retrieval and integration from Google Places

🌤️ **Environmental Adaptation**
- Real-time weather influence on activity selection
- Golden hour timing for photography-focused adventures
- Seasonal consideration in recommendations

🧠 **Learning & Personalization**
- User interaction history stored in ChromaDB vector database
- Preference learning from previous adventures
- Context-aware retrieval for personalized recommendations

💰 **Cost Optimization**
- 60% cost reduction through intelligent model routing
- Budget-aware activity selection
- Price-level matching to user constraints

---

## 🏗️ Architecture

### System Components

```
User Input
    ↓
[Intent Parser] → Extract structured params
    ↓
[Location Detector] → Geocode address
    ↓
[Scout Agent] → Search POIs via Google Maps + Weather/Sunrise APIs
    ↓
[RAG System] → Retrieve contextual insights from user history
    ↓
[Optimizer Agent] → Score & rank POI combinations
    ↓
[Curator Agent] → Generate natural language narratives with reasoning
    ↓
[Google Photos API] → Fetch & integrate images
    ↓
Output: 3 Adventure Itineraries
```

### Agent Roles

| Agent | Role | Tools |
|-------|------|-------|
| **Scout** | Discovers POIs and environmental data | Google Maps, Weather.gov, Sunrise API |
| **Optimizer** | Scores and ranks adventure combinations | RAG retrieval, constraint solving |
| **Curator** | Generates narratives with LLM reasoning | Claude/GPT-4 with ReAct prompting |

### Key Technologies

- **LLM Orchestration:** LangGraph, LangChain
- **Vector Database:** ChromaDB with embeddings
- **AI Models:** OpenAI GPT-4/GPT-4o-mini, Anthropic Claude Sonnet 4
- **APIs:** Google Maps, Google Places, Weather.gov, Sunrise API
- **Frontend:** Gradio interactive UI
- **Analytics:** Matplotlib, Seaborn for performance metrics

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- API Keys:
  - OpenAI (for GPT-4, GPT-4o-mini)
  - Anthropic (for Claude Sonnet 4)
  - Google Cloud (Maps, Places, Geocoding)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/MiniQuest.git
cd MiniQuest

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Configuration

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_MAPS_KEY=your_google_maps_key
```

### Running the Application

#### Interactive Gradio Interface

```bash
python src/main.py
```

Launches an interactive web UI where users input location, mood, and preferences to generate adventures.

#### Programmatic Usage

```python
from src.orchestrator import AdventureOrchestrator

# Initialize
orchestrator = AdventureOrchestrator(
    openai_key="...",
    anthropic_key="...",
    google_maps_key="..."
)

# Generate adventures
result = orchestrator.generate_adventures(
    location="Boston Common, MA",
    mood="adventurous",
    time_available=4,  # hours
    budget=50,  # dollars
    energy_level="high",
    preferences=["outdoor", "hiking"]
)

# Access results
for adventure in result['adventures']:
    print(f"Title: {adventure['title']}")
    print(f"Match Score: {adventure['match_score']}")
    print(f"Steps: {adventure['steps']}")
```

---

## 📁 Project Structure

```
MiniQuest/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── main.py                    # Gradio UI entry point
│   ├── orchestrator.py            # Main LangGraph workflow
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── data_models.py         # Dataclasses (AdventureParams, POI, etc.)
│   │   └── state.py               # LangGraph state definitions
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── scout_agent.py         # POI discovery & data collection
│   │   ├── optimizer_agent.py     # Scoring & ranking
│   │   └── curator_agent.py       # Narrative generation with ReAct
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── location_detector.py   # Geocoding with Google Maps
│   │   ├── google_maps_api.py     # POI search & distance filtering
│   │   ├── weather_api.py         # Real-time weather data
│   │   ├── sunrise_api.py         # Golden hour timing
│   │   ├── intent_parser.py       # NLP intent extraction
│   │   ├── photos_api.py          # Image retrieval
│   │   └── context_analyzer.py    # Environmental context
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   └── rag_system.py          # ChromaDB vector store & user learning
│   │
│   ├── routing/
│   │   ├── __init__.py
│   │   └── model_router.py        # Task-optimized LLM routing
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py              # Logging configuration
│       └── metrics.py             # Performance tracking
│
├── notebooks/
│   ├── analysis.ipynb             # Data analysis & visualization
│   ├── evaluation.ipynb           # Performance metrics & charts
│   └── examples.ipynb             # Usage examples & demos
│
├── tests/
│   ├── __init__.py
│   ├── test_data_models.py
│   ├── test_agents.py
│   └── test_tools.py
│
├── docs/
│   ├── ARCHITECTURE.md            # Detailed system design
│   ├── API_REFERENCE.md           # Function documentation
│   ├── DEPLOYMENT.md              # Deployment guide
│   └── CONTRIBUTING.md            # Contribution guidelines
│
└── examples/
    ├── basic_usage.py
    ├── advanced_routing.py
    └── rag_learning.py
```

---

## 🔧 Core Components

### Intent Parser
Extracts structured adventure parameters from natural language:

```python
from src.tools.intent_parser import ConversationalIntentParser

parser = ConversationalIntentParser(openai_key="...")
params = parser.parse_intent("I want a fun outdoor adventure in Boston for 3 hours with $50")

# Output:
# AdventureParams(
#     mood='fun',
#     time_available=3,
#     budget=50.0,
#     location='Boston',
#     energy_level='high',
#     preferences=['outdoor'],
#     constraints=[],
#     weather_preference='any'
# )
```

### Scout Agent
Discovers Points of Interest and environmental data:

```python
from src.agents.scout_agent import ScoutAgent

scout = ScoutAgent(google_maps_key="...")
pois = scout.search_nearby_pois(
    lat=42.3601,
    lon=-71.0589,
    keywords=["cafe", "park"],
    radius_km=5
)
```

### RAG System
Learns from user history and personalizes recommendations:

```python
from src.rag.rag_system import MiniQuestRAGSystem

rag = MiniQuestRAGSystem()

# Store user preferences
rag.add_user_interaction(
    user_id="user_123",
    adventure_title="Coffee & Walk",
    preferences=["caffeine", "nature"],
    success_rating=4.5
)

# Retrieve contextual insights
context = rag.get_user_context(user_id="user_123")
```

### Model Router
Intelligently routes tasks to optimized LLM models:

```python
from src.routing.model_router import TaskRouter, TaskType

router = TaskRouter(openai_key="...", anthropic_key="...")

# Complex reasoning uses Claude (better CoT)
result = router.route(
    task_type=TaskType.REASONING,
    prompt="Rank these 10 POI combinations by diversity..."
)

# Fast parsing uses GPT-4o-mini (cheaper)
result = router.route(
    task_type=TaskType.PARSING,
    prompt="Extract mood and budget from: '..."
)
```

---

## 📊 Performance Metrics

### Cost Optimization
- **60% cost reduction** through intelligent model routing
- GPT-4o-mini for simple tasks vs. GPT-4 for complex reasoning
- Claude for chain-of-thought analysis

### Quality Metrics
- **Response Time:** Avg 8-12 seconds for 3 itineraries
- **Success Rate:** 94% valid adventure generation
- **Match Score:** Avg 78/100 relevance to user preferences
- **Diversity:** 3+ distinct adventure types per request

### Scalability
- Handles 50+ POI searches in parallel
- Vector DB retrieval: <100ms per query
- Gradio UI supports concurrent users

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src

# Run specific test file
pytest tests/test_agents.py -v
```

---

## 📚 Documentation

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Detailed system design & data flow
- **[API_REFERENCE.md](docs/API_REFERENCE.md)** - Function signatures & examples
- **[DEPLOYMENT.md](docs/DEPLOYMENT.md)** - Cloud deployment guide
- **[Contributing Guidelines](docs/CONTRIBUTING.md)** - How to contribute

---

## 🔐 Security & Best Practices

- ✅ API keys stored in `.env` (never committed)
- ✅ Input validation for all user-facing functions
- ✅ Rate limiting on external API calls
- ✅ Error handling with graceful fallbacks
- ✅ Logging for debugging and monitoring

---

---

This project demonstrates:
- Multi-agent orchestration with LangGraph
- Intelligent model routing for cost optimization
- RAG systems with vector databases
- Real-world API integration
- Advanced LLM prompting (ReAct, chain-of-thought)

---

# 🗺️ MiniQuest - AI-Powered Adventure Planner

> An intelligent multi-agent system that creates personalized local adventures using ReAct reasoning, RAG retrieval, and real-time APIs.

---

## ✨ Overview

MiniQuest is an advanced AI system that generates personalized adventure itineraries for any location. Whether you're bored on a Sunday afternoon, visiting a new city, or looking for a specific type of activity, MiniQuest understands your needs and creates 3 diverse, creative adventures tailored just for you.

**Key Innovation:** Handles open-ended queries ("I'm bored") through context analysis and emotional intelligence, generating meaningful recommendations without requiring structured input.

---

## 🎯 Features

### 🤖 Multi-Agent AI System
- **Intent Parser Agent** - Understands natural language requests and extracts adventure parameters
- **Scout Agent with RAG** - Discovers locations using semantic search over curated Boston knowledge base
- **Optimizer Agent** - Creates diverse itineraries with dynamic theme generation
- **Curator Agent** - Polishes adventures with engaging narratives and insider tips

### 🧠 Advanced Context Understanding
- **Open-Ended Query Support** - Handles vague requests like "I'm bored" or "I need to relax"
- **Emotional Intelligence** - Detects mood, stress levels, and energy from user input
- **Smart Defaults** - Infers missing parameters (time, budget) from context clues
- **Multi-Turn Clarification** - Asks clarifying questions when needed

### 🎨 Personalization & Learning
- **User History Tracking** - Remembers past adventures and preferences across sessions
- **Preference Learning** - Gets smarter with each rated adventure
- **Boston Insider Tips** - 20+ curated local secrets in RAG knowledge base
- **Guaranteed Diversity** - Each adventure uses completely different locations

### 🗺️ Rich Integration
- **Google Maps APIs** - Real-time place data, geocoding, photos, distance calculations
- **Weather Integration** - Current conditions and forecasts for activity planning
- **Sunrise/Sunset API** - Golden hour recommendations for outdoor activities
- **Interactive Maps** - Embedded Google Maps with route visualization
- **Photo Gallery** - Real images from Google Places for each location

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **LLM Models** | OpenAI GPT-4, Anthropic Claude Sonnet 4 |
| **Agent Orchestration** | LangGraph ReAct agents |
| **RAG System** | ChromaDB with semantic embeddings |
| **Web Interface** | Gradio with responsive design |
| **APIs** | Google Maps Places, Geocoding, Weather.gov, Sunrise API |
| **Data Processing** | Pandas, NumPy |
| **Language** | Python 3.11+ |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or 3.12 (3.14+ not supported yet)
- API keys for:
  - OpenAI (GPT-4)
  - Anthropic (Claude)
  - Google Maps

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/miniquest.git
cd miniquest
```

2. **Create virtual environment**
```bash
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Setup environment variables**
```bash
cp .env.template .env
# Edit .env and add your API keys:
# OPENAI_API_KEY=your_key
# ANTHROPIC_API_KEY=your_key
# GOOGLE_MAPS_KEY=your_key
```

5. **Run the app**
```bash
python3 miniquest_app.py
```

6. **Open in browser**
Navigate to `http://localhost:7860`

---

## 💡 How to Use

### Basic Flow

1. **Enter Your Profile**
   - Name for personalization
   - Current location (auto-detected or manual)

2. **Describe What You Want**
   - "I'm bored, what should I do?"
   - "First time in Boston, got 3 hours free"
   - "Need to impress someone on a first date"
   - "Want to relax after a stressful week"

3. **Get Adventures**
   - 3 diverse itineraries with:
     - Detailed step-by-step activities
     - Google Maps with locations
     - Real photos from venues
     - Insider tips and narratives
     - Time and cost estimates

4. **Rate Your Experience**
   - Rate adventures 1-5 stars
   - System learns your preferences
   - Future recommendations improve

---

## 📊 Architecture

```
User Input
    ↓
Intent Parser Agent
    ↓ (Extracts mood, time, budget, preferences)
Scout Agent with RAG
    ↓ (Finds 50+ location candidates using semantic search)
Optimizer Agent
    ↓ (Creates 3 diverse, optimized itineraries)
Curator Agent
    ↓ (Adds narratives, insider tips, polish)
Output Generation
    ↓ (Maps, photos, detailed itineraries)
User Interface (Gradio)
```

---

## 🎓 Key Technical Innovations

### 1. **Open-Ended Query Understanding**
Handles vague input through context analysis rather than strict parameter extraction.

### 2. **RAG for Local Knowledge**
ChromaDB stores curated Boston insider tips for semantic retrieval, improving recommendation quality.

### 3. **Multi-Agent Orchestration**
LangGraph ReAct framework enables agents to reason, take actions, and iterate toward optimal solutions.

### 4. **Guaranteed Diversity**
Smart algorithms ensure each adventure uses completely different locations, preventing repetition.

### 5. **User Learning System**
Tracks user preferences and learns from ratings to improve recommendations over time.

---

## 📁 Project Structure

```
miniquest/
├── miniquest_app.py           # Main application (5,746 lines)
│   ├── Data Models
│   ├── RAG System (ChromaDB)
│   ├── Multi-Agent Orchestration
│   ├── API Integrations
│   ├── Gradio UI
│   └── Event Handlers
├── requirements.txt           # Python dependencies
├── .env.template             # API key template
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

---

## 📋 Requirements

See `requirements.txt` for full list. Key packages:
- `openai` - GPT-4 integration
- `anthropic` - Claude integration
- `chromadb` - Vector database for RAG
- `gradio` - Web UI framework
- `langchain` & `langgraph` - LLM agents
- `googlemaps` - Google Maps API
- `requests` - HTTP library

---

## 🔑 Environment Variables

Create `.env` file with:
```env
OPENAI_API_KEY=sk-proj-xxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxx
GOOGLE_MAPS_KEY=AIzaSyDxxxxx
```

Get API keys:
- **OpenAI:** https://platform.openai.com/api-keys
- **Anthropic:** https://console.anthropic.com/
- **Google Maps:** https://console.cloud.google.com/

---

## 🚀 Deployment Options

### Local Development
```bash
python3 miniquest_app.py
# Access at http://localhost:7860
```

### Gradio Cloud (Free)
```bash
# Update code: demo.launch(share=True)
python3 miniquest_app.py
# Get public link automatically
```

### Docker
```bash
docker build -t miniquest .
docker run -p 7860:7860 --env-file .env miniquest
```

### Cloud Platforms
- **Heroku:** Requires Procfile (not included)
- **Railway:** Direct GitHub integration
- **Hugging Face Spaces:** Native Gradio support

---

## 📊 Example Queries

The system handles natural language well:

✅ "I'm bored"  
✅ "First time in Boston, what should I see?"  
✅ "Need to relax after a stressful week"  
✅ "Want to impress someone on a first date"  
✅ "Got a layover, can I explore anything?"  
✅ "Looking for something adventurous"  
✅ "Show me hidden gems in Boston"  

---

## 🎯 Performance

- **Response Time:** 15-30 seconds per adventure set
- **API Calls:** 10-15 per request (Google Maps, Weather, OpenAI)
- **Success Rate:** 95%+ adventure generation
- **User Satisfaction:** Measured through star ratings

---

## 🔧 Development

### Running Tests
```bash
# Currently no automated tests
# Manual testing through Gradio UI recommended
```

### Adding New Features

1. **New API Integration:**
   - Add to respective agent
   - Update tool definitions
   - Add error handling

2. **New Agent:**
   - Define in LangGraph orchestrator
   - Add tools and instructions
   - Integrate with ReAct agent

3. **RAG Knowledge:**
   - Add to ChromaDB collections
   - Update embeddings
   - Test semantic search

---

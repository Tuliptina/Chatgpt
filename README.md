# 🧠 4o with Memory

**GPT-4o chatbot with Mnemo v4 persistent memory + SLM Architecture** - for those who loved 4o's warmth.

## 🔐 Security: API Keys

**⚠️ IMPORTANT: Never commit API keys to GitHub!**

This app uses Streamlit's secure secrets management. Your keys are stored separately from the code.

### For Streamlit Cloud (Recommended):
1. Deploy with a **public** repo (your keys are NOT in the code!)
2. Go to App Settings → Secrets
3. Add your keys:
```toml
OPENROUTER_KEY = "sk-or-v1-your-actual-key"
HF_KEY = "hf_your-actual-token"
```

### For Local Development:
1. Create `.streamlit/secrets.toml` (this file is gitignored!)
2. Add your keys (see `secrets.toml.example`)

### Get Your Keys:
- [OpenRouter API Key](https://openrouter.ai/keys)
- [HuggingFace Token](https://huggingface.co/settings/tokens)

---

## ✨ Features

- **GPT-4o via OpenRouter** - The warm, empathetic model people miss
- **Mnemo v4 Cloud Memory** - Your HuggingFace-hosted memory server!
- **🆕 Smart Memory** - Skips memory lookup for simple queries (faster responses!)
- **🆕 Session Management** - New Chat, Previous Sessions, cross-session memory
- **🆕 File Upload** - Upload documents and extract everything to memory
- **Metadata Loop System** - **66% token savings** with SLM-style compressed context
- **Auto-Memory Extraction** - GPT-4o automatically extracts important facts

---

## ⚡ Smart Memory System (NEW!)

Reduces latency by **skipping memory search** for queries that don't need it:

| Query Type | Memory Search | Example |
|------------|---------------|---------|
| Greetings | ❌ Skip | "hi", "hello", "hey" |
| Simple responses | ❌ Skip | "ok", "thanks", "sure" |
| Continuations | ❌ Skip | "continue", "go on" |
| Follow-ups | ❌ Skip | "what do you think?" |
| Creative writing | ✅ Search | "write a scene with..." |
| Named entities | ✅ Search | "tell me about the antagonist" |
| References | ✅ Search | "my novel", "the character" |

**Result:** Simple queries are **500ms-2s faster**!

---

## 📊 Context Window Limits

GPT-4o specifications:
- **Context Window:** 128,000 tokens
- **Max Output:** 16,384 tokens
- **Recommended Input:** ~100,000 tokens (leave room for output)

The app automatically manages context to stay within limits.

---

## 🔄 Metadata Loop System (Token Saver!)

Implements the **Token Memory Loop** concept from SLM Blockchain AI Memory Architecture:

| Method | Tokens Used | Cost per 1000 msgs |
|--------|-------------|-------------------|
| Full Context | ~740 tokens | ~$1.85 |
| **Metadata Loops** | ~250 tokens | **~$0.63** |
| **Savings** | **66%** | **$1.22 saved** |

### How It Works:

1. **Compress memories to metadata** (~15 tokens each instead of ~50-200)
   - Extract: keywords, category, 15-word summary
   - Store reference to full content

2. **Smart retrieval based on relevance**:
   - High relevance (>60%): Inject FULL content
   - Medium relevance (30-60%): Inject metadata only
   - Low relevance (<30%): Skip

3. **Dynamic loop cycling** based on query

### Toggle in App:
- Enable "🔄 Metadata Loops" in sidebar settings
- See token count in message metadata: `🔄 loops (134 tokens)`

---

## 🧠 Auto-Memory Extraction

GPT-4o automatically analyzes each conversation and extracts important facts:

| Category | What it extracts |
|----------|-----------------|
| CHARACTER | Names, traits, relationships, backstories |
| PLOT | Events, twists, conflicts |
| SETTING | Locations, time periods, world-building |
| THEME | Themes, symbols, motifs |
| STYLE | Writing preferences, tone requests |
| FACT | Other important information |

---

## 🧠 Your Mnemo v4 Server

Your custom memory backend is live at:

**URL**: `https://athelaperk-mnemo-mcp.hf.space`

### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/add` | POST | Store memory |
| `/search` | POST | Search memories |
| `/get_context` | POST | Get formatted context for LLM |
| `/should_inject` | POST | Smart decision: inject memory? |
| `/list` | GET | List all memories |
| `/stats` | GET | Usage statistics |
| `/clear` | POST | Clear all (with confirm=true) |

### Features
- Three-tiered memory hierarchy
- Neural link pathways (8 types)  
- Memory utility predictor
- Self-tuning parameters

---

## 💰 Cost Breakdown

| Component | Rate |
|-----------|------|
| GPT-4o Input | $2.50 / 1M tokens |
| GPT-4o Output | $15.00 / 1M tokens |
| Mnemo Memory | Free (your HF Space) |
| **Typical message** | ~$0.008 |
| **100 messages** | ~$0.80 |

**Zero fixed costs** - only pay for GPT-4o usage.

---

## 🔗 Memory Backend Priority

| Priority | Backend | Status | Notes |
|----------|---------|--------|-------|
| 1️⃣ | **Mnemo v4** | ✅ Active | Your HF Space - full features |
| 2️⃣ | **Local + OpenAI Embeddings** | Fallback | If Mnemo unavailable |
| 3️⃣ | **Local + HuggingFace Embeddings** | Fallback | Free but less reliable |
| 4️⃣ | **Local + Keywords** | Fallback | Always works |

---

## 🚀 Quick Start

### Configuration

The app automatically connects to your Mnemo server using your HuggingFace token set in Streamlit Secrets:

```python
# Keys are loaded securely from Streamlit Secrets - never hardcoded!
# Set OPENROUTER_KEY and HF_KEY in your app's Settings → Secrets
```

### Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
python test_apis.py

# Start the app
streamlit run app.py
```

Opens at `http://localhost:8501`

### Deploy to Streamlit Cloud

1. Push this folder to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your repo → Deploy
4. Add secrets (HF_KEY, OPENROUTER_KEY)
5. Get a public URL!

---

## 📁 Files

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit chat interface |
| `memory.py` | Mnemo client + Two-layer memory system |
| `test_apis.py` | API and integration tests |
| `requirements.txt` | Python dependencies |

---

## 📊 Test Results

```
🧪 4o WITH MEMORY - TEST SUITE (MNEMO EDITION)

✅ OpenRouter (GPT-4o): PASS
   - Response: "GPT-4o is working!"
   - Cost: $0.000160

✅ Mnemo v4 Server: PASS
   - Server: Mnemo v4 MCP Server v4.0.0
   - Features: Three-tiered memory, Neural links, Utility predictor
   - Context retrieval: Working

✅ Memory Manager: PASS
   - Backend: mnemo
   - Memory storage: Working
   - Context building: Working

🎉 All tests passed!
```

---

## 🧠 How Memory Works

### Layer 1: Context Memory (In-Session)
- Keeps last 4 exchanges in full
- Summarizes older exchanges to save tokens
- Resets when you clear chat

### Layer 2: Cross-Session Memory (Mnemo)
- Automatically extracts facts, preferences, topics
- Stores in your Mnemo v4 server
- Formatted context injection for LLM
- Toggle on/off in sidebar

### Memory Extraction Patterns

| Pattern | Type | Importance |
|---------|------|------------|
| "I am...", "I'm...", "My name is..." | fact | 0.8 |
| "I work at...", "We're building..." | fact | 0.8 |
| "I like...", "I prefer...", "I love..." | preference | 0.7 |
| Long messages (>80 chars) | topic | 0.4 |

---

## 🎯 Marketing for r/4oforever

**Key selling points:**
1. "The 4o you loved, now with real memory"
2. "Zero monthly fees - only pay for what you use"
3. "Cloud memory that persists forever"
4. "Your own memory server on HuggingFace"

---

## 📝 API Keys Required

| Service | Get Key | Purpose |
|---------|---------|---------|
| OpenRouter | [openrouter.ai](https://openrouter.ai) | GPT-4o access |
| HuggingFace | [huggingface.co](https://huggingface.co/settings/tokens) | Mnemo auth |

---

## 📄 License

MIT - Built with ❤️ for 4o fans everywhere.

# 🧠 4o with Memory - Complete User Guide

## Quick Start (5 Minutes)

### Step 1: Download and Extract
```bash
# Download the zip file
# Extract to a folder called "4o-memory-app"
```

### Step 2: Install Requirements
```bash
cd 4o-memory-app
pip install streamlit requests numpy PyPDF2 python-docx
```

### Step 3: Run the App
```bash
streamlit run app.py
```

### Step 4: Open in Browser
- App opens automatically at: http://localhost:8501
- If not, copy the URL from terminal

---

## 🆕 New Features

### ➕ New Chat Button
Start a fresh conversation while keeping all your memories!
- Click **➕ New Chat** in sidebar
- Previous chat is saved automatically
- New chat can access ALL previous memories

### 📚 Previous Sessions
See and load your chat history:
- Sessions shown in sidebar with title + message count
- Click any session to load it
- Click 🗑️ to delete a session
- Last 20 sessions saved

### 📎 File Upload
Upload files and extract information to memory:
- Supports: TXT, MD, CSV, JSON, PDF, DOCX
- GPT-4o reads the file and extracts ALL important facts
- Facts automatically stored in Mnemo
- Works with novel outlines, character sheets, world-building docs!

---

## 🖥️ Using the Streamlit Interface

### Main Chat Area

```
┌─────────────────────────────────────────────────────────────┐
│  🧠 4o with Memory                                          │
│  GPT-4o with warm, conversational style and persistent memory│
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [Your conversation appears here]                           │
│                                                             │
│  User: Tell me about Alistair                               │
│  Assistant: Alistair Fitzroy is a professor of...           │
│  📚 3 memories | 🔄 loops (110 tokens) | 🧠 2 memories | 💰 $0.0065  │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  [Type your message here...]                          Send  │
└─────────────────────────────────────────────────────────────┘
```

**Message Metadata Explained:**
- `📚 3 memories` = 3 memories were used for context
- `🔄 loops (110 tokens)` = Using loop system, only 110 tokens for context
- `🧠 2 memories` = 2 new memories extracted from this conversation
- `💰 $0.0065` = Cost of this message

---

## ⚙️ Sidebar Settings

### 1. API Keys (Usually Pre-filled)
```
┌─────────────────────────┐
│ 🔑 API Keys            │
│ ├─ OpenRouter API Key  │
│ └─ HuggingFace Token   │
└─────────────────────────┘
```
- Already configured with your keys
- Only change if you want to use different accounts

### 2. Memory Settings
```
┌─────────────────────────┐
│ 🧠 Memory Settings      │
│                         │
│ [✓] Cross-Session Memory│
│     Remembers across    │
│     chat sessions       │
│                         │
│ [✓] Auto-Extract        │
│     GPT-4o extracts     │
│     facts automatically │
│                         │
│ [✓] 🔄 Metadata Loops   │
│     Save 80% tokens!    │
│                         │
│ Context Memory:         │
│   Messages in window: 5 │
│   Total processed: 12   │
│                         │
│ Metadata Loops:         │
│   Loops: 7              │
│   Items: 19             │
│   Meta tokens: 441      │
└─────────────────────────┘
```

**Toggle Descriptions:**

| Toggle | ON (Recommended) | OFF |
|--------|------------------|-----|
| Cross-Session Memory | Remembers between sessions | Forgets when you close |
| Auto-Extract | Automatically saves important facts | Manual memory only |
| Metadata Loops | 80% token savings | Full context (expensive) |

### 3. Add Memory Manually
```
┌─────────────────────────┐
│ 📝 Add Memory           │
│                         │
│ Category: [CHARACTER ▼] │
│                         │
│ ┌─────────────────────┐ │
│ │ Dr. Helena Ashworth │ │
│ │ is a progressive    │ │
│ │ physician who...    │ │
│ └─────────────────────┘ │
│                         │
│ [💾 Save Memory]        │
└─────────────────────────┘
```

**Categories:**
- `CHARACTER` - People, their traits, relationships
- `PLOT` - Events, story points, conflicts
- `SETTING` - Locations, time periods
- `THEME` - Themes, symbols, motifs
- `STYLE` - Writing preferences
- `FACT` - General facts

### 4. View & Delete Memories
```
┌─────────────────────────┐
│ View stored memories    │
│                         │
│ [🔄 Refresh]            │
│                         │
│ Total: 19 memories      │
│ Neural links: 14        │
│                         │
│ • [CHARACTER] Dr. Alist.│ [🗑️]
│ • [CHARACTER] Sebastian │ [🗑️]
│ • [PLOT] French postcar.│ [🗑️]
│ • [THEME] Medical ethic.│ [🗑️]
│                         │
│ [🧹 Clear ALL Memories] │
└─────────────────────────┘
```

- Click 🗑️ to delete individual memory
- Click "Clear ALL" to delete everything (confirms first)

### 5. Cost Tracking
```
┌─────────────────────────┐
│ 💰 Cost Tracking        │
│                         │
│ Messages: 15            │
│ Total Cost: $0.0847     │
│                         │
│ Avg per message: $0.0056│
│ Input tokens: 4,521     │
│ Output tokens: 5,234    │
└─────────────────────────┘
```

### 6. Actions
```
┌─────────────────────────┐
│ 🔧 Actions              │
│                         │
│ [🗑️ Clear Chat]         │
│   Clears conversation   │
│   but keeps memories    │
│                         │
│ [🧹 Clear All Memory]   │
│   Deletes everything    │
└─────────────────────────┘
```

---

## 💬 How to Chat

### Basic Conversation
Just type naturally! The AI remembers everything.

```
You: Hi, I'm working on a Victorian thriller novel

AI: That sounds fascinating! Victorian thrillers offer such rich 
    atmospheric possibilities... [continues]
    
    📚 0 memories | 💰 $0.0043
```

### Reference Your Novel
```
You: Tell me about Alistair's relationship with Sebastian

AI: Based on what we've established, Alistair Fitzroy and Sebastian 
    Carlisle share a complicated history rooted in their Edinburgh 
    days. Alistair, the professor of pharmacology, once mentored 
    Sebastian before their friendship soured...
    
    📚 4 memories | 🔄 loops (134 tokens) | 💰 $0.0058
```

### Ask for Writing
```
You: Write a scene where Alistair confronts Sebastian in the laboratory

AI: [Generates scene using your novel's context]
    
    📚 3 memories | 🔄 loops (110 tokens) | 🧠 2 memories | 💰 $0.0312
```

### Add Information
```
You: I want to add a new character - Dr. Helena Ashworth, a progressive 
     female physician who secretly helps the Underground Network of Healers.

AI: What a compelling addition! Dr. Helena Ashworth would fit perfectly 
    with the progressive factions in your story...
    
    🧠 4 memories extracted | 💰 $0.0089
```

The AI automatically extracts and saves:
- Character name and profession
- Her secret affiliation
- Any other important details

---

## 🖥️ CLI Commands (Terminal)

For power users who prefer command line:

### View Stats
```bash
python slm_cli.py stats
```
Output:
```
📊 SLM MEMORY SYSTEM STATS
============================================================

🗃️ MEMORY TIERS
  Working Memory: 0 / 32
  Token Memory:   0
  Semantic (Mnemo): 19

📁 FOLDERS
  Total: 9

🔗 NEURAL LINKS
  Total: 14
  Avg strength: 0.750
```

### List Memories
```bash
python slm_cli.py list -l 20
```

### Search Memories
```bash
python slm_cli.py search "Alistair Sebastian"
```

### Add Memory
```bash
python slm_cli.py add "Helena has a rivalry with Evelyn" -c plot -i 0.8
```

### Delete Memory
```bash
python slm_cli.py delete mem_abc123
```

### View Folders
```bash
python slm_cli.py folders list
```

---

## 📁 Folder Organization

Your memories are automatically organized:

```
/
├── /character     ← Character info
│   ├── Alistair Fitzroy
│   ├── Sebastian Carlisle
│   └── Evelyn Whitmore
├── /plot          ← Story events
│   ├── French postcard scheme
│   └── Captivity arc
├── /setting       ← Locations/time
│   └── Victorian Edinburgh
├── /theme         ← Themes
│   └── Medical ethics
├── /style         ← Preferences
├── /fact          ← General info
└── /general       ← Uncategorized
```

---

## 🔄 Understanding Metadata Loops

### What They Do
Instead of sending ALL your memories (expensive!), the system:

1. **Compresses** each memory to keywords + summary (~15 tokens)
2. **Scores** relevance to your current question
3. **Injects** only what's needed:
   - High relevance (>60%): Full content
   - Medium (30-60%): Just metadata
   - Low (<30%): Skipped

### Visual Example
```
Your Question: "Write a scene with Alistair"

FULL CONTEXT METHOD (739 tokens):
┌────────────────────────────────────────┐
│ [All 19 memories - full text]          │
│ [CHARACTER] Dr. Alistair Fitzroy is... │
│ [CHARACTER] Sebastian Carlisle was...  │
│ [PLOT] The French postcard scheme...   │
│ [THEME] Medical ethics versus...       │
│ ... (continues for all 19)             │
└────────────────────────────────────────┘

LOOP METHOD (110 tokens):
┌────────────────────────────────────────┐
│ [RELEVANT - Full text]                 │
│ • Alistair: professor, manipulator...  │
│ • Sebastian: captive, blood disorder   │
│                                        │
│ [RELATED - Keywords only]              │
│ • plot: captivity, drugs               │
│ • setting: laboratory, Edinburgh       │
└────────────────────────────────────────┘

SAVINGS: 85% fewer tokens!
```

---

## 💰 Cost Guide

### Per Message (typical chat)
| Component | Cost |
|-----------|------|
| GPT-4o response | ~$0.004 |
| Memory extraction | ~$0.003 |
| **Total** | **~$0.007** |

### Per 1000 Words Generated
| Type | Cost |
|------|------|
| Short replies | ~$0.01 |
| Medium scenes | ~$0.02 |
| Long chapters | ~$0.10 |

### Monthly Estimate
| Usage | Messages | Cost |
|-------|----------|------|
| Light | 100/month | ~$0.70 |
| Medium | 500/month | ~$3.50 |
| Heavy | 2000/month | ~$14.00 |

---

## 🚀 Workflow Examples

### Example 1: Starting a New Novel

```
1. Clear old memories (sidebar → Clear ALL)
2. Start describing your world:
   "My novel is set in 1880s London, featuring a secret society 
    called the Red Rose Order that conducts unethical medical experiments"
3. Add characters one by one:
   "The protagonist is Evelyn Whitmore, a female medical student 
    fighting against the establishment"
4. The AI extracts and remembers everything
5. Start writing scenes!
```

### Example 2: Continuing a Writing Session

```
1. Open the app (memories are already there)
2. Ask: "What were we working on?"
3. AI recalls from memory: "We were developing the confrontation 
   between Alistair and Sebastian..."
4. Continue: "Write the next scene"
```

### Example 3: Adding Plot Points

```
You: I decided that Sebastian will escape in chapter 5 by 
     stealing Alistair's keys during the injection

AI: That's a great twist! The irony of Alistair's medical 
    precision being his undoing...
    
    🧠 2 memories extracted
    
[Later...]

You: Write the escape scene

AI: [Uses the plot point you added]
```

---

## ❓ Troubleshooting

### "No memories found"
- Check if Cross-Session Memory is ON
- Try: `python slm_cli.py stats` to see if Mnemo is connected

### "High token costs"
- Make sure Metadata Loops is ON
- Check sidebar for token count

### "AI doesn't remember"
- Memories take a moment to sync
- Click Refresh in "View stored memories"
- Try being more specific in your question

### "Connection error"
- Check internet connection
- Verify API keys in sidebar
- Try: `python test_extraction.py` to test APIs

---

## 📱 Quick Reference Card

```
┌─────────────────────────────────────────────────┐
│           4o with Memory - Quick Ref            │
├─────────────────────────────────────────────────┤
│ START:        streamlit run app.py              │
│ ADD MEMORY:   Type naturally or use sidebar     │
│ VIEW MEMORY:  Sidebar → View stored memories    │
│ DELETE ONE:   Click 🗑️ next to memory           │
│ DELETE ALL:   Sidebar → Clear ALL Memories      │
│ SAVE TOKENS:  Enable 🔄 Metadata Loops          │
│ AUTO-SAVE:    Enable Auto-Extract Memories      │
├─────────────────────────────────────────────────┤
│ CLI COMMANDS:                                   │
│   python slm_cli.py stats                       │
│   python slm_cli.py list                        │
│   python slm_cli.py search "query"              │
│   python slm_cli.py add "content" -c category   │
│   python slm_cli.py delete memory_id            │
├─────────────────────────────────────────────────┤
│ CATEGORIES: character, plot, setting,           │
│             theme, style, fact, general         │
├─────────────────────────────────────────────────┤
│ COST: ~$0.007/message | ~$0.02/1000 words       │
└─────────────────────────────────────────────────┘
```

---

## 🎉 You're Ready!

1. Run `streamlit run app.py`
2. Enable all three toggles (Cross-Session, Auto-Extract, Loops)
3. Start chatting about your novel
4. Watch as GPT-4o remembers everything!

Happy writing! 🧠✨

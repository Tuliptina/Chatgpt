"""
4o with Memory - Enhanced Edition

Features:
- GPT-4o via OpenRouter with warm, conversational style
- Mnemo v4 Cloud Memory (persistent across sessions)
- Metadata Loop System (80% token savings)
- Auto-Memory Extraction
- NEW: Session Management (New Chat, Previous Sessions)
- NEW: File Upload with Memory Extraction
"""

import streamlit as st
import requests
import json
import time
import os
import uuid
from datetime import datetime
from memory import MnemoMemoryManager
from metadata_loops import LoopManager, LoopConfig
from smart_memory import SmartMemory, ContextWindowManager
from persistent_storage import PersistentStorage
from context_engine import ContextEngine, store_with_context

# ============================================================================
# CONFIGURATION
# ============================================================================

# API Keys - Load from Streamlit Secrets (NEVER hardcode!)
def get_secret(key, default=""):
    """Get secret from Streamlit Cloud or environment variable"""
    # Try Streamlit secrets first (for cloud deployment)
    try:
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    # Try environment variable (for local development)
    return os.environ.get(key, default)

# These will be empty in the code - set in Streamlit Cloud Secrets UI
DEFAULT_OPENROUTER_KEY = get_secret("OPENROUTER_KEY", "")
DEFAULT_HF_KEY = get_secret("HF_KEY", "")
MNEMO_URL = "https://athelaperk-mnemo-mcp.hf.space"

# Model configuration
MODEL_ID = "openai/gpt-4o-2024-11-20"
TEMPERATURE = 0.75

# Context window settings (GPT-4o: 128K context, 16K max output)
MAX_CONVERSATION_MESSAGES = 8  # Number of recent messages to include in context
MAX_SESSIONS_STORED = 20       # Number of previous sessions to keep

# System prompt
SYSTEM_PROMPT = """You are a warm, intelligent AI companion with persistent memory. You remember past conversations and uploaded context.

Your personality:
- Warm and direct, like a knowledgeable collaborator
- Excellent memory for details people share
- Ask thoughtful follow-up questions when needed
- Natural, conversational tone (not formal or robotic)

IMPORTANT — Match your response style to what the user is asking:

When the user asks you to RECALL, RETRIEVE, LIST, or SUMMARIZE information:
- Give clean, factual answers drawn from your memory context
- Use the exact details stored in memory — do NOT embellish, infer, or fill gaps with imagination
- If memory doesn't contain something, say so honestly instead of inventing details
- Structure clearly: bullet points or short paragraphs, no dramatic prose

When the user asks you to WRITE creatively (scenes, chapters, dialogue, prose):
- Match the genre, tone, and atmosphere of the project (from memory context)
- Deep psychological complexity in characters
- Setting-accurate language (historical, futuristic, contemporary — whatever the project requires)
- Show don't tell
- Default to third person past tense unless the user specifies otherwise or memory contains a different VOICE preference
- Match the user's writing voice if style samples are in memory
- Stay consistent with all CHARACTER, PLOT, CONTEXT, and INSTRUCTION memories

When in doubt about which mode: ask the user.

Always acknowledge context from memory naturally."""

# ============================================================================
# SESSION MANAGEMENT (Persistent via Mnemo)
# ============================================================================

def get_persistent_storage(hf_key=None):
    """Get or create persistent storage instance"""
    # Use provided key or get from secrets
    key = hf_key or DEFAULT_HF_KEY
    
    # If key changed or not initialized, recreate
    if "persistent_storage" not in st.session_state or st.session_state.get("_ps_key") != key:
        if key:  # Only create if we have a key
            st.session_state.persistent_storage = PersistentStorage(
                hf_key=key,
                mnemo_url=MNEMO_URL
            )
            st.session_state._ps_key = key
    
    return st.session_state.get("persistent_storage")

def generate_session_id():
    """Generate unique session ID"""
    return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

def get_session_title(messages):
    """Generate intelligent title from first user message"""
    for msg in messages:
        if msg["role"] == "user":
            content = msg["content"].strip()
            
            # Remove common prefixes
            prefixes_to_remove = ["can you", "could you", "please", "help me", "i want to", "i need to", "let's"]
            content_lower = content.lower()
            for prefix in prefixes_to_remove:
                if content_lower.startswith(prefix):
                    content = content[len(prefix):].strip()
                    break
            
            # Capitalize first letter
            if content:
                content = content[0].upper() + content[1:] if len(content) > 1 else content.upper()
            
            # Truncate smartly (at word boundary)
            if len(content) > 40:
                # Find last space before 40 chars
                last_space = content[:40].rfind(' ')
                if last_space > 20:
                    content = content[:last_space] + "..."
                else:
                    content = content[:40] + "..."
            
            return content if content else "New Chat"
    return "New Chat"

def save_current_session():
    """Save current session to PERSISTENT storage (survives browser refresh)"""
    if "messages" not in st.session_state or not st.session_state.messages:
        return
    
    session_id = st.session_state.get("current_session_id", generate_session_id())
    
    # Use manually set title if available, otherwise auto-generate
    custom_titles = st.session_state.get("custom_titles", {})
    if session_id in custom_titles:
        title = custom_titles[session_id]
    else:
        title = get_session_title(st.session_state.messages)
    
    messages_copy = [msg.copy() for msg in st.session_state.messages]  # Deep copy
    
    # Save to persistent storage (Mnemo cloud)
    storage = get_persistent_storage()
    if storage:
        try:
            storage.save_session(
                session_id=session_id,
                title=title,
                messages=messages_copy,
                timestamp=datetime.now().isoformat()
            )
            # Periodically clean up stale duplicate session entries
            msg_count = len([m for m in messages_copy if m.get("role") == "user"])
            if msg_count > 0 and msg_count % 25 == 0:
                storage.cleanup_stale_sessions()
        except Exception as e:
            pass  # Continue even if storage fails
    
    # Also update local session state WITH MESSAGES
    if "session_history" not in st.session_state:
        st.session_state.session_history = []
    
    current_session = {
        "id": session_id,
        "title": title,
        "timestamp": datetime.now().isoformat(),
        "message_count": len([m for m in messages_copy if m["role"] == "user"]),
        "preview": messages_copy[0]["content"][:100] if messages_copy else "",
        "messages": messages_copy  # INCLUDE MESSAGES!
    }
    
    # Update existing or add new
    existing_idx = None
    for i, session in enumerate(st.session_state.session_history):
        if session["id"] == current_session["id"]:
            existing_idx = i
            break
    
    if existing_idx is not None:
        st.session_state.session_history[existing_idx] = current_session
    else:
        st.session_state.session_history.insert(0, current_session)
    
    st.session_state.session_history = st.session_state.session_history[:MAX_SESSIONS_STORED]

def load_sessions_from_storage(hf_key=None):
    """Load session history from persistent storage"""
    storage = get_persistent_storage(hf_key)
    if not storage:
        return []
    sessions = storage.load_sessions(limit=MAX_SESSIONS_STORED)
    st.session_state.session_history = sessions
    return sessions

def start_new_chat():
    """Start a new chat session"""
    # Save current session first (to persistent storage)
    save_current_session()
    
    # Clear messages but keep memory
    st.session_state.messages = []
    st.session_state.current_session_id = generate_session_id()
    
    # Keep memory manager and loop manager (cross-session memory persists!)

def load_session(session_id):
    """Load a previous session with its messages"""
    save_current_session()  # Save current first
    
    for session in st.session_state.get("session_history", []):
        if session["id"] == session_id:
            # Load messages from session (if available)
            messages = session.get("messages", [])
            if messages:
                # Deep copy the messages
                st.session_state.messages = [msg.copy() for msg in messages]
            else:
                st.session_state.messages = []
            st.session_state.current_session_id = session_id
            return  # Found and loaded
    
    # Session not found in local state - this shouldn't happen normally
    st.session_state.messages = []
    st.session_state.current_session_id = session_id

def delete_session(session_id):
    """Delete a session from history"""
    # Delete from local state
    st.session_state.session_history = [
        s for s in st.session_state.get("session_history", [])
        if s["id"] != session_id
    ]
    # Also delete from persistent storage
    try:
        storage = get_persistent_storage()
        if storage:
            storage.delete_session(session_id)
    except Exception:
        pass

# ============================================================================
# FILE PROCESSING
# ============================================================================

def extract_text_from_file(uploaded_file):
    """Extract text content from uploaded file"""
    file_type = uploaded_file.type
    content = ""
    
    try:
        if file_type == "text/plain":
            # Plain text
            content = uploaded_file.read().decode("utf-8")
        
        elif file_type == "text/csv":
            # CSV - read as text
            content = uploaded_file.read().decode("utf-8")
        
        elif file_type == "application/json":
            # JSON
            data = json.load(uploaded_file)
            content = json.dumps(data, indent=2)
        
        elif file_type == "text/markdown":
            # Markdown
            content = uploaded_file.read().decode("utf-8")
        
        elif "pdf" in file_type:
            # PDF - try to extract
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(uploaded_file)
                for page in reader.pages:
                    content += page.extract_text() + "\n"
            except ImportError:
                content = "[PDF support requires PyPDF2. Please upload as text file.]"
        
        elif "word" in file_type or "docx" in file_type:
            # Word doc
            try:
                import docx
                doc = docx.Document(uploaded_file)
                content = "\n".join([para.text for para in doc.paragraphs])
            except ImportError:
                content = "[Word support requires python-docx. Please upload as text file.]"
        
        else:
            # Try reading as text
            try:
                content = uploaded_file.read().decode("utf-8")
            except Exception:
                content = f"[Cannot read file type: {file_type}]"
    
    except Exception as e:
        content = f"[Error reading file: {str(e)}]"
    
    return content

def extract_memories_from_file(content, filename, openrouter_key, hf_key):
    """Use GPT-4o to extract memories with DEEP CONTEXT from file content.
    
    For large files, splits into overlapping chunks and extracts from each,
    then deduplicates. This ensures nothing is lost from long documents.
    """
    
    CHUNK_SIZE = 12000       # chars per chunk
    CHUNK_OVERLAP = 1500     # overlap to catch cross-boundary content
    MAX_CHUNKS = 5           # safety limit (5 chunks = ~60K chars)
    
    # Split into chunks if needed
    if len(content) <= CHUNK_SIZE:
        chunks = [content]
    else:
        chunks = []
        start = 0
        while start < len(content) and len(chunks) < MAX_CHUNKS:
            end = start + CHUNK_SIZE
            # Try to break at a paragraph boundary
            if end < len(content):
                newline_pos = content[end-200:end].rfind('\n\n')
                if newline_pos > 0:
                    end = end - 200 + newline_pos
            chunks.append(content[start:end])
            start = end - CHUNK_OVERLAP  # overlap
    
    all_memories = []
    total_cost = 0
    
    for i, chunk in enumerate(chunks):
        chunk_label = f" (part {i+1}/{len(chunks)})" if len(chunks) > 1 else ""
        
        prompt = f"""Analyze this document and extract ALL information in multiple layers.
Be EXHAUSTIVE — extract every character, plot point, relationship, rule, and detail. 
Do not summarize or generalize. Extract specific facts.

DOCUMENT: {filename}{chunk_label}

CONTENT:
{chunk}

Extract in these categories:

LAYER 1 - FACTS (specific details):
- CHARACTER: Name, age, role, traits, background, relationships, affiliations
- PLOT: Specific events, sequences, twists, reveals, story structure
- SETTING: Locations, time periods, institutions, environments
- THEME: Themes, motifs, philosophical questions explored
- FACT: Rules, data, specifications, organizational details

LAYER 2 - DEEP CONTEXT (meaning and connections):
- CONTEXT: Deeper meaning behind facts (e.g. "the character's exile was self-imposed guilt, not punishment")
- CLARIFICATION: Misinterpretations to avoid (e.g. "Character X is NOT a villain — they're morally grey")
- RELATIONSHIP: How elements connect across the story (e.g. "Character A's fear of failure drives their betrayal of Character B")
- INSTRUCTION: Writing rules to always follow (e.g. "Never reveal the twist before chapter 12", "Character Y always speaks in short sentences")

LAYER 3 - STYLE:
- PROSE_SAMPLE: Distinctive passages showing voice (max 250 chars each)
- DIALOGUE_SAMPLE: Character speech patterns
- VOICE: Narrative voice, POV, tense, atmosphere
- VOCABULARY: Distinctive terms and phrases

Return ONLY a JSON array. Extract AS MANY items as needed — do not limit yourself:
[
  {{"category": "CHARACTER", "content": "Example: John Mercer, mid-30s detective, haunted by a cold case, estranged from his daughter"}},
  {{"category": "CONTEXT", "content": "Example: John's obsession with the cold case is really about his guilt over failing to save his partner"}},
  {{"category": "RELATIONSHIP", "content": "Example: John's estrangement from his daughter mirrors his inability to let go of the past"}},
  {{"category": "INSTRUCTION", "content": "Example: The real killer is not revealed until the final act. Never have John drink alcohol — he's been sober for 5 years."}}
]

Be thorough. Extract EVERY named character, organization, plot point, and rule. Miss nothing."""

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openrouter_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "openai/gpt-4o-2024-11-20",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 4096
                },
                timeout=90
            )
            
            if response.status_code != 200:
                continue
            
            data = response.json()
            raw = data["choices"][0]["message"]["content"]
            
            # Calculate cost
            usage = data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            total_cost += (input_tokens * 2.50 + output_tokens * 15.00) / 1_000_000
            
            # Parse JSON
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            
            chunk_memories = json.loads(raw.strip())
            all_memories.extend(chunk_memories)
            
        except Exception as e:
            st.warning(f"Extraction error on chunk {i+1}: {str(e)}")
            continue
    
    # Deduplicate by content similarity (exact match on first 80 chars)
    seen = set()
    unique_memories = []
    for mem in all_memories:
        key = mem.get("content", "")[:80].lower().strip()
        if key not in seen:
            seen.add(key)
            unique_memories.append(mem)
    
    return unique_memories, total_cost

def store_file_memories(memories, hf_key, session_id=None):
    """Store extracted memories in Mnemo with session tracking"""
    stored = 0
    headers = {
        "Authorization": f"Bearer {hf_key}",
        "Content-Type": "application/json"
    }
    
    for mem in memories:
        try:
            category = mem.get("category", "FACT").upper()
            content = mem.get("content", "")
            
            if not content:
                continue
            
            # Include session_id for proper deletion later
            metadata = {"category": category, "source": "file_upload"}
            if session_id:
                metadata["session_id"] = session_id
            
            response = requests.post(
                f"{MNEMO_URL}/add",
                headers=headers,
                json={
                    "content": f"[{category}] {content}",
                    "metadata": metadata
                },
                timeout=10
            )
            
            if response.status_code == 200:
                stored += 1
        except Exception:
            continue
    
    return stored

# ============================================================================
# MEMORY EXTRACTION (from conversations)
# ============================================================================

def extract_memories_with_gpt(conversation, openrouter_key):
    """Extract memories from conversation using GPT-4o"""
    prompt = f"""Analyze this conversation and extract important facts to remember.

CONVERSATION:
{conversation}

Categories:
- CHARACTER: Names, traits, relationships
- PLOT: Events, story points
- SETTING: Locations, time periods
- THEME: Themes, symbols
- STYLE: Writing preferences
- FACT: Other important info

Return ONLY JSON array:
[{{"category": "CATEGORY", "content": "fact"}}]

If nothing important, return: []"""

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openrouter_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "openai/gpt-4o-2024-11-20",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 500
            },
            timeout=30
        )
        
        if response.status_code != 200:
            return [], 0
        
        data = response.json()
        raw = data["choices"][0]["message"]["content"]
        
        usage = data.get("usage", {})
        # GPT-4o-2024-11-20 pricing: $2.5/M input, $15/M output
        cost = (usage.get("prompt_tokens", 0) * 2.50 + usage.get("completion_tokens", 0) * 15.00) / 1_000_000
        
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        
        memories = json.loads(raw.strip())
        return memories, cost
        
    except Exception:
        return [], 0

def store_memories_in_mnemo(memories, hf_key, session_id=None):
    """Store memories in Mnemo with session tracking for proper deletion"""
    stored = 0
    headers = {
        "Authorization": f"Bearer {hf_key}",
        "Content-Type": "application/json"
    }
    
    for mem in memories:
        try:
            category = mem.get("category", "FACT").upper()
            content = mem.get("content", "")
            
            if not content:
                continue
            
            # Include session_id in metadata for proper deletion later
            metadata = {"category": category}
            if session_id:
                metadata["session_id"] = session_id
            
            response = requests.post(
                f"{MNEMO_URL}/add",
                headers=headers,
                json={
                    "content": f"[{category}] {content}",
                    "metadata": metadata
                },
                timeout=10
            )
            
            if response.status_code == 200:
                stored += 1
        except Exception:
            continue
    
    return stored

# ============================================================================
# API CALLS
# ============================================================================

def call_openrouter(messages, api_key):
    """Call OpenRouter API"""
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL_ID,
                "messages": messages,
                "temperature": TEMPERATURE,
                "max_tokens": 4000
            },
            timeout=120
        )
        
        if response.status_code != 200:
            return None, 0, 0, f"API Error: {response.status_code}"
        
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        
        return content, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), None
        
    except Exception as e:
        return None, 0, 0, str(e)

# ============================================================================
# COST TRACKING
# ============================================================================

class CostTracker:
    """Track API costs"""
    
    # GPT-4o pricing
    INPUT_COST = 2.50 / 1_000_000
    OUTPUT_COST = 15.00 / 1_000_000
    
    def __init__(self):
        if "total_cost" not in st.session_state:
            st.session_state.total_cost = 0.0
        if "message_count" not in st.session_state:
            st.session_state.message_count = 0
    
    def add_usage(self, input_tokens, output_tokens):
        cost = (input_tokens * self.INPUT_COST) + (output_tokens * self.OUTPUT_COST)
        st.session_state.total_cost += cost
        st.session_state.message_count += 1
        return cost

# ============================================================================
# MNEMO HELPER FUNCTIONS
# ============================================================================

def get_mnemo_stats(hf_key):
    """Get Mnemo statistics"""
    try:
        response = requests.get(
            f"{MNEMO_URL}/stats",
            headers={"Authorization": f"Bearer {hf_key}"},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return {}
    except Exception:
        return {}

def list_mnemo_memories(hf_key, limit=10):
    """List memories from Mnemo"""
    try:
        response = requests.get(
            f"{MNEMO_URL}/list",
            headers={"Authorization": f"Bearer {hf_key}"},
            timeout=10
        )
        if response.status_code == 200:
            memories = response.json().get("memories", [])
            return memories[:limit]
        return []
    except Exception:
        return []

def delete_memory(memory_id, hf_key):
    """Delete a memory from Mnemo"""
    try:
        response = requests.delete(
            f"{MNEMO_URL}/delete/{memory_id}",
            headers={"Authorization": f"Bearer {hf_key}"},
            timeout=10
        )
        return response.status_code == 200
    except Exception:
        return False

def clear_all_mnemo_memories(hf_key):
    """Clear all memories from Mnemo"""
    try:
        response = requests.post(
            f"{MNEMO_URL}/clear",
            headers={
                "Authorization": f"Bearer {hf_key}",
                "Content-Type": "application/json"
            },
            json={"confirm": True},
            timeout=30
        )
        return response.status_code == 200
    except Exception:
        return False

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="4o with Memory",
        page_icon="🧠",
        layout="wide"
    )
    
    # Title
    st.title("🧠 4o with Memory")
    st.caption("GPT-4o with warm, conversational style and persistent memory")
    
    # ========================================================================
    # SECRETS CHECK
    # ========================================================================
    
    # Check if secrets are configured
    if not DEFAULT_OPENROUTER_KEY or not DEFAULT_HF_KEY:
        st.error("⚠️ **API Keys Not Configured!**")
        st.markdown("""
        **For Streamlit Cloud:**
        1. Go to your app's Settings → Secrets
        2. Add your keys in TOML format:
        ```toml
        OPENROUTER_KEY = "sk-or-v1-your-key"
        HF_KEY = "hf_your-token"
        ```
        
        **For Local Development:**
        1. Create `.streamlit/secrets.toml`
        2. Add your keys (see `secrets.toml.example`)
        
        [Get OpenRouter Key](https://openrouter.ai/keys) | [Get HuggingFace Token](https://huggingface.co/settings/tokens)
        """)
        st.stop()
    
    # ========================================================================
    # EARLY INITIALIZATION (must happen before sidebar)
    # ========================================================================
    
    # Load session history from persistent storage FIRST (so sidebar can show it)
    if "session_history_loaded" not in st.session_state:
        st.session_state.session_history_loaded = True
        st.session_state.session_history = []
        try:
            storage = get_persistent_storage(DEFAULT_HF_KEY)
            if storage:
                sessions = storage.load_sessions(limit=MAX_SESSIONS_STORED)
                if sessions:
                    st.session_state.session_history = sessions
        except Exception:
            pass
    
    # Initialize session ID
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = generate_session_id()
    
    # Initialize custom titles tracker (for manual renames that survive auto-title)
    if "custom_titles" not in st.session_state:
        st.session_state.custom_titles = {}
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # ========================================================================
    # SIDEBAR
    # ========================================================================
    
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # API Keys
        with st.expander("🔑 API Keys", expanded=False):
            openrouter_key = st.text_input(
                "OpenRouter API Key",
                value=DEFAULT_OPENROUTER_KEY,
                type="password"
            )
            hf_key = st.text_input(
                "HuggingFace Token",
                value=DEFAULT_HF_KEY,
                type="password"
            )
        
        # Use provided keys or defaults
        openrouter_key = openrouter_key or DEFAULT_OPENROUTER_KEY
        hf_key = hf_key or DEFAULT_HF_KEY
        
        st.divider()
        
        # ====================================================================
        # NEW CHAT BUTTON
        # ====================================================================
        
        st.subheader("💬 Chat")
        
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            start_new_chat()
            st.rerun()
        
        st.divider()
        
        # ====================================================================
        # PREVIOUS SESSIONS WITH FOLDERS
        # ====================================================================
        
        col_title, col_refresh = st.columns([3, 1])
        with col_title:
            st.subheader("📚 Sessions")
        with col_refresh:
            if st.button("🔄", key="refresh_sessions", help="Refresh from cloud"):
                try:
                    storage = PersistentStorage(hf_key=hf_key, mnemo_url=MNEMO_URL)
                    sessions = storage.load_sessions(limit=MAX_SESSIONS_STORED)
                    st.session_state.session_history = sessions
                    st.rerun()
                except Exception:
                    pass
        
        # Initialize folders
        if "session_folders" not in st.session_state:
            st.session_state.session_folders = {"📁 Default": []}
        
        # Folder management
        with st.expander("📂 Manage Folders", expanded=False):
            new_folder = st.text_input("New folder name", key="new_folder_input", placeholder="e.g., Story Ideas")
            if st.button("➕ Create Folder", key="create_folder"):
                if new_folder and new_folder.strip():
                    folder_name = f"📁 {new_folder.strip()}"
                    if folder_name not in st.session_state.session_folders:
                        st.session_state.session_folders[folder_name] = []
                        st.success(f"Created {folder_name}")
                        st.rerun()
            
            # Show existing folders for deletion
            folders = list(st.session_state.session_folders.keys())
            if len(folders) > 1:
                folder_to_delete = st.selectbox("Delete folder", [""] + [f for f in folders if f != "📁 Default"], key="del_folder")
                if folder_to_delete and st.button("🗑️ Delete Folder", key="delete_folder_btn"):
                    # Move sessions back to default
                    st.session_state.session_folders["📁 Default"].extend(
                        st.session_state.session_folders.get(folder_to_delete, [])
                    )
                    del st.session_state.session_folders[folder_to_delete]
                    st.rerun()
        
        sessions = st.session_state.get("session_history", [])
        
        # Create scrollable container for sessions
        with st.container():
            if sessions:
                # Get folder assignments
                session_to_folder = {}
                for folder, session_ids in st.session_state.session_folders.items():
                    for sid in session_ids:
                        session_to_folder[sid] = folder
                
                # Group sessions by folder
                folders_with_sessions = {"📁 Default": []}
                for folder in st.session_state.session_folders:
                    if folder not in folders_with_sessions:
                        folders_with_sessions[folder] = []
                
                for session in sessions:
                    sid = session.get("id", "")
                    folder = session_to_folder.get(sid, "📁 Default")
                    if folder not in folders_with_sessions:
                        folder = "📁 Default"
                    folders_with_sessions[folder].append(session)
                
                # Display by folder
                for folder, folder_sessions in folders_with_sessions.items():
                    if folder_sessions:
                        st.caption(folder)
                        for session in folder_sessions[:10]:
                            session_id = session.get("id", "")
                            title = session.get("title", "Untitled")[:30]
                            msg_count = session.get("message_count", 0)
                            
                            col1, col2, col3, col4 = st.columns([6, 1, 1, 1])
                            
                            with col1:
                                if st.button(
                                    f"💬 {title}",
                                    key=f"load_{session_id}",
                                    use_container_width=True
                                ):
                                    load_session(session_id)
                                    st.rerun()
                            
                            with col2:
                                # Rename button
                                if st.button("✏️", key=f"rename_{session_id}", help="Rename"):
                                    st.session_state.renaming_session = session_id
                                    st.rerun()
                            
                            with col3:
                                # Move to folder
                                if st.button("📂", key=f"move_{session_id}", help="Move to folder"):
                                    st.session_state.moving_session = session_id
                                    st.rerun()
                            
                            with col4:
                                if st.button("🗑️", key=f"del_{session_id}"):
                                    delete_session(session_id)
                                    st.rerun()
                        
                        st.caption("")  # Spacing
                
                # Rename dialog
                if st.session_state.get("renaming_session"):
                    sid = st.session_state.renaming_session
                    
                    # Get current title for placeholder
                    current_title = "New Chat"
                    for s in st.session_state.session_history:
                        if s.get("id") == sid:
                            current_title = s.get("title", "New Chat")
                            break
                    
                    with st.container():
                        st.markdown("---")
                        new_name = st.text_input("New name:", value=current_title, key="rename_input")
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("💾 Save", key="save_rename", use_container_width=True):
                                if new_name and new_name.strip():
                                    # Store custom title in persistent dict
                                    if "custom_titles" not in st.session_state:
                                        st.session_state.custom_titles = {}
                                    st.session_state.custom_titles[sid] = new_name.strip()
                                    
                                    # Update session history in RAM
                                    for s in st.session_state.session_history:
                                        if s.get("id") == sid:
                                            s["title"] = new_name.strip()
                                            break
                                    
                                    # Persist to Mnemo immediately
                                    save_current_session()
                                
                                st.session_state.renaming_session = None
                                st.rerun()
                        with col2:
                            if st.button("❌ Cancel", key="cancel_rename", use_container_width=True):
                                st.session_state.renaming_session = None
                                st.rerun()
                
                # Move to folder dialog
                if st.session_state.get("moving_session"):
                    sid = st.session_state.moving_session
                    with st.container():
                        st.markdown("---")
                        folders = list(st.session_state.session_folders.keys())
                        target_folder = st.selectbox("Move to:", folders, key="move_target")
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("📂 Move", key="confirm_move"):
                                # Remove from all folders first
                                for f in st.session_state.session_folders:
                                    if sid in st.session_state.session_folders[f]:
                                        st.session_state.session_folders[f].remove(sid)
                                # Add to target folder
                                st.session_state.session_folders[target_folder].append(sid)
                                st.session_state.moving_session = None
                                st.rerun()
                        with col2:
                            if st.button("❌ Cancel", key="cancel_move"):
                                st.session_state.moving_session = None
                                st.rerun()
            else:
                st.caption("No previous sessions")
        
        st.divider()
        
        # ====================================================================
        # FILE UPLOAD
        # ====================================================================
        # SETTINGS (Upload Files → Deep Context + Memory Extraction)
        # ====================================================================
        
        with st.expander("⚙️ Settings", expanded=False):
            
            st.markdown("**📎 Upload File → Memory**")
            st.caption("Upload a file — extracts facts, deep context, style, and saves everything to memory loops in one pass")
            
            uploaded_file = st.file_uploader(
                "Upload file",
                type=["txt", "md", "csv", "json", "pdf", "docx"],
                help="Supports text, markdown, CSV, JSON, PDF, DOCX",
                label_visibility="collapsed"
            )
            
            if uploaded_file is not None:
                if st.button("🧠 Extract Deep Context + Memories", use_container_width=True):
                    with st.spinner("Reading file..."):
                        content = extract_text_from_file(uploaded_file)
                    
                    if content and not content.startswith("["):
                        # Show file size info
                        n_chunks = max(1, (len(content) - 1) // 12000 + 1)
                        if n_chunks > 1:
                            st.info(f"📄 {len(content):,} chars → splitting into {min(n_chunks, 5)} chunks for thorough extraction")
                        
                        with st.spinner(f"Extracting deep context & memories ({min(n_chunks, 5)} API call{'s' if n_chunks > 1 else ''})..."):
                            memories, cost = extract_memories_from_file(
                                content, 
                                uploaded_file.name,
                                openrouter_key,
                                hf_key
                            )
                        
                        if memories:
                            with st.spinner("Storing to memory loops..."):
                                current_session = st.session_state.get("current_session_id")
                                stored = store_file_memories(memories, hf_key, session_id=current_session)
                            
                            # Count by layer
                            facts = [m for m in memories if m.get("category") in ("CHARACTER", "PLOT", "SETTING", "THEME", "FACT")]
                            context = [m for m in memories if m.get("category") in ("CONTEXT", "CLARIFICATION", "RELATIONSHIP", "INSTRUCTION")]
                            style = [m for m in memories if m.get("category") in ("PROSE_SAMPLE", "DIALOGUE_SAMPLE", "VOICE", "VOCABULARY")]
                            
                            st.success(f"✅ Stored {stored} memories")
                            st.caption(f"📊 {len(facts)} facts · {len(context)} deep context · {len(style)} style | Cost: ${cost:.4f}")
                            
                            # Show what was extracted
                            with st.expander("View extracted memories", expanded=False):
                                for mem in memories:
                                    cat = mem.get("category", "FACT")
                                    txt = mem.get("content", "")[:150]
                                    st.caption(f"**[{cat}]** {txt}")
                            
                            # Reload loop manager
                            if "loop_manager" in st.session_state:
                                st.session_state.loop_manager.load_from_mnemo()
                        else:
                            st.warning("No memories extracted. Try a different file.")
                    else:
                        st.error("Could not read file content")
        
        st.divider()
        
        # ====================================================================
        # MEMORY SETTINGS
        # ====================================================================
        
        st.subheader("🧠 Memory Settings")
        
        cross_session_enabled = st.toggle(
            "Cross-Session Memory",
            value=True,
            help="Remember across chat sessions"
        )
        
        auto_extract = st.toggle(
            "Auto-Extract Memories",
            value=True,
            help="Automatically extract facts from conversations"
        )
        st.session_state.auto_extract = auto_extract
        
        use_loops = st.toggle(
            "🔄 Metadata Loops (Save 80% tokens)",
            value=True,
            help="Use token-efficient context injection"
        )
        st.session_state.use_loops = use_loops
        
        # Show stats
        if "loop_manager" in st.session_state and use_loops:
            loop_stats = st.session_state.loop_manager.get_stats()
            st.caption(f"📊 {loop_stats['total_items']} memories | {loop_stats['total_metadata_tokens']} tokens")
        
        st.divider()
        
        # ====================================================================
        # MANUAL MEMORY
        # ====================================================================
        
        st.subheader("📝 Add Memory")
        
        with st.expander("Add manually", expanded=False):
            memory_category = st.selectbox(
                "Category",
                ["CHARACTER", "PLOT", "SETTING", "THEME", "STYLE", "FACT"]
            )
            memory_content = st.text_area(
                "Content",
                placeholder="e.g., Detective Mercer has a fear of water since childhood",
                height=80
            )
            if st.button("💾 Save", use_container_width=True):
                if memory_content.strip():
                    current_session = st.session_state.get("current_session_id")
                    stored = store_memories_in_mnemo(
                        [{"category": memory_category, "content": memory_content}],
                        hf_key,
                        session_id=current_session
                    )
                    if stored:
                        st.success(f"✅ Saved [{memory_category}]")
                        if "loop_manager" in st.session_state:
                            st.session_state.loop_manager.add_to_loop(
                                memory_content, memory_category.lower()
                            )
                    else:
                        st.error("Failed to save")
        
        # ====================================================================
        # VIEW MEMORIES
        # ====================================================================
        
        with st.expander("View memories", expanded=False):
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🔄 Refresh", use_container_width=True, key="refresh_mem"):
                    st.rerun()
            with col2:
                if st.button("📖 View All", use_container_width=True, key="view_all_mem"):
                    st.session_state.show_all_memories = True
                    st.rerun()
            
            stats = get_mnemo_stats(hf_key)
            st.caption(f"Total: {stats.get('total_memories', 0)} | Links: {stats.get('neural_links', 0)}")
            
            # Scrollable container for memories
            memories = list_mnemo_memories(hf_key, limit=15)
            
            # Create scrollable div
            st.markdown("""
            <style>
            .memory-scroll {
                max-height: 300px;
                overflow-y: auto;
                padding: 5px;
                border: 1px solid #333;
                border-radius: 5px;
            }
            </style>
            """, unsafe_allow_html=True)
            
            for mem in memories:
                col1, col2 = st.columns([5, 1])
                with col1:
                    content = mem.get("content", "")[:60]
                    st.caption(f"{content}...")
                with col2:
                    if st.button("🗑️", key=f"del_mem_{mem.get('id', '')}"):
                        if delete_memory(mem.get("id"), hf_key):
                            st.rerun()
            
            if len(memories) >= 15:
                st.caption("... showing first 15. Click 'View All' for complete list")
            
            st.markdown("---")
            
            if st.button("🧹 Clear ALL Memories", use_container_width=True):
                st.session_state.confirm_clear = True
            
            if st.session_state.get("confirm_clear"):
                st.warning("⚠️ Delete ALL memories?")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Yes, delete all"):
                        clear_all_mnemo_memories(hf_key)
                        st.session_state.confirm_clear = False
                        st.rerun()
                with col2:
                    if st.button("Cancel"):
                        st.session_state.confirm_clear = False
                        st.rerun()
        
        # Full-screen memory viewer modal
        if st.session_state.get("show_all_memories"):
            st.markdown("---")
            st.subheader("📖 All Memories")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button("❌ Close", use_container_width=True, key="close_all_mem"):
                    st.session_state.show_all_memories = False
                    st.rerun()
            with col2:
                search_query = st.text_input("🔍 Search", key="mem_search", placeholder="Filter memories...")
            with col3:
                category_filter = st.selectbox("Category", ["All", "CHARACTER", "PLOT", "SETTING", "THEME", "CONTEXT", "STYLE", "FACT"], key="cat_filter")
            
            # Get ALL memories
            all_memories = list_mnemo_memories(hf_key, limit=500)
            
            # Filter
            if search_query:
                all_memories = [m for m in all_memories if search_query.lower() in m.get("content", "").lower()]
            if category_filter != "All":
                all_memories = [m for m in all_memories if f"[{category_filter}]" in m.get("content", "")]
            
            st.caption(f"Showing {len(all_memories)} memories")
            
            # Scrollable container with all memories
            st.markdown("""
            <style>
            .full-memory-scroll {
                max-height: 500px;
                overflow-y: auto;
                padding: 10px;
                border: 1px solid #444;
                border-radius: 8px;
                background: #1a1a1a;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Display in scrollable area
            for mem in all_memories:
                content = mem.get("content", "")
                mem_id = mem.get("id", "")
                
                # Extract category
                category = "OTHER"
                if content.startswith("["):
                    category = content.split("]")[0][1:]
                
                col1, col2, col3 = st.columns([1, 8, 1])
                with col1:
                    st.caption(f"[{category}]")
                with col2:
                    # Show full content
                    st.text(content[len(f"[{category}]"):].strip()[:200])
                with col3:
                    if st.button("🗑️", key=f"del_full_{mem_id}"):
                        if delete_memory(mem_id, hf_key):
                            st.rerun()
        
        st.divider()
        
        # ====================================================================
        # MEMORY HEALTH
        # ====================================================================
        
        with st.expander("🏥 Memory Health", expanded=False):
            if "context_engine" in st.session_state and "loop_manager" in st.session_state:
                health = st.session_state.context_engine.degradation.get_health_report(
                    st.session_state.loop_manager
                )
                
                st.metric("Health", health['health_score'])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"📊 Items: {health['total_items']}")
                    st.caption(f"🔍 Low relevance: {health['low_relevance_items']}")
                with col2:
                    st.caption(f"😴 Never accessed: {health['never_accessed_items']}")
                    st.caption(f"🗑️ Prune candidates: {health['prune_candidates']}")
                
                if health['potential_duplicates'] > 0:
                    st.caption(f"👯 Duplicates: {health['potential_duplicates']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Run Decay", use_container_width=True):
                        result = st.session_state.context_engine.degradation.apply_decay(
                            st.session_state.loop_manager
                        )
                        if result.get("skipped"):
                            st.info(result.get("reason", "Skipped"))
                        else:
                            st.success(f"Decayed {result.get('decayed_items', 0)} items")
                with col2:
                    if st.button("🧹 Prune Stale", use_container_width=True):
                        result = st.session_state.context_engine.degradation.prune_memories(
                            st.session_state.loop_manager, dry_run=False
                        )
                        st.success(f"Pruned {result.get('pruned', 0)} items")
                        st.rerun()
                
                # Memory Consolidation Button
                st.markdown("---")
                st.markdown("**🧠 Memory Consolidation**")
                st.caption("Analyze memories & generate deep context")
                
                # Track last consolidation
                last_consol = st.session_state.get("last_consolidation")
                if last_consol:
                    st.caption(f"Last run: {last_consol[:16]}")
                
                if st.button("🧠 Consolidate Now", use_container_width=True, 
                            help="Analyze all memories and generate context/relationship entries"):
                    with st.spinner("Analyzing memories... (this may take 30-60 seconds)"):
                        result = st.session_state.context_engine.consolidate_memories(openrouter_key)
                        
                        if result.get("error"):
                            st.error(f"Error: {result['error']}")
                        else:
                            st.session_state.last_consolidation = result["timestamp"]
                            st.success(f"✅ Created {result['created']} new context entries!")
                            st.caption(f"Analyzed: {result['memories_analyzed']} memories")
                            st.caption(f"Cost: ${result['cost']:.4f}")
                            
                            if result.get("new_entries"):
                                with st.expander("New entries created"):
                                    for entry in result["new_entries"]:
                                        st.caption(f"[{entry['category']}] {entry['content']}")
                            
                            # Reload loop manager to include new entries
                            st.session_state.loop_manager.load_from_mnemo(use_smart_extraction=False)
        
        st.divider()
        
        # ====================================================================
        # COST TRACKING
        # ====================================================================
        
        st.subheader("💰 Costs")
        st.caption(f"Messages: {st.session_state.get('message_count', 0)}")
        st.caption(f"Total: ${st.session_state.get('total_cost', 0):.4f}")
    
    # ========================================================================
    # INITIALIZATION
    # ========================================================================
    
    # Initialize memory manager
    if "memory_manager" not in st.session_state:
        st.session_state.memory_manager = MnemoMemoryManager(
            openrouter_key=openrouter_key,
            hf_key=hf_key,
            user_id="default_user",
            cross_session_enabled=cross_session_enabled
        )
    
    # Initialize loop manager
    if "loop_manager" not in st.session_state:
        st.session_state.loop_manager = LoopManager(
            openrouter_key=openrouter_key,
            hf_key=hf_key,
            mnemo_url=MNEMO_URL
        )
        st.session_state.loop_manager.load_from_mnemo(use_smart_extraction=False)
    
    # Initialize smart memory (reduces latency)
    if "smart_memory" not in st.session_state:
        st.session_state.smart_memory = SmartMemory()
    
    # Initialize context engine (deep understanding + style + degradation)
    if "context_engine" not in st.session_state:
        st.session_state.context_engine = ContextEngine(
            hf_key=hf_key,
            openrouter_key=openrouter_key,
            mnemo_url=MNEMO_URL
        )
    
    # Initialize context manager WITH loop system integration
    if "context_manager" not in st.session_state:
        st.session_state.context_manager = ContextWindowManager(
            loop_manager=st.session_state.loop_manager
        )
    else:
        # Update loop manager reference if needed
        st.session_state.context_manager.set_loop_manager(st.session_state.loop_manager)
    
    # Run maintenance periodically (every 50 messages)
    msg_count = len(st.session_state.get("messages", []))
    if msg_count > 0 and msg_count % 50 == 0:
        if "last_maintenance" not in st.session_state or \
           st.session_state.last_maintenance != msg_count:
            st.session_state.context_engine.maintenance(
                st.session_state.loop_manager, 
                apply_decay=True, 
                prune=False
            )
            st.session_state.last_maintenance = msg_count
    
    # Update cross-session toggle
    st.session_state.memory_manager.toggle_cross_session(cross_session_enabled)
    
    # ========================================================================
    # CHAT INTERFACE
    # ========================================================================
    
    # Display chat history
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show metadata and copy button for assistant messages
            if message["role"] == "assistant":
                col1, col2 = st.columns([8, 1])
                
                with col1:
                    if "metadata" in message:
                        meta = message["metadata"]
                        memory_info = []
                        if meta.get("cross_session_memories_used", 0) > 0:
                            memory_info.append(f"📚 {meta['cross_session_memories_used']} memories")
                        if meta.get("mode") == "smart":
                            memory_info.append(f"🔄 {meta.get('context_tokens', 0)} tokens")
                        elif meta.get("mode") == "skip":
                            memory_info.append("⚡ fast")
                        if meta.get("extracted", 0) > 0:
                            memory_info.append(f"🧠 {meta['extracted']} extracted")
                        if meta.get("cost"):
                            memory_info.append(f"💰 ${meta['cost']:.4f}")
                        if memory_info:
                            st.caption(" | ".join(memory_info))
                
                with col2:
                    # Copy button - stores content for display
                    if st.button("📋", key=f"copy_{idx}", help="Copy response"):
                        st.session_state.show_copy_modal = idx
                        st.session_state.copy_content = message["content"]
    
    # Copy modal - shows copyable text
    if st.session_state.get("show_copy_modal") is not None:
        with st.container():
            st.markdown("---")
            st.markdown("**📋 Copy this text:**")
            st.code(st.session_state.copy_content, language=None)
            if st.button("✅ Done", key="close_copy"):
                st.session_state.show_copy_modal = None
                st.session_state.copy_content = None
                st.rerun()
    
    # Chat input
    if prompt := st.chat_input("What's on your mind?"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # SMART MEMORY: Check if this query needs memory lookup
                conversation_length = len(st.session_state.messages)
                needs_memory, memory_reason = st.session_state.smart_memory.should_use_memory(
                    prompt, conversation_length
                )
                
                # Search past conversations for context (cross-session memory)
                past_conversation_context = ""
                sessions_found = 0
                current_session_id = st.session_state.get("current_session_id", "")
                skip_loops_for_this_query = False  # Flag to skip loop system
                
                if needs_memory and st.session_state.get("use_loops", True):
                    try:
                        storage = get_persistent_storage()
                        if storage:
                            # Check if user is asking about past chats specifically
                            prompt_lower = prompt.lower()
                            asking_about_past = any(phrase in prompt_lower for phrase in [
                                "last chat", "previous chat", "last conversation", 
                                "previous conversation", "earlier chat", "before this",
                                "what did we talk", "what were we", "did we discuss",
                                "remember when", "last time", "our last", "previous session",
                                "talked about", "chatting about", "we discussed", "we were talking"
                            ])
                            
                            if asking_about_past:
                                # IMPORTANT: Skip loop system for past chat queries!
                                # Only use actual session content
                                skip_loops_for_this_query = True
                                
                                # Get actual session content (SKIP CURRENT SESSION!)
                                session_results = storage.search_sessions(
                                    prompt, 
                                    current_session_id=current_session_id,
                                    limit=2
                                )
                                if session_results:
                                    sessions_found = len(session_results)
                                    past_conversation_context = "\n\n[PREVIOUS CHAT SESSIONS - The user is asking about past conversations. Use ONLY this information to answer. Do NOT use other memories:]\n" + "\n---\n".join(session_results)
                                else:
                                    # Fallback to recent sessions (SKIP CURRENT!)
                                    recent = storage.get_previous_sessions_content(
                                        current_session_id=current_session_id,
                                        limit=2
                                    )
                                    if recent:
                                        sessions_found = len(recent)
                                        summaries = [f"Session '{s['title']}':\n{s['summary']}" for s in recent]
                                        past_conversation_context = "\n\n[RECENT CHAT SESSIONS - The user is asking about past conversations. Use ONLY this information to answer:]\n" + "\n---\n".join(summaries)
                            else:
                                # Regular memory search + DEEP CONTEXT
                                past_convos = storage.search_conversations(prompt, limit=3)
                                if past_convos:
                                    past_conversation_context = "\n\n[PAST CONVERSATIONS]\n" + "\n".join(
                                        f"• {conv[:200]}" for conv in past_convos
                                    )
                                
                                # Add enriched context (CONTEXT, CLARIFICATION entries)
                                if "context_engine" in st.session_state:
                                    enriched, enrich_meta = st.session_state.context_engine.build_rich_context(
                                        prompt, st.session_state.loop_manager
                                    )
                                    if enriched:
                                        past_conversation_context += f"\n\n[DEEP CONTEXT]\n{enriched}"
                    except Exception as e:
                        pass  # Silently fail
                
                # Build full system prompt
                full_system_prompt = SYSTEM_PROMPT
                
                # Add past conversation context
                full_system_prompt += past_conversation_context
                
                # Use INTEGRATED context manager with loop system
                # BUT skip loops if asking about past chats
                messages, context_stats = st.session_state.context_manager.build_optimized_context(
                    system_prompt=full_system_prompt,
                    query=prompt,
                    conversation_history=st.session_state.messages,
                    max_messages=MAX_CONVERSATION_MESSAGES,
                    use_loops=(needs_memory and st.session_state.get("use_loops", True) and not skip_loops_for_this_query)
                )
                
                # Context metadata for display
                context_meta = {
                    "cross_session_memories_used": context_stats.get("memory_items_full", 0) + context_stats.get("memory_items_meta", 0),
                    "context_tokens": context_stats.get("total_tokens", 0),
                    "memory_full_tokens": context_stats.get("memory_full_tokens", 0),
                    "memory_meta_tokens": context_stats.get("memory_meta_tokens", 0),
                    "mode": "smart" if needs_memory else "skip",
                    "memory_reason": memory_reason,
                    "within_budget": context_stats.get("within_budget", True),
                    "sessions_found": sessions_found  # Show if past sessions were found
                }
                
                # Call API
                response, input_tokens, output_tokens, error = call_openrouter(messages, openrouter_key)
                
                if error:
                    st.error(error)
                else:
                    # Track costs
                    cost_tracker = CostTracker()
                    msg_cost = cost_tracker.add_usage(input_tokens, output_tokens)
                    
                    # Update memory
                    st.session_state.memory_manager.process_turn(prompt, response)
                    
                    # Save conversation turn for cross-session memory
                    # BUT skip for meta-queries about past chats (avoid feedback loop!)
                    if not skip_loops_for_this_query:
                        try:
                            storage = get_persistent_storage()
                            storage.save_conversation_turn(
                                user_message=prompt,
                                assistant_response=response,
                                session_id=st.session_state.get("current_session_id")
                            )
                        except Exception:
                            pass
                    
                    # Auto-extract memories
                    # BUT skip for meta-queries about past chats (avoid feedback loop!)
                    extracted = 0
                    if st.session_state.get("auto_extract", True) and not skip_loops_for_this_query:
                        conversation = f"User: {prompt}\n\nAssistant: {response}"
                        memories, extract_cost = extract_memories_with_gpt(conversation, openrouter_key)
                        if memories:
                            current_session = st.session_state.get("current_session_id")
                            stored = store_memories_in_mnemo(memories, hf_key, session_id=current_session)
                            msg_cost += extract_cost
                            extracted = stored
                            
                            for mem in memories:
                                st.session_state.loop_manager.add_to_loop(
                                    content=mem.get("content", ""),
                                    category=mem.get("category", "general").lower()
                                )
                    
                    # Display response
                    st.markdown(response)
                    
                    # Show metadata
                    meta_parts = []
                    if context_meta.get("sessions_found", 0) > 0:
                        meta_parts.append(f"📜 {context_meta['sessions_found']} past chats")
                    if context_meta.get("cross_session_memories_used", 0) > 0:
                        meta_parts.append(f"📚 {context_meta['cross_session_memories_used']} memories")
                    if context_meta.get("mode") == "smart":
                        meta_parts.append(f"🔄 {context_meta.get('context_tokens', 0)} tokens")
                    elif context_meta.get("mode") == "skip":
                        meta_parts.append("⚡ fast")  # Skipped memory lookup
                    if extracted > 0:
                        meta_parts.append(f"🧠 {extracted} extracted")
                    meta_parts.append(f"💰 ${msg_cost:.4f}")
                    st.caption(" | ".join(meta_parts))
                    
                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "metadata": {
                            "cross_session_memories_used": context_meta.get("cross_session_memories_used", 0),
                            "context_tokens": context_meta.get("context_tokens", 0),
                            "mode": context_meta.get("mode", "full"),
                            "extracted": extracted,
                            "cost": msg_cost
                        }
                    })
                    
                    # Auto-save session
                    save_current_session()
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    
    st.divider()
    st.caption("🧠 4o with Memory | GPT-4o + Mnemo v4 + Metadata Loops")

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    main()

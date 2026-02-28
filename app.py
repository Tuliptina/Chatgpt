"""
4o with Memory - Enhanced Edition

Features:
- GPT-4o via OpenRouter with warm, conversational style
- Mnemo v4 Cloud Memory (persistent across sessions)
- Metadata Loop System (80% token savings)
- Auto-Memory Extraction
- NEW: Session Management (New Chat, Previous Sessions)
- NEW: File Upload with Memory Extraction (Async + pypdf)
- NEW: Native JSON Structured Outputs
"""

import streamlit as st
import requests
import httpx
import asyncio
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
    try:
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)

DEFAULT_OPENROUTER_KEY = get_secret("OPENROUTER_KEY", "")
DEFAULT_HF_KEY = get_secret("HF_KEY", "")
MNEMO_URL = "https://athelaperk-mnemo-mcp.hf.space"

MODEL_ID = "openai/gpt-4o-2024-11-20"
TEMPERATURE = 0.75

MAX_CONVERSATION_MESSAGES = 8
MAX_SESSIONS_STORED = 20

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
    key = hf_key or DEFAULT_HF_KEY
    if "persistent_storage" not in st.session_state or st.session_state.get("_ps_key") != key:
        if key:
            st.session_state.persistent_storage = PersistentStorage(
                hf_key=key,
                mnemo_url=MNEMO_URL
            )
            st.session_state._ps_key = key
    return st.session_state.get("persistent_storage")

def generate_session_id():
    return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

def get_session_title(messages):
    for msg in messages:
        if msg["role"] == "user":
            content = msg["content"].strip()
            prefixes_to_remove = ["can you", "could you", "please", "help me", "i want to", "i need to", "let's"]
            content_lower = content.lower()
            for prefix in prefixes_to_remove:
                if content_lower.startswith(prefix):
                    content = content[len(prefix):].strip()
                    break
            if content:
                content = content[0].upper() + content[1:] if len(content) > 1 else content.upper()
            if len(content) > 40:
                last_space = content[:40].rfind(' ')
                if last_space > 20:
                    content = content[:last_space] + "..."
                else:
                    content = content[:40] + "..."
            return content if content else "New Chat"
    return "New Chat"

def save_current_session():
    if "messages" not in st.session_state or not st.session_state.messages:
        return
    
    session_id = st.session_state.get("current_session_id", generate_session_id())
    
    custom_titles = st.session_state.get("custom_titles", {})
    if session_id in custom_titles:
        title = custom_titles[session_id]
    else:
        title = get_session_title(st.session_state.messages)
    
    messages_copy = [msg.copy() for msg in st.session_state.messages]
    
    storage = get_persistent_storage()
    if storage:
        try:
            storage.save_session(
                session_id=session_id,
                title=title,
                messages=messages_copy,
                timestamp=datetime.now().isoformat()
            )
            msg_count = len([m for m in messages_copy if m.get("role") == "user"])
            if msg_count > 0 and msg_count % 25 == 0:
                storage.cleanup_stale_sessions()
        except Exception:
            pass
    
    if "session_history" not in st.session_state:
        st.session_state.session_history = []
    
    current_session = {
        "id": session_id,
        "title": title,
        "timestamp": datetime.now().isoformat(),
        "message_count": len([m for m in messages_copy if m["role"] == "user"]),
        "preview": messages_copy[0]["content"][:100] if messages_copy else "",
        "messages": messages_copy
    }
    
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
    storage = get_persistent_storage(hf_key)
    if not storage:
        return []
    sessions = storage.load_sessions(limit=MAX_SESSIONS_STORED)
    st.session_state.session_history = sessions
    return sessions

def start_new_chat():
    save_current_session()
    st.session_state.messages = []
    st.session_state.current_session_id = generate_session_id()

def load_session(session_id):
    save_current_session()
    for session in st.session_state.get("session_history", []):
        if session["id"] == session_id:
            messages = session.get("messages", [])
            st.session_state.messages = [msg.copy() for msg in messages] if messages else []
            st.session_state.current_session_id = session_id
            return
    st.session_state.messages = []
    st.session_state.current_session_id = session_id

def delete_session(session_id):
    st.session_state.session_history = [
        s for s in st.session_state.get("session_history", [])
        if s["id"] != session_id
    ]
    try:
        storage = get_persistent_storage()
        if storage:
            storage.delete_session(session_id)
    except Exception:
        pass

# ============================================================================
# FILE PROCESSING (Async + pypdf)
# ============================================================================

def extract_text_from_file(uploaded_file):
    """Extract text content from uploaded file"""
    file_type = uploaded_file.type
    content = ""
    
    try:
        if file_type == "text/plain" or file_type == "text/csv" or file_type == "text/markdown":
            content = uploaded_file.read().decode("utf-8")
        elif file_type == "application/json":
            data = json.load(uploaded_file)
            content = json.dumps(data, indent=2)
        elif "pdf" in file_type:
            try:
                import pypdf
                reader = pypdf.PdfReader(uploaded_file)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        content += text + "\n"
            except ImportError:
                content = "[PDF support requires pypdf. Please run: pip install pypdf]"
        elif "word" in file_type or "docx" in file_type:
            try:
                import docx
                doc = docx.Document(uploaded_file)
                content = "\n".join([para.text for para in doc.paragraphs])
            except ImportError:
                content = "[Word support requires python-docx. Please upload as text file.]"
        else:
            try:
                content = uploaded_file.read().decode("utf-8")
            except Exception:
                content = f"[Cannot read file type: {file_type}]"
    except Exception as e:
        content = f"[Error reading file: {str(e)}]"
    
    return content

async def process_chunk_async(client, chunk, filename, i, total_chunks, openrouter_key):
    """Async worker to process a single chunk with structured JSON outputs"""
    chunk_label = f" (part {i+1}/{total_chunks})" if total_chunks > 1 else ""
    
    prompt = f"""Analyze this document and extract ALL information in multiple layers.
Be EXHAUSTIVE — extract every character, plot point, relationship, rule, and detail. 

DOCUMENT: {filename}{chunk_label}

CONTENT:
{chunk}

Extract in these categories:
LAYER 1 - FACTS: CHARACTER, PLOT, SETTING, THEME, FACT
LAYER 2 - DEEP CONTEXT: CONTEXT, CLARIFICATION, RELATIONSHIP, INSTRUCTION
LAYER 3 - STYLE: PROSE_SAMPLE, DIALOGUE_SAMPLE, VOICE, VOCABULARY

Return ONLY a JSON object containing an array called "memories". Example format:
{{
  "memories": [
    {{"category": "CHARACTER", "content": "John Mercer, mid-30s detective"}},
    {{"category": "CONTEXT", "content": "John's obsession is driven by guilt"}}
  ]
}}"""

    try:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openrouter_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 4096,
                "response_format": {"type": "json_object"}
            },
            timeout=90.0
        )
        
        if response.status_code != 200:
            return [], 0
        
        data = response.json()
        raw = data["choices"][0]["message"]["content"]
        
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        cost = (input_tokens * 2.50 + output_tokens * 15.00) / 1_000_000
        
        parsed = json.loads(raw)
        return parsed.get("memories", []), cost
        
    except Exception:
        return [], 0

def extract_memories_from_file(content, filename, openrouter_key, hf_key):
    """Extract memories concurrently using httpx and asyncio."""
    CHUNK_SIZE = 12000
    CHUNK_OVERLAP = 1500
    MAX_CHUNKS = 5
    
    # Smarter Chunking
    chunks = []
    if len(content) <= CHUNK_SIZE:
        chunks = [content]
    else:
        start = 0
        while start < len(content) and len(chunks) < MAX_CHUNKS:
            end = start + CHUNK_SIZE
            if end < len(content):
                boundary = content.rfind('\n', start, end)
                if boundary == -1 or boundary < end - 1000:
                    boundary = content.rfind('. ', start, end)
                if boundary != -1 and boundary > start + (CHUNK_SIZE // 2):
                    end = boundary + 1
            chunks.append(content[start:end])
            start = end - CHUNK_OVERLAP
            
    async def run_all_chunks():
        async with httpx.AsyncClient() as client:
            tasks = [
                process_chunk_async(client, chunk, filename, i, len(chunks), openrouter_key)
                for i, chunk in enumerate(chunks)
            ]
            return await asyncio.gather(*tasks)

    # Execute async loop from Streamlit
    try:
        loop = asyncio.get_running_loop()
        import nest_asyncio
        nest_asyncio.apply()
        results = loop.run_until_complete(run_all_chunks())
    except RuntimeError:
        results = asyncio.run(run_all_chunks())
    
    all_memories = []
    total_cost = 0
    
    for memories, cost in results:
        if memories:
            all_memories.extend(memories)
        total_cost += cost
    
    seen = set()
    unique_memories = []
    for mem in all_memories:
        key = mem.get("content", "")[:80].lower().strip()
        if key not in seen:
            seen.add(key)
            unique_memories.append(mem)
    
    return unique_memories, total_cost

def store_file_memories(memories, hf_key, session_id=None):
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
    """Extract memories from conversation using GPT-4o with JSON Mode"""
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

Return ONLY a JSON object containing an array called "memories". Example format:
{{
  "memories": [
    {{"category": "CATEGORY", "content": "fact"}}
  ]
}}"""

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openrouter_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 500,
                "response_format": {"type": "json_object"}
            },
            timeout=30
        )
        
        if response.status_code != 200:
            return [], 0
        
        data = response.json()
        raw = data["choices"][0]["message"]["content"]
        
        usage = data.get("usage", {})
        cost = (usage.get("prompt_tokens", 0) * 2.50 + usage.get("completion_tokens", 0) * 15.00) / 1_000_000
        
        parsed = json.loads(raw)
        memories = parsed.get("memories", [])
        return memories, cost
        
    except Exception:
        return [], 0

def store_memories_in_mnemo(memories, hf_key, session_id=None):
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
    try:
        response = requests.get(f"{MNEMO_URL}/stats", headers={"Authorization": f"Bearer {hf_key}"}, timeout=10)
        return response.json() if response.status_code == 200 else {}
    except Exception:
        return {}

def list_mnemo_memories(hf_key, limit=10):
    try:
        response = requests.get(f"{MNEMO_URL}/list", headers={"Authorization": f"Bearer {hf_key}"}, timeout=10)
        if response.status_code == 200:
            return response.json().get("memories", [])[:limit]
        return []
    except Exception:
        return []

def delete_memory(memory_id, hf_key):
    try:
        response = requests.delete(f"{MNEMO_URL}/delete/{memory_id}", headers={"Authorization": f"Bearer {hf_key}"}, timeout=10)
        return response.status_code == 200
    except Exception:
        return False

def clear_all_mnemo_memories(hf_key):
    try:
        response = requests.post(
            f"{MNEMO_URL}/clear",
            headers={"Authorization": f"Bearer {hf_key}", "Content-Type": "application/json"},
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
    st.set_page_config(page_title="4o with Memory", page_icon="🧠", layout="wide")
    st.title("🧠 4o with Memory")
    st.caption("GPT-4o with warm, conversational style and persistent memory")
    
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
        """)
        st.stop()
    
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
    
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = generate_session_id()
    if "custom_titles" not in st.session_state:
        st.session_state.custom_titles = {}
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    with st.sidebar:
        st.header("⚙️ Settings")
        
        with st.expander("🔑 API Keys", expanded=False):
            openrouter_key = st.text_input("OpenRouter API Key", value=DEFAULT_OPENROUTER_KEY, type="password")
            hf_key = st.text_input("HuggingFace Token", value=DEFAULT_HF_KEY, type="password")
        
        openrouter_key = openrouter_key or DEFAULT_OPENROUTER_KEY
        hf_key = hf_key or DEFAULT_HF_KEY
        
        st.divider()
        st.subheader("💬 Chat")
        
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            start_new_chat()
            st.rerun()
        
        st.divider()
        
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
        
        if "session_folders" not in st.session_state:
            st.session_state.session_folders = {"📁 Default": []}
        
        with st.expander("📂 Manage Folders", expanded=False):
            new_folder = st.text_input("New folder name", key="new_folder_input", placeholder="e.g., Story Ideas")
            if st.button("➕ Create Folder", key="create_folder"):
                if new_folder and new_folder.strip():
                    folder_name = f"📁 {new_folder.strip()}"
                    if folder_name not in st.session_state.session_folders:
                        st.session_state.session_folders[folder_name] = []
                        st.success(f"Created {folder_name}")
                        st.rerun()
            
            folders = list(st.session_state.session_folders.keys())
            if len(folders) > 1:
                folder_to_delete = st.selectbox("Delete folder", [""] + [f for f in folders if f != "📁 Default"], key="del_folder")
                if folder_to_delete and st.button("🗑️ Delete Folder", key="delete_folder_btn"):
                    st.session_state.session_folders["📁 Default"].extend(
                        st.session_state.session_folders.get(folder_to_delete, [])
                    )
                    del st.session_state.session_folders[folder_to_delete]
                    st.rerun()
        
        sessions = st.session_state.get("session_history", [])
        
        with st.container():
            if sessions:
                session_to_folder = {}
                for folder, session_ids in st.session_state.session_folders.items():
                    for sid in session_ids:
                        session_to_folder[sid] = folder
                
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
                
                for folder, folder_sessions in folders_with_sessions.items():
                    if folder_sessions:
                        st.caption(folder)
                        for session in folder_sessions[:10]:
                            session_id = session.get("id", "")
                            title = session.get("title", "Untitled")[:30]
                            
                            col1, col2, col3, col4 = st.columns([6, 1, 1, 1])
                            with col1:
                                if st.button(f"💬 {title}", key=f"load_{session_id}", use_container_width=True):
                                    load_session(session_id)
                                    st.rerun()
                            with col2:
                                if st.button("✏️", key=f"rename_{session_id}", help="Rename"):
                                    st.session_state.renaming_session = session_id
                                    st.rerun()
                            with col3:
                                if st.button("📂", key=f"move_{session_id}", help="Move to folder"):
                                    st.session_state.moving_session = session_id
                                    st.rerun()
                            with col4:
                                if st.button("🗑️", key=f"del_{session_id}"):
                                    delete_session(session_id)
                                    st.rerun()
                        st.caption("")
                
                if st.session_state.get("renaming_session"):
                    sid = st.session_state.renaming_session
                    current_title = next((s.get("title", "New Chat") for s in st.session_state.session_history if s.get("id") == sid), "New Chat")
                    with st.container():
                        st.markdown("---")
                        new_name = st.text_input("New name:", value=current_title, key="rename_input")
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("💾 Save", key="save_rename", use_container_width=True):
                                if new_name and new_name.strip():
                                    if "custom_titles" not in st.session_state:
                                        st.session_state.custom_titles = {}
                                    st.session_state.custom_titles[sid] = new_name.strip()
                                    for s in st.session_state.session_history:
                                        if s.get("id") == sid:
                                            s["title"] = new_name.strip()
                                            break
                                    save_current_session()
                                st.session_state.renaming_session = None
                                st.rerun()
                        with col2:
                            if st.button("❌ Cancel", key="cancel_rename", use_container_width=True):
                                st.session_state.renaming_session = None
                                st.rerun()
                
                if st.session_state.get("moving_session"):
                    sid = st.session_state.moving_session
                    with st.container():
                        st.markdown("---")
                        folders = list(st.session_state.session_folders.keys())
                        target_folder = st.selectbox("Move to:", folders, key="move_target")
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("📂 Move", key="confirm_move"):
                                for f in st.session_state.session_folders:
                                    if sid in st.session_state.session_folders[f]:
                                        st.session_state.session_folders[f].remove(sid)
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
        
        with st.expander("⚙️ Settings", expanded=False):
            st.markdown("**📎 Upload File → Memory**")
            st.caption("Upload a file — extracts facts, deep context, style, and saves everything to memory loops in one pass")
            
            uploaded_file = st.file_uploader("Upload file", type=["txt", "md", "csv", "json", "pdf", "docx"], label_visibility="collapsed")
            
            if uploaded_file is not None:
                if st.button("🧠 Extract Deep Context + Memories", use_container_width=True):
                    with st.spinner("Reading file..."):
                        content = extract_text_from_file(uploaded_file)
                    
                    if content and not content.startswith("["):
                        n_chunks = max(1, (len(content) - 1) // 12000 + 1)
                        if n_chunks > 1:
                            st.info(f"📄 {len(content):,} chars → splitting into {min(n_chunks, 5)} chunks for thorough extraction")
                        
                        with st.spinner(f"Extracting deep context & memories ({min(n_chunks, 5)} API call{'s' if n_chunks > 1 else ''})..."):
                            memories, cost = extract_memories_from_file(content, uploaded_file.name, openrouter_key, hf_key)
                        
                        if memories:
                            with st.spinner("Storing to memory loops..."):
                                current_session = st.session_state.get("current_session_id")
                                stored = store_file_memories(memories, hf_key, session_id=current_session)
                            
                            facts = [m for m in memories if m.get("category") in ("CHARACTER", "PLOT", "SETTING", "THEME", "FACT")]
                            context = [m for m in memories if m.get("category") in ("CONTEXT", "CLARIFICATION", "RELATIONSHIP", "INSTRUCTION")]
                            style = [m for m in memories if m.get("category") in ("PROSE_SAMPLE", "DIALOGUE_SAMPLE", "VOICE", "VOCABULARY")]
                            
                            st.success(f"✅ Stored {stored} memories")
                            st.caption(f"📊 {len(facts)} facts · {len(context)} deep context · {len(style)} style | Cost: ${cost:.4f}")
                            
                            with st.expander("View extracted memories", expanded=False):
                                for mem in memories:
                                    cat = mem.get("category", "FACT")
                                    txt = mem.get("content", "")[:150]
                                    st.caption(f"**[{cat}]** {txt}")
                            
                            if "loop_manager" in st.session_state:
                                st.session_state.loop_manager.load_from_mnemo()
                        else:
                            st.warning("No memories extracted. Try a different file.")
                    else:
                        st.error("Could not read file content")
        
        st.divider()
        st.subheader("🧠 Memory Settings")
        
        cross_session_enabled = st.toggle("Cross-Session Memory", value=True, help="Remember across chat sessions")
        auto_extract = st.toggle("Auto-Extract Memories", value=True, help="Automatically extract facts from conversations")
        st.session_state.auto_extract = auto_extract
        
        use_loops = st.toggle("🔄 Metadata Loops (Save 80% tokens)", value=True, help="Use token-efficient context injection")
        st.session_state.use_loops = use_loops
        
        if "loop_manager" in st.session_state and use_loops:
            loop_stats = st.session_state.loop_manager.get_stats()
            st.caption(f"📊 {loop_stats['total_items']} memories | {loop_stats['total_metadata_tokens']} tokens")
        
        st.divider()
        st.subheader("📝 Add Memory")
        
        with st.expander("Add manually", expanded=False):
            memory_category = st.selectbox("Category", ["CHARACTER", "PLOT", "SETTING", "THEME", "STYLE", "FACT"])
            memory_content = st.text_area("Content", placeholder="e.g., Detective Mercer has a fear of water since childhood", height=80)
            if st.button("💾 Save", use_container_width=True):
                if memory_content.strip():
                    current_session = st.session_state.get("current_session_id")
                    stored = store_memories_in_mnemo([{"category": memory_category, "content": memory_content}], hf_key, session_id=current_session)
                    if stored:
                        st.success(f"✅ Saved [{memory_category}]")
                        if "loop_manager" in st.session_state:
                            st.session_state.loop_manager.add_to_loop(memory_content, memory_category.lower())
                    else:
                        st.error("Failed to save")
        
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
            
            memories = list_mnemo_memories(hf_key, limit=15)
            
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
            
            all_memories = list_mnemo_memories(hf_key, limit=500)
            
            if search_query:
                all_memories = [m for m in all_memories if search_query.lower() in m.get("content", "").lower()]
            if category_filter != "All":
                all_memories = [m for m in all_memories if f"[{category_filter}]" in m.get("content", "")]
            
            st.caption(f"Showing {len(all_memories)} memories")
            
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
            
            for mem in all_memories:
                content = mem.get("content", "")
                mem_id = mem.get("id", "")
                
                category = "OTHER"
                if content.startswith("["):
                    category = content.split("]")[0][1:]
                
                col1, col2, col3 = st.columns([1, 8, 1])
                with col1:
                    st.caption(f"[{category}]")
                with col2:
                    st.text(content[len(f"[{category}]"):].strip()[:200])
                with col3:
                    if st.button("🗑️", key=f"del_full_{mem_id}"):
                        if delete_memory(mem_id, hf_key):
                            st.rerun()
        
        st.divider()
        
        with st.expander("🏥 Memory Health", expanded=False):
            if "context_engine" in st.session_state and "loop_manager" in st.session_state:
                health = st.session_state.context_engine.degradation.get_health_report(st.session_state.loop_manager)
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
                        result = st.session_state.context_engine.degradation.apply_decay(st.session_state.loop_manager)
                        if result.get("skipped"):
                            st.info(result.get("reason", "Skipped"))
                        else:
                            st.success(f"Decayed {result.get('decayed_items', 0)} items")
                with col2:
                    if st.button("🧹 Prune Stale", use_container_width=True):
                        result = st.session_state.context_engine.degradation.prune_memories(st.session_state.loop_manager, dry_run=False)
                        st.success(f"Pruned {result.get('pruned', 0)} items")
                        st.rerun()
                
                st.markdown("---")
                st.markdown("**🧠 Memory Consolidation**")
                st.caption("Analyze memories & generate deep context")
                
                last_consol = st.session_state.get("last_consolidation")
                if last_consol:
                    st.caption(f"Last run: {last_consol[:16]}")
                
                if st.button("🧠 Consolidate Now", use_container_width=True):
                    with st.spinner("Analyzing memories... (this may take 30-60 seconds)"):
                        result = st.session_state.context_engine.consolidate_memories(openrouter_key)
                        if result.get("error"):
                            st.error(f"Error: {result['error']}")
                        else:
                            st.session_state.last_consolidation = result["timestamp"]
                            st.success(f"✅ Created {result['created']} new context entries!")
                            st.caption(f"Analyzed: {result['memories_analyzed']} memories | Cost: ${result['cost']:.4f}")
                            if result.get("new_entries"):
                                with st.expander("New entries created"):
                                    for entry in result["new_entries"]:
                                        st.caption(f"[{entry['category']}] {entry['content']}")
                            st.session_state.loop_manager.load_from_mnemo(use_smart_extraction=False)
        
        st.divider()
        st.subheader("💰 Costs")
        st.caption(f"Messages: {st.session_state.get('message_count', 0)}")
        st.caption(f"Total: ${st.session_state.get('total_cost', 0):.4f}")
    
    # Initialize Core Engines
    if "memory_manager" not in st.session_state:
        st.session_state.memory_manager = MnemoMemoryManager(
            openrouter_key=openrouter_key,
            hf_key=hf_key,
            user_id="default_user",
            cross_session_enabled=cross_session_enabled
        )
    if "loop_manager" not in st.session_state:
        st.session_state.loop_manager = LoopManager(openrouter_key=openrouter_key, hf_key=hf_key, mnemo_url=MNEMO_URL)
        st.session_state.loop_manager.load_from_mnemo(use_smart_extraction=False)
    if "smart_memory" not in st.session_state:
        st.session_state.smart_memory = SmartMemory()
    if "context_engine" not in st.session_state:
        st.session_state.context_engine = ContextEngine(hf_key=hf_key, openrouter_key=openrouter_key, mnemo_url=MNEMO_URL)
    if "context_manager" not in st.session_state:
        st.session_state.context_manager = ContextWindowManager(loop_manager=st.session_state.loop_manager)
    else:
        st.session_state.context_manager.set_loop_manager(st.session_state.loop_manager)
    
    msg_count = len(st.session_state.get("messages", []))
    if msg_count > 0 and msg_count % 50 == 0:
        if "last_maintenance" not in st.session_state or st.session_state.last_maintenance != msg_count:
            st.session_state.context_engine.maintenance(st.session_state.loop_manager, apply_decay=True, prune=False)
            st.session_state.last_maintenance = msg_count
    
    st.session_state.memory_manager.toggle_cross_session(cross_session_enabled)
    
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
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
                    if st.button("📋", key=f"copy_{idx}", help="Copy response"):
                        st.session_state.show_copy_modal = idx
                        st.session_state.copy_content = message["content"]
    
    if st.session_state.get("show_copy_modal") is not None:
        with st.container():
            st.markdown("---")
            st.markdown("**📋 Copy this text:**")
            st.code(st.session_state.copy_content, language=None)
            if st.button("✅ Done", key="close_copy"):
                st.session_state.show_copy_modal = None
                st.session_state.copy_content = None
                st.rerun()
    
    if prompt := st.chat_input("What's on your mind?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                conversation_length = len(st.session_state.messages)
                needs_memory, memory_reason = st.session_state.smart_memory.should_use_memory(prompt, conversation_length)
                
                past_conversation_context = ""
                sessions_found = 0
                current_session_id = st.session_state.get("current_session_id", "")
                skip_loops_for_this_query = False
                
                if needs_memory and st.session_state.get("use_loops", True):
                    try:
                        storage = get_persistent_storage()
                        if storage:
                            prompt_lower = prompt.lower()
                            asking_about_past = any(phrase in prompt_lower for phrase in [
                                "last chat", "previous chat", "last conversation", 
                                "previous conversation", "earlier chat", "before this",
                                "what did we talk", "what were we", "did we discuss",
                                "remember when", "last time", "our last", "previous session",
                                "talked about", "chatting about", "we discussed", "we were talking"
                            ])
                            
                            if asking_about_past:
                                skip_loops_for_this_query = True
                                session_results = storage.search_sessions(prompt, current_session_id=current_session_id, limit=2)
                                if session_results:
                                    sessions_found = len(session_results)
                                    past_conversation_context = "\n\n[PREVIOUS CHAT SESSIONS - The user is asking about past conversations. Use ONLY this information to answer. Do NOT use other memories:]\n" + "\n---\n".join(session_results)
                                else:
                                    recent = storage.get_previous_sessions_content(current_session_id=current_session_id, limit=2)
                                    if recent:
                                        sessions_found = len(recent)
                                        summaries = [f"Session '{s['title']}':\n{s['summary']}" for s in recent]
                                        past_conversation_context = "\n\n[RECENT CHAT SESSIONS - The user is asking about past conversations. Use ONLY this information to answer:]\n" + "\n---\n".join(summaries)
                            else:
                                past_convos = storage.search_conversations(prompt, limit=3)
                                if past_convos:
                                    past_conversation_context = "\n\n[PAST CONVERSATIONS]\n" + "\n".join(f"• {conv[:200]}" for conv in past_convos)
                                
                                if "context_engine" in st.session_state:
                                    enriched, enrich_meta = st.session_state.context_engine.build_rich_context(prompt, st.session_state.loop_manager)
                                    if enriched:
                                        past_conversation_context += f"\n\n[DEEP CONTEXT]\n{enriched}"
                    except Exception:
                        pass
                
                full_system_prompt = SYSTEM_PROMPT + past_conversation_context
                
                messages, context_stats = st.session_state.context_manager.build_optimized_context(
                    system_prompt=full_system_prompt,
                    query=prompt,
                    conversation_history=st.session_state.messages,
                    max_messages=MAX_CONVERSATION_MESSAGES,
                    use_loops=(needs_memory and st.session_state.get("use_loops", True) and not skip_loops_for_this_query)
                )
                
                context_meta = {
                    "cross_session_memories_used": context_stats.get("memory_items_full", 0) + context_stats.get("memory_items_meta", 0),
                    "context_tokens": context_stats.get("total_tokens", 0),
                    "mode": "smart" if needs_memory else "skip",
                    "memory_reason": memory_reason,
                    "sessions_found": sessions_found
                }
                
                response, input_tokens, output_tokens, error = call_openrouter(messages, openrouter_key)
                
                if error:
                    st.error(error)
                else:
                    cost_tracker = CostTracker()
                    msg_cost = cost_tracker.add_usage(input_tokens, output_tokens)
                    
                    st.session_state.memory_manager.process_turn(prompt, response)
                    
                    if not skip_loops_for_this_query:
                        try:
                            storage = get_persistent_storage()
                            storage.save_conversation_turn(prompt, response, st.session_state.get("current_session_id"))
                        except Exception:
                            pass
                    
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
                                st.session_state.loop_manager.add_to_loop(content=mem.get("content", ""), category=mem.get("category", "general").lower())
                    
                    st.markdown(response)
                    
                    meta_parts = []
                    if context_meta.get("sessions_found", 0) > 0:
                        meta_parts.append(f"📜 {context_meta['sessions_found']} past chats")
                    if context_meta.get("cross_session_memories_used", 0) > 0:
                        meta_parts.append(f"📚 {context_meta['cross_session_memories_used']} memories")
                    if context_meta.get("mode") == "smart":
                        meta_parts.append(f"🔄 {context_meta.get('context_tokens', 0)} tokens")
                    elif context_meta.get("mode") == "skip":
                        meta_parts.append("⚡ fast")
                    if extracted > 0:
                        meta_parts.append(f"🧠 {extracted} extracted")
                    meta_parts.append(f"💰 ${msg_cost:.4f}")
                    st.caption(" | ".join(meta_parts))
                    
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
                    
                    save_current_session()
    
    st.divider()
    st.caption("🧠 4o with Memory | GPT-4o + Mnemo v4 + Metadata Loops")

if __name__ == "__main__":
    main()

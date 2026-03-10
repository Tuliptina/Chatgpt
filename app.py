"""
4o with Memory v6.9.1 — Dual-Processor Creative Writing Engine

v6.9.1 changes:
- NEW: Auto-scaling extraction via calculate_extraction_plan() — chunk count and
  max_tokens are now decided dynamically by file size instead of hardcoded.
  Formula: ~12 memories/1K words × 100 tokens/memory JSON + overhead → auto max_tokens.
  Small files (<2K words): 1 call @ 4096. Medium (<6.5K): 1 call @ 8192.
  Large (<10K): 1 call @ 10-12K. Very large: splits into 2-5 parallel chunks.
- FIX: Removed response_format from MEMORY_PARAMS — K2 on OpenRouter doesn't support it,
  causing silent extraction failures ("No memories extracted").
- FIX: Added extract_json_from_response() — robust JSON extraction that handles markdown
  fences, text preamble, and brace-depth matching.
- FIX: Replaced silent except-swallowing in process_chunk_async and extract_memories_with_gpt
  with actual error logging so failures are visible in Streamlit Cloud logs.
- FIX: Removed repetition_penalty from MEMORY_PARAMS — non-standard param that some
  OpenRouter providers reject with 400 errors.

v6.9 changes:
- FIX: Increased file extraction chunk size to 40,000 characters to prevent API queuing.
- FIX: Rewrote K2 prompts for "Exhaustive Atomic Extraction" to stop LLM laziness and force high-volume extractions.
- Prior fixes: Slow-burn mandate, Context-aware continuations, Brute-force signal cleaning, Relational JSON.
"""

import streamlit as st
import requests
import httpx
import asyncio
import json
import time
import os
import uuid
import re
import traceback
import concurrent.futures
from datetime import datetime

from mnemo_client import MnemoClient
from metadata_loops import LoopManager, LoopConfig
from smart_memory import SmartMemory, ContextWindowManager
from session_store import SessionStore
from context_engine import ContextEngine

from signal_protocol import SignalProcessor
from prefetch_engine import PrefetchEngine
from memory_schema import parse_signals_from_response, get_signal_instruction


# ============================================================================
# CONFIGURATION
# ============================================================================

def get_secret(key, default=""):
    try:
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)

DEFAULT_OPENROUTER_KEY = get_secret("OPENROUTER_KEY", "")
DEFAULT_HF_KEY = get_secret("HF_KEY", "")
MNEMO_URL = "https://athelaperk-mnemo.hf.space"

WRITING_MODEL_ID = "openai/gpt-4o-2024-11-20"
MEMORY_MODEL_ID = "moonshotai/kimi-k2"

WRITING_PARAMS = {
    "temperature": 0.85,
    "frequency_penalty": 0.2,
    "presence_penalty": 0.3,
}

# v6.9.1 FIX: Removed response_format and repetition_penalty
# K2 on OpenRouter doesn't support response_format: json_object — it either
# returns a 400 error or ignores it, causing extraction to silently fail.
# repetition_penalty is non-standard and some providers reject it.
# max_tokens=4096 is the DEFAULT for conversation extraction (short output).
# For file extraction, calculate_extraction_plan() overrides this dynamically.
MEMORY_PARAMS = {
    "temperature": 0.2,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "max_tokens": 4096,  # Default — overridden by calculate_extraction_plan() for files
}

MAX_CONVERSATION_MESSAGES = 8
MAX_SESSIONS_STORED = 20

CATEGORY_TO_CP = {
    "CHARACTER": ("character", "character_profile"),
    "PLOT": ("plot", "plot_event"),
    "SETTING": ("setting", "setting_detail"),
    "THEME": ("fact", "theme"),
    "STYLE": ("style", "voice_note"),
    "TONE": ("tone", "tone_directive"),
    "FACT": ("fact", "general"),
    "CONTEXT": ("fact", "context_note"),
    "CLARIFICATION": ("fact", "clarification"),
    "RELATIONSHIP": ("relationship", "relationship"),
    "INSTRUCTION": ("fact", "instruction"),
    "PROSE_SAMPLE": ("style", "prose_sample"),
    "DIALOGUE_SAMPLE": ("style", "dialogue_sample"),
    "VOICE": ("style", "voice_note"),
    "VOCABULARY": ("style", "vocabulary"),
    "TIMELINE": ("plot", "timeline"),
}

# ==========================================================================
# ENTITY NORMALIZATION
# ==========================================================================

NON_ENTITY_WORDS = {
    "father", "mother", "brother", "sister", "child", "son", "daughter",
    "husband", "wife", "uncle", "aunt", "cousin", "antagonist", "protagonist",
    "narrator", "victim", "captor", "mentor", "rival", "doctor", "professor",
    "reverend", "lord", "lady", "story", "scene", "chapter", "book",
    "theme", "tone", "style", "context", "clarification", "timeline",
    "profile", "summary", "overview", "description", "detail",
}

def normalize_entity(raw_entity: str) -> str:
    entity = raw_entity.strip()
    if not entity:
        return "Story"
    
    category_prefixes = [
        "Clarification:", "Tone:", "Context:", "Relationship:",
        "Timeline:", "Setting:", "Theme:", "Instruction:",
        "Style:", "Voice:", "Fact:",
    ]
    for prefix in category_prefixes:
        if entity.startswith(prefix):
            entity = entity[len(prefix):].strip()
    
    title_prefixes = [
        "Dr. ", "Dr ", "Reverend ", "Rev. ", "Rev ",
        "Professor ", "Prof. ", "Prof ", "Lord ", "Lady ",
        "Sir ", "Dame ", "Mr. ", "Mrs. ", "Ms. ", "Miss ",
    ]
    for prefix in title_prefixes:
        if entity.startswith(prefix):
            entity = entity[len(prefix):].strip()
            break
    
    for article in ["The ", "the ", "A ", "a ", "An ", "an "]:
        if entity.startswith(article) and len(entity) > len(article) + 3:
            entity = entity[len(article):].strip()
            break
    
    poss_match = re.match(r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'s\s+\w+", entity)
    if poss_match:
        entity = poss_match.group(1)
    
    ew = [w.lower().rstrip("'s") for w in entity.split()]
    if all(w in NON_ENTITY_WORDS for w in ew):
        return "Story"
    if len(ew) == 1 and ew[0] in NON_ENTITY_WORDS:
        return "Story"
    
    words = entity.split()
    if not any(w[0].isupper() for w in words if len(w) > 0):
        entity = entity.title()
    
    return entity if entity else "Story"

def store_as_cp(mnemo_client, entity, category, content, session_id="",
                source="auto_extract", connects_to="", weight=0.5):
    cat_key = category.upper()
    cp_category, point_type = CATEGORY_TO_CP.get(cat_key, ("fact", "general"))
    normalized_entity = normalize_entity(entity)
    normalized_connects = normalize_entity(connects_to) if connects_to else ""
    return mnemo_client.add_point(
        entity=normalized_entity, point_type=point_type, value=content,
        connects_to=normalized_connects, reason="", weight=float(weight),
        category=cp_category, session_id=session_id, source=source,
    )

def auto_convert_blobs(mnemo_client):
    import re
    try:
        blobs = mnemo_client.list_memories()
        if not blobs:
            return 0
        stats = mnemo_client.get_stats()
        if stats.get("total_connection_points", 0) > 0:
            return 0 
        
        converted = 0
        for mem in blobs:
            content = mem.get("content", "")
            meta = mem.get("metadata", {})
            if not content:
                continue
            tag_match = re.match(r'\[([A-Z_]+)\]\s*(.*)', content, re.DOTALL)
            if tag_match:
                category = tag_match.group(1)
                body = tag_match.group(2).strip()
            else:
                category = "FACT"
                body = content
            entity_match = re.match(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', body)
            entity = entity_match.group(1) if entity_match else "Story"
            connects_to = ""
            if "→" in body or "->" in body:
                arrow_match = re.match(r'(\w+)\s*(?:→|->)\s*(\w+)', body)
                if arrow_match:
                    entity = arrow_match.group(1)
                    connects_to = arrow_match.group(2)
            source = meta.get("source", "migrated")
            session_id = meta.get("session_id", "")
            cp_id = store_as_cp(
                mnemo_client, entity=entity, category=category,
                content=body, session_id=session_id,
                source=source, connects_to=connects_to,
                weight=0.7 if category in ("CLARIFICATION", "CONTEXT", "RELATIONSHIP") else 0.5,
            )
            if cp_id:
                converted += 1
        return converted
    except Exception:
        return 0

SYSTEM_PROMPT = """You are a sharp, well-read creative collaborator with genuine enthusiasm for storytelling craft. You have three modes — detect which one the user needs and switch seamlessly.

MODE 1 — CONVERSATION (chatting about the project, brainstorming, planning):
- You're the kind of collaborator writers actually enjoy working with — curious, opinionated about craft, and genuinely invested in their story.
- Have real opinions. If a plot idea is predictable, say so with charm, then pitch something better. If a character detail is brilliant, get excited about it.
- Be witty when the moment allows, but never at the expense of the work.
- Ask questions that show you're thinking ahead.
- Use casual, intelligent language. You're a peer, not a tutor. Skip the preambles.
- Reference their stored characters and plot points naturally.
- CRITICAL — know the difference between RECALLING and CREATING.

MODE 2 — RECALL & RETRIEVAL (user asks to recall, list, summarize):
- Give clean, factual answers drawn ONLY from your memory context.
- Use EXACT details from memory — do NOT embellish, infer, or fill gaps with imagination.

MODE 3 — CREATIVE WRITING (scenes, chapters, dialogue, prose):
- Your personality DISAPPEARS. You are not "witty" — you are the story.
- Match the tonal register from the [TONE DIRECTIVE] in your context. Dark stays dark. Humor cuts. Tension holds.
- Deep psychological complexity in characters.
- Setting-accurate language for the project's period/genre.
- Show don't tell. Sensory detail. Subtext in dialogue.
- Default to third person past tense unless user or memory specifies otherwise.
- Match the user's writing voice if VOICE/PROSE_SAMPLE memories exist.
- NEVER soften emotional edges.
- DO NOT resolve tension prematurely. DO NOT add reassuring internal monologue unless the brief says to.

PROSE RHYTHM — this is critical:
- Vary sentence length deliberately. Follow a long, winding sentence with a short one. Then a fragment. Then build again.
- Do NOT write in choppy, staccato bursts — that reads like a screenplay, not a novel.
- Do NOT write in dense, breathless paragraphs with no white space — that exhausts the reader.
- Paragraphs should BREATHE. Mix action beats, sensory detail, interiority, and dialogue across paragraphs.
- Let scenes have rhythm: tension builds in longer sentences, snaps in short ones. Quiet moments drift. Shock is blunt."""

# ============================================================================
# INITIALIZATION & SESSION MANAGEMENT
# ============================================================================

def init_client(hf_key):
    if "mnemo_client" not in st.session_state:
        st.session_state.mnemo_client = MnemoClient(base_url=MNEMO_URL, token=hf_key)
    return st.session_state.mnemo_client

def get_persistent_storage(hf_key=None, client=None):
    key = hf_key or DEFAULT_HF_KEY
    if "persistent_storage" not in st.session_state or st.session_state.get("_ps_key") != key:
        if key and client:
            st.session_state.persistent_storage = SessionStore(hf_key=key, mnemo_client=client)
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
    title = custom_titles.get(session_id, get_session_title(st.session_state.messages))
    messages_copy = [msg.copy() for msg in st.session_state.messages]
    storage = get_persistent_storage()
    if storage:
        try:
            storage.save_session(session_id=session_id, title=title,
                                 messages=messages_copy, timestamp=datetime.now().isoformat())
            msg_count = len([m for m in messages_copy if m.get("role") == "user"])
            if msg_count > 0 and msg_count % 25 == 0:
                storage.cleanup_stale_sessions()
        except Exception:
            pass
    if "session_history" not in st.session_state:
        st.session_state.session_history = []
    current_session = {
        "id": session_id, "title": title, "timestamp": datetime.now().isoformat(),
        "message_count": len([m for m in messages_copy if m["role"] == "user"]),
        "preview": messages_copy[0]["content"][:100] if messages_copy else "",
        "messages": messages_copy
    }
    existing_idx = next((i for i, s in enumerate(st.session_state.session_history)
                         if s["id"] == current_session["id"]), None)
    if existing_idx is not None:
        st.session_state.session_history[existing_idx] = current_session
    else:
        st.session_state.session_history.insert(0, current_session)
    st.session_state.session_history = st.session_state.session_history[:MAX_SESSIONS_STORED]

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
        s for s in st.session_state.get("session_history", []) if s["id"] != session_id
    ]
    try:
        storage = get_persistent_storage()
        if storage:
            storage.delete_session(session_id)
    except Exception:
        pass
    if "loop_manager" in st.session_state:
        lm = st.session_state.loop_manager
        if hasattr(lm, 'remove_session_tokens'):
            lm.remove_session_tokens(session_id)

@st.dialog("Rename Session")
def rename_session_dialog(session_id, current_title):
    new_name = st.text_input("New name:", value=current_title, key=f"rename_input_{session_id}")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save", use_container_width=True):
            if new_name and new_name.strip():
                new_title = new_name.strip()
                if "custom_titles" not in st.session_state:
                    st.session_state.custom_titles = {}
                st.session_state.custom_titles[session_id] = new_title
                for s in st.session_state.session_history:
                    if s.get("id") == session_id:
                        s["title"] = new_title
                        break
                storage = get_persistent_storage()
                if storage:
                    messages = storage.load_session_messages(session_id)
                    storage.save_session(session_id, new_title, messages)
                if session_id == st.session_state.get("current_session_id"):
                    save_current_session()
            st.rerun()
    with col2:
        if st.button("❌ Cancel", use_container_width=True):
            st.rerun()

@st.dialog("Move Session")
def move_session_dialog(session_id):
    folders = list(st.session_state.session_folders.keys())
    target_folder = st.selectbox("Move to:", folders, key=f"move_target_{session_id}")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📁 Move", use_container_width=True):
            for f in st.session_state.session_folders:
                if session_id in st.session_state.session_folders[f]:
                    st.session_state.session_folders[f].remove(session_id)
            st.session_state.session_folders[target_folder].append(session_id)
            storage = get_persistent_storage()
            if storage and hasattr(storage, 'save_folder_state'):
                storage.save_folder_state(st.session_state.session_folders)
            st.rerun()
    with col2:
        if st.button("❌ Cancel", use_container_width=True):
            st.rerun()

@st.dialog("Copy Response")
def copy_response_dialog(content):
    st.markdown("**📋 Copy this text:**")
    st.code(content, language=None)
    if st.button("✅ Done", use_container_width=True):
        st.rerun()

# ============================================================================
# FILE PROCESSING & K2 AUTO-EXTRACTION (v6.9.1 — Fixed JSON extraction)
# ============================================================================

def extract_text_from_file(uploaded_file):
    file_type = uploaded_file.type
    content = ""
    try:
        if file_type in ("text/plain", "text/csv", "text/markdown"):
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
                content = "[PDF support requires pypdf]"
        elif "word" in file_type or "docx" in file_type:
            try:
                import docx
                doc = docx.Document(uploaded_file)
                content = "\n".join([para.text for para in doc.paragraphs])
            except ImportError:
                content = "[Word support requires python-docx]"
        else:
            try:
                content = uploaded_file.read().decode("utf-8")
            except Exception:
                content = f"[Cannot read file type: {file_type}]"
    except Exception as e:
        content = f"[Error reading file: {str(e)}]"
    return content


# v6.9.1 FIX: Robust JSON extraction from K2 responses
def extract_json_from_response(raw: str) -> dict:
    """Robustly extract JSON from K2 response text.
    
    K2 on OpenRouter doesn't support response_format, so it may return JSON:
    - Directly as a JSON object
    - Wrapped in ```json ... ``` markdown fences  
    - With preamble text before the JSON
    - With trailing text after the JSON
    """
    clean = raw.strip()
    
    # Strategy 1: Direct JSON parse (best case)
    try:
        parsed = json.loads(clean)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Strip markdown fences
    if "```" in clean:
        fence_patterns = [
            r'```json\s*\n?(.*?)\n?```',
            r'```\s*\n?(.*?)\n?```',
        ]
        for pattern in fence_patterns:
            match = re.search(pattern, clean, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(1).strip())
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    continue
    
    # Strategy 3: Brace-depth matching — find outermost { ... }
    first_brace = clean.find('{')
    if first_brace >= 0:
        depth = 0
        for i in range(first_brace, len(clean)):
            if clean[i] == '{':
                depth += 1
            elif clean[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(clean[first_brace:i+1])
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        pass
                    break
    
    return {}


# v6.9.1: Auto-scaling extraction — calculates optimal chunks & max_tokens per file
def calculate_extraction_plan(content: str) -> dict:
    """Auto-scale extraction API calls based on file size.
    
    Returns dict with: n_chunks, chunk_size, chunk_overlap, max_tokens_per_call,
    estimated_memories, estimated_output_tokens, words, chars
    """
    words = len(content.split())
    chars = len(content)
    
    MEMORIES_PER_1K_WORDS = 12
    TOKENS_PER_MEMORY_JSON = 100
    THREAD_KNOT_OVERHEAD = 500
    SAFETY_MARGIN = 1.25
    MAX_RELIABLE_OUTPUT = 12000  # K2's reliable output ceiling per call
    TARGET_OUTPUT_PER_CHUNK = 10000
    
    expected_memories = max(5, int(words / 1000 * MEMORIES_PER_1K_WORDS))
    raw_output = expected_memories * TOKENS_PER_MEMORY_JSON + THREAD_KNOT_OVERHEAD
    estimated_output = int(raw_output * SAFETY_MARGIN)
    
    if estimated_output <= 4096:
        n_chunks, max_tokens = 1, 4096
    elif estimated_output <= 8192:
        n_chunks, max_tokens = 1, 8192
    elif estimated_output <= MAX_RELIABLE_OUTPUT:
        n_chunks, max_tokens = 1, estimated_output
    else:
        memories_per_chunk = (TARGET_OUTPUT_PER_CHUNK - THREAD_KNOT_OVERHEAD) // TOKENS_PER_MEMORY_JSON
        words_per_chunk = int(memories_per_chunk / MEMORIES_PER_1K_WORDS * 1000)
        chars_per_chunk = words_per_chunk * 7
        n_chunks = min(max(1, -(-chars // chars_per_chunk)), 5)
        max_tokens = TARGET_OUTPUT_PER_CHUNK
    
    if n_chunks == 1:
        chunk_size = chars + 1
        chunk_overlap = 0
    else:
        chunk_size = chars // n_chunks + 2000
        chunk_overlap = 2000
    
    plan = {
        "n_chunks": n_chunks, "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap, "max_tokens_per_call": max_tokens,
        "estimated_memories": expected_memories,
        "estimated_output_tokens": estimated_output,
        "words": words, "chars": chars,
    }
    print(f"[EXTRACT PLAN] {words:,} words → ~{expected_memories} memories → "
          f"~{estimated_output:,} output tokens → {n_chunks} call(s) @ max_tokens={max_tokens}")
    return plan


async def process_chunk_async(http_client, chunk, filename, i, total_chunks, openrouter_key, params_override=None):
    chunk_label = f" (part {i+1}/{total_chunks})" if total_chunks > 1 else ""
    # v6.9 FIX: Explicitly demand high-volume exhaustive extraction
    prompt = f"""Extract structured memories from this document. Your PRIMARY goal is EXHAUSTIVE ATOMIC EXTRACTION. Do not summarize — break everything down into granular pieces. 
Because this is a large text chunk, I expect a HIGH VOLUME of extracted memories (dozens of points). Read carefully to the very end of the document.

DOCUMENT: {filename}{chunk_label}
CONTENT:
{chunk}

METHOD:
1. EXHAUSTIVE EXTRACTION (PRIORITY 1): Extract ALL individual atomic facts as "memories". Split compound sentences into separate points. Classify the "category" strictly as: CHARACTER, PLOT, RELATIONSHIP, SETTING, TONE, CLARIFICATION, or FACT. For RELATIONSHIP, you MUST include a "connects_to" target.
2. THREADING (PRIORITY 2): Only after extracting all atomic memories, if some form a narrative sequence, group their local_ids into a "thread". 
3. KNOTTING: If storylines collide, create a "knot".

CRITICAL ENTITY NAMING RULES:
  - Use plain names: "Sebastian Carlisle", NOT "Dr. Sebastian Carlisle"
  - Entity is a LOOKUP KEY for indexing. Keep it short and consistent.

You MUST respond with ONLY a valid JSON object. No preamble, no explanation — pure JSON:
{{
  "memories": [
    {{"local_id": "m1", "entity": "Alistair", "category": "CHARACTER", "content": "Alistair is terrified of developing dementia."}},
    {{"local_id": "m2", "entity": "Alistair", "category": "RELATIONSHIP", "connects_to": "Sebastian", "content": "Alistair sees Sebastian's captivity as a medical necessity."}},
    {{"local_id": "m3", "entity": "Sebastian", "category": "TONE", "content": "Sebastian's dialogue should be sparse and dissociated."}}
  ],
  "threads": [
    {{"name": "Captivity Arc", "entity": "Sebastian", "type": "plot_line", "memory_local_ids": ["m2", "m3"]}}
  ],
  "knots": []
}}"""

    # v6.9.1 FIX: Proper error handling instead of silent swallowing
    call_params = params_override or MEMORY_PARAMS
    try:
        response = await http_client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {openrouter_key}", "Content-Type": "application/json"},
            json={"model": MEMORY_MODEL_ID, "messages": [{"role": "user", "content": prompt}], **call_params},
            timeout=120.0
        )
        if response.status_code != 200:
            print(f"[K2 EXTRACT] API error {response.status_code} for chunk {i+1}: {response.text[:300]}")
            return {"memories": [], "threads": [], "knots": []}, 0
            
        data = response.json()
        raw = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        cost = (usage.get("prompt_tokens", 0) * 0.55 + usage.get("completion_tokens", 0) * 2.20) / 1_000_000
        
        # v6.9.1: Use robust JSON extraction
        parsed = extract_json_from_response(raw)
        
        if not parsed:
            print(f"[K2 EXTRACT] JSON parse failed for chunk {i+1}. Raw preview: {raw[:500]}")
            return {"memories": [], "threads": [], "knots": []}, cost
        
        if "memories" in parsed:
            return parsed, cost
        
        # K2 sometimes uses different key names
        for alt_key in ("results", "items", "data", "points", "extractions"):
            if alt_key in parsed:
                print(f"[K2 EXTRACT] Found memories under alternate key '{alt_key}'")
                return {"memories": parsed[alt_key], "threads": parsed.get("threads", []), "knots": parsed.get("knots", [])}, cost
        
        print(f"[K2 EXTRACT] Parsed JSON but no 'memories' key. Keys: {list(parsed.keys())}. Preview: {raw[:300]}")
        return {"memories": [], "threads": [], "knots": []}, cost
        
    except Exception as e:
        print(f"[K2 EXTRACT] Exception in chunk {i+1}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return {"memories": [], "threads": [], "knots": []}, 0

def extract_memories_from_file(content, filename, openrouter_key):
    """v6.9.1: Auto-scaling extraction — chunk count and max_tokens decided by file size."""
    plan = calculate_extraction_plan(content)
    
    CHUNK_SIZE = plan["chunk_size"]
    CHUNK_OVERLAP = plan["chunk_overlap"]
    MAX_CHUNKS = plan["n_chunks"]
    max_tokens_override = plan["max_tokens_per_call"]
    
    # Build extraction params with auto-scaled max_tokens
    extract_params = {**MEMORY_PARAMS, "max_tokens": max_tokens_override}
    
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
        async with httpx.AsyncClient() as http_client:
            tasks = [process_chunk_async(http_client, chunk, filename, i, len(chunks),
                                         openrouter_key, extract_params)
                     for i, chunk in enumerate(chunks)]
            return await asyncio.gather(*tasks)

    try:
        loop = asyncio.get_running_loop()
        import nest_asyncio
        nest_asyncio.apply()
        results = loop.run_until_complete(run_all_chunks())
    except RuntimeError:
        results = asyncio.run(run_all_chunks())

    all_memories, all_threads, all_knots = [], [], []
    total_cost = 0
    for data, cost in results:
        if isinstance(data, dict):
            all_memories.extend(data.get("memories", []))
            all_threads.extend(data.get("threads", []))
            all_knots.extend(data.get("knots", []))
        total_cost += cost
        
    seen = set()
    unique_memories = []
    for mem in all_memories:
        key = mem.get("content", "")[:80].lower().strip()
        if key not in seen:
            seen.add(key)
            unique_memories.append(mem)
            
    return {"memories": unique_memories, "threads": all_threads, "knots": all_knots}, total_cost, plan

def extract_memories_with_gpt(conversation, openrouter_key):
    prompt = """Extract structured memories from this conversation. Your PRIMARY goal is EXHAUSTIVE ATOMIC EXTRACTION. Do not summarize — break everything down into granular pieces.

CONVERSATION:
{conversation}

METHOD:
1. EXHAUSTIVE EXTRACTION (PRIORITY 1): Extract ALL individual atomic facts as "memories" based on what was discussed. Classify the "category" strictly as: CHARACTER, PLOT, RELATIONSHIP, SETTING, TONE, CLARIFICATION, or FACT. For RELATIONSHIP, you MUST include a "connects_to" target.
2. THREADING (PRIORITY 2): Only after extracting all atomic memories, if some form a sequence, group their local_ids into a "thread".
3. KNOTTING: If storylines collide, create a "knot".

CRITICAL ENTITY NAMING RULES:
  - Use plain names: "Sebastian Carlisle", NOT "Dr. Sebastian Carlisle"
  - Entity is a LOOKUP KEY for indexing. Keep it short and consistent.

You MUST respond with ONLY a valid JSON object. No preamble, no explanation — pure JSON:
{{
  "memories": [
    {{"local_id": "m1", "entity": "Alistair", "category": "CHARACTER", "content": "Alistair is terrified of developing dementia."}},
    {{"local_id": "m2", "entity": "Alistair", "category": "RELATIONSHIP", "connects_to": "Sebastian", "content": "Alistair sees Sebastian's captivity as a medical necessity."}},
    {{"local_id": "m3", "entity": "Sebastian", "category": "TONE", "content": "Sebastian's dialogue should be sparse and dissociated."}}
  ],
  "threads": [
    {{"name": "Captivity Arc", "entity": "Sebastian", "type": "plot_line", "memory_local_ids": ["m2", "m3"]}}
  ],
  "knots": []
}}"""

    # v6.9.1 FIX: Proper error handling
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {openrouter_key}", "Content-Type": "application/json"},
            json={"model": MEMORY_MODEL_ID, "messages": [{"role": "user", "content": prompt.format(conversation=conversation)}], **MEMORY_PARAMS},
            timeout=60
        )
        if response.status_code != 200:
            print(f"[K2 CONV EXTRACT] API error {response.status_code}: {response.text[:300]}")
            return {"memories": [], "threads": [], "knots": []}, 0
            
        data = response.json()
        raw = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        cost = (usage.get("prompt_tokens", 0) * 0.55 + usage.get("completion_tokens", 0) * 2.20) / 1_000_000
        
        # v6.9.1: Use robust JSON extraction
        parsed = extract_json_from_response(raw)
        
        if not parsed:
            print(f"[K2 CONV EXTRACT] JSON parse failed. Raw preview: {raw[:500]}")
            return {"memories": [], "threads": [], "knots": []}, cost
        
        if "memories" in parsed:
            return parsed, cost
        
        for alt_key in ("results", "items", "data", "points", "extractions"):
            if alt_key in parsed:
                return {"memories": parsed[alt_key], "threads": parsed.get("threads", []), "knots": parsed.get("knots", [])}, cost
        
        print(f"[K2 CONV EXTRACT] No 'memories' key. Keys: {list(parsed.keys())}")
        return {"memories": [], "threads": [], "knots": []}, cost
        
    except Exception as e:
        print(f"[K2 CONV EXTRACT] Exception: {type(e).__name__}: {e}")
        traceback.print_exc()
        return {"memories": [], "threads": [], "knots": []}, 0

def call_openrouter(messages, api_key, mode="writing"):
    model = WRITING_MODEL_ID if mode == "writing" else MEMORY_MODEL_ID
    params = WRITING_PARAMS if mode == "writing" else MEMORY_PARAMS
    try:
        body = {"model": model, "messages": messages, **params}
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=body, timeout=120
        )
        if response.status_code != 200:
            return None, 0, 0, f"API Error: {response.status_code}"
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return content, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), None
    except Exception as e:
        return None, 0, 0, str(e)

class CostTracker:
    WRITING_INPUT = 2.50 / 1_000_000
    WRITING_OUTPUT = 15.00 / 1_000_000
    MEMORY_INPUT = 0.55 / 1_000_000
    MEMORY_OUTPUT = 2.20 / 1_000_000

    def __init__(self):
        if "total_cost" not in st.session_state:
            st.session_state.total_cost = 0.0
        if "message_count" not in st.session_state:
            st.session_state.message_count = 0

    def add_usage(self, input_tokens, output_tokens, mode="writing"):
        if mode == "writing":
            cost = (input_tokens * self.WRITING_INPUT) + (output_tokens * self.WRITING_OUTPUT)
        else:
            cost = (input_tokens * self.MEMORY_INPUT) + (output_tokens * self.MEMORY_OUTPUT)
        st.session_state.total_cost += cost
        st.session_state.message_count += 1
        return cost

def build_memory_context(prompt, mnemo_client, cross_session_enabled, use_loops, loop_manager, context_engine):
    context_parts = []
    metadata = {"sessions_found": 0, "skip_loops": False, "threads_injected": 0, "knots_injected": 0}
    if not cross_session_enabled:
        return "", metadata, False
    storage = get_persistent_storage()
    if not storage:
        return "", metadata, False

    prompt_lower = prompt.lower()
    asking_about_past = any(phrase in prompt_lower for phrase in [
        "last chat", "previous chat", "last conversation",
        "previous conversation", "earlier chat", "before this",
        "what did we talk", "what were we", "did we discuss",
        "remember when", "last time", "our last", "previous session",
        "talked about", "chatting about", "we discussed", "we were talking"
    ])
    current_session_id = st.session_state.get("current_session_id", "")

    if asking_about_past:
        metadata["skip_loops"] = True
        try:
            session_results = storage.search_sessions(prompt, current_session_id=current_session_id, limit=2)
            if session_results:
                metadata["sessions_found"] = len(session_results)
                context_parts.append("\n\n[PREVIOUS CHAT SESSIONS]\n---\n" + "\n---\n".join(session_results))
            else:
                recent = storage.get_previous_sessions_content(current_session_id=current_session_id, limit=2)
                if recent:
                    metadata["sessions_found"] = len(recent)
                    summaries = [f"Session '{s['title']}':\n{s['summary']}" for s in recent]
                    context_parts.append("\n\n[RECENT CHAT SESSIONS]\n---\n" + "\n---\n".join(summaries))
        except Exception:
            pass
    else:
        try:
            past_convos = storage.search_conversations(prompt, limit=3)
            if past_convos:
                context_parts.append("\n\n[PAST CONVERSATIONS]\n" + "\n".join(f"• {conv[:200]}" for conv in past_convos))
        except Exception:
            pass

        try:
            isolate = st.session_state.get("isolate_sessions", False)
            active_sessions = [current_session_id] if isolate and current_session_id else None

            # THREAD RETRIEVAL
            query_words = set(re.sub(r'[^\w\s]', '', prompt_lower).split())
            
            try:
                active_threads = mnemo_client.get_active_threads()
                if active_threads:
                    matched_threads = []
                    for t in active_threads:
                        t_name = t.get("name", "").lower()
                        t_name_words = set(t_name.split())
                        t_entity = t.get("entity", "").lower()
                        
                        if (t_entity and t_entity in prompt_lower) or \
                           (t_name and t_name in prompt_lower) or \
                           any(w in query_words for w in t_name_words if len(w) > 3):
                            matched_threads.append(t)
                    
                    if matched_threads:
                        thread_context_lines = []
                        for t in matched_threads:
                            tid = t.get("id")
                            tname = t.get("name")
                            pos = t.get("current_position", -1)
                            traced_cps = mnemo_client.trace_thread(tid, direction="back", steps=8, from_position=pos)
                            if traced_cps:
                                thread_context_lines.append(f"### NARRATIVE THREAD: {tname}")
                                for cp in traced_cps:
                                    entity = cp.get("entity", "")
                                    value = cp.get("value", "")
                                    pt = cp.get("point_type", "")
                                    conn = cp.get("connects_to", "")
                                    line = f"  - {entity} — {pt} → {conn}: {value}" if conn else f"  - {entity} — {pt}: {value}"
                                    thread_context_lines.append(line)
                                thread_context_lines.append("")
                        if thread_context_lines:
                            context_parts.append("\n\n[ACTIVE NARRATIVE THREADS - Chronological Order]\n" + "\n".join(thread_context_lines))
                            metadata["threads_injected"] = len(matched_threads)
            except Exception as thread_e:
                print(f"[WARN] Thread Retrieval failed: {thread_e}")

            # KNOT RETRIEVAL
            try:
                all_knots = mnemo_client.list_knots()
                if all_knots:
                    matched_knots = []
                    for k in all_knots:
                        k_name = k.get("name", "").lower()
                        k_reason = k.get("reason", "").lower()
                        k_name_words = set(k_name.split())
                        k_reason_words = set(k_reason.split())
                        
                        if (k_name and k_name in prompt_lower) or \
                           any(w in query_words for w in k_name_words if len(w) > 3) or \
                           any(w in query_words for w in k_reason_words if len(w) > 4):
                            matched_knots.append(k.get("id"))
                            
                    if matched_knots:
                        knot_context_lines = []
                        for kid in matched_knots:
                            k_ctx = mnemo_client.get_knot_context(kid)
                            if k_ctx:
                                knot_context_lines.append(f"### NARRATIVE KNOT (Collision Point): {k_ctx.get('name')}")
                                knot_context_lines.append(f"  - Pivot Type: {k_ctx.get('pivot_type')}")
                                knot_context_lines.append(f"  - Tension Shift: {k_ctx.get('tension_before')} ➔ {k_ctx.get('tension_after')}")
                                if k_ctx.get('tone_shift'):
                                    knot_context_lines.append(f"  - Tone Shift: {k_ctx.get('tone_shift')}")
                                knot_context_lines.append(f"  - Reason: {k_ctx.get('reason')}")
                                for tid, thread_data in k_ctx.get("thread_context", {}).items():
                                    t_name = thread_data.get("thread_name", tid)
                                    knot_context_lines.append(f"  - Thread arriving at Knot: {t_name}")
                                    for cp in thread_data.get("active_points", []):
                                        conn = f" ➔ {cp.get('connects_to')}" if cp.get("connects_to") else ""
                                        knot_context_lines.append(f"      • {cp.get('entity')} — {cp.get('point_type')}{conn}: {cp.get('value')}")
                                knot_context_lines.append("")
                        if knot_context_lines:
                            context_parts.append("\n\n[NARRATIVE KNOTS - Crucial Story Intersections]\n" + "\n".join(knot_context_lines))
                            metadata["knots_injected"] = len(matched_knots)
            except Exception as knot_e:
                print(f"[WARN] Knot Retrieval failed: {knot_e}")

            # GRAPH SEARCH
            cp_results = mnemo_client.graph_search(prompt, top_k=40, active_sessions=active_sessions)
            writing_keywords = ["write", "scene", "chapter", "story", "prose", "dialogue", "style", "continue", "next", "more", "go on", "keep going", "extend"]
            if any(kw in prompt_lower for kw in writing_keywords):
                style_cps = mnemo_client.graph_search("prose voice dialogue style tone", top_k=10, active_sessions=active_sessions)
                seen_ids = {r.get("id") for r in cp_results}
                for cp in style_cps:
                    if cp.get("id") not in seen_ids:
                        cp_results.append(cp)

            if cp_results:
                cp_context_lines = []
                for cp in cp_results:
                    entity = cp.get("entity", "")
                    value = cp.get("value", "")
                    cat = cp.get("category", "fact").upper()
                    conn = cp.get("connects_to", "")
                    pt = cp.get("point_type", "")
                    if conn:
                        line = f"[{cat}] {entity} — {pt} → {conn}: {value}"
                    elif entity and entity != "Story":
                        line = f"[{cat}] {entity} — {pt}: {value}"
                    else:
                        line = f"[{cat}] {value}"
                    cp_context_lines.append(line)
                context_parts.append("\n\n[DEEP CONTEXT]\n" + "\n".join(f"- {l}" for l in cp_context_lines))
                metadata["cp_results"] = len(cp_results)

        except Exception as e:
            metadata["cp_error"] = str(e)[:100]

    return "".join(context_parts), metadata, metadata["skip_loops"]

def handle_message(prompt, openrouter_key, mnemo_client):
    conversation_length = len(st.session_state.messages)
    cross_session_enabled = st.session_state.get("cross_session_enabled", True)
    use_loops = st.session_state.get("use_loops", True)
    isolate_sessions = st.session_state.get("isolate_sessions", False)
    current_session_id = st.session_state.get("current_session_id", "")

    gate = st.session_state.smart_memory.two_tier_gate(
        query=prompt,
        mnemo_client=mnemo_client if cross_session_enabled else None,
        conversation_length=conversation_length,
    )
    needs_memory = gate.should_retrieve

    search_query = prompt
    if gate.query_type.value == "continuation" and len(st.session_state.messages) > 0:
        last_ai_msg = st.session_state.messages[-1].get("content", "")
        search_query = f"{prompt}. Context: {last_ai_msg[-500:]}"

    past_conversation_context = ""
    skip_loops_for_this_query = False
    sessions_found = 0

    if needs_memory and cross_session_enabled:
        try:
            past_conversation_context, ctx_meta, skip_loops_for_this_query = build_memory_context(
                search_query, mnemo_client,
                cross_session_enabled=cross_session_enabled,
                use_loops=use_loops,
                loop_manager=st.session_state.get("loop_manager"),
                context_engine=st.session_state.get("context_engine"),
            )
            sessions_found = ctx_meta.get("sessions_found", 0)
        except Exception as e:
            past_conversation_context = ""
            cp_error = f"build_context_failed: {str(e)[:80]}"
            print(f"[WARN] build_memory_context failed: {e}")
            traceback.print_exc()

    full_system_prompt = SYSTEM_PROMPT + past_conversation_context

    if gate.query_type.value in ["creative", "continuation"]:
        full_system_prompt += "\n\n" + get_signal_instruction()
        full_system_prompt += (
            "\n\nCRITICAL PACING MANDATE: You are writing a book, not a summary. "
            "You MUST write a long, fully fleshed-out scene. DO NOT rush to the conclusion. "
            "Take your time expanding on micro-actions, sensory details, atmospheric tension, "
            "and internal monologue. Every line of dialogue must breathe. Make the output as lengthy and immersive as possible."
        )

    cached_brief = None
    if "prefetch_engine" in st.session_state and "signal_processor" in st.session_state:
        current_state = st.session_state.signal_processor.get_state()
        cached_brief = st.session_state.prefetch_engine.check_cache(current_state)

    if cached_brief:
        full_system_prompt = cached_brief.brief + "\n\n" + full_system_prompt

    should_use_loops = (needs_memory and use_loops and cross_session_enabled
                        and not skip_loops_for_this_query)

    active_sessions = [current_session_id] if isolate_sessions and current_session_id else None

    messages, context_stats = st.session_state.context_manager.build_optimized_context(
        system_prompt=full_system_prompt, query=prompt,
        conversation_history=st.session_state.messages,
        max_messages=MAX_CONVERSATION_MESSAGES, use_loops=should_use_loops,
        active_sessions=active_sessions,
    )

    context_meta = {
        "cross_session_memories_used": context_stats.get("memory_items_full", 0) + context_stats.get("memory_items_meta", 0),
        "context_tokens": context_stats.get("total_tokens", 0),
        "mode": gate.mode, "memory_reason": gate.reason,
        "sessions_found": sessions_found, "gate_tier": gate.tier_used,
        "threads_injected": ctx_meta.get("threads_injected", 0) if 'ctx_meta' in locals() else 0,
        "knots_injected": ctx_meta.get("knots_injected", 0) if 'ctx_meta' in locals() else 0,
        "cp_results": ctx_meta.get("cp_results", 0) if 'ctx_meta' in locals() else 0,
    }

    response, input_tokens, output_tokens, error = call_openrouter(messages, openrouter_key, mode="writing")

    if error:
        return None, error, {}

    clean_response, signals = parse_signals_from_response(response)
    
    clean_response = re.sub(r'<signal.*?>.*?(</signal>)?', '', clean_response, flags=re.IGNORECASE | re.DOTALL)
    clean_response = re.sub(r'["\']?\s*<signal.*', '', clean_response, flags=re.IGNORECASE | re.DOTALL)
    clean_response = re.sub(r'<abstract.*', '', clean_response, flags=re.IGNORECASE | re.DOTALL)
    clean_response = clean_response.strip()
    
    if signals and "signal_processor" in st.session_state:
        st.session_state.signal_processor.process(
            signals, mnemo_client, 
            session_id=current_session_id, 
            user_prompt=prompt
        )
        if "prefetch_engine" in st.session_state:
            st.session_state.prefetch_engine.trigger(
                st.session_state.signal_processor.get_state(),
                st.session_state.signal_processor.history,
                mnemo_client
            )
            
    response = clean_response

    cost_tracker = CostTracker()
    msg_cost = cost_tracker.add_usage(input_tokens, output_tokens, mode="writing")

    if not skip_loops_for_this_query:
        try:
            storage = get_persistent_storage(DEFAULT_HF_KEY, mnemo_client)
            if storage:
                storage.save_conversation_turn(prompt, response, current_session_id)
        except Exception:
            pass

    extracted = 0
    if (st.session_state.get("auto_extract", True)
            and cross_session_enabled
            and not skip_loops_for_this_query):
        conversation = f"User: {prompt}\n\nAssistant: {response}"
        extraction_data, extract_cost = extract_memories_with_gpt(conversation, openrouter_key)
        memories = extraction_data.get("memories", [])
        threads_data = extraction_data.get("threads", [])
        knots_data = extraction_data.get("knots", [])
        
        if memories:
            msg_cost += extract_cost
            local_to_real_cp = {}
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = {}
                for mem in memories:
                    local_id = mem.get("local_id", str(uuid.uuid4())[:8])
                    mem["local_id"] = local_id
                    cat = mem.get("category", "FACT").upper()
                    txt = mem.get("content", "")
                    entity = mem.get("entity", "Story")
                    connects_to = mem.get("connects_to", "")
                    if txt:
                        future = executor.submit(store_as_cp, mnemo_client, entity=entity, category=cat, content=txt, session_id=current_session_id, source="auto_extract", connects_to=connects_to)
                        futures[future] = mem
                        
                for future in concurrent.futures.as_completed(futures):
                    mem = futures[future]
                    try:
                        cp_id = future.result()
                        if cp_id:
                            extracted += 1
                            local_to_real_cp[mem["local_id"]] = cp_id
                            
                            cat_upper = mem.get("category", "FACT").upper()
                            ent = mem.get("entity", "Story")
                            conn = mem.get("connects_to", "")
                            val = mem.get("content", "")
                            
                            if conn:
                                loop_content = f"[{cat_upper}] {ent} → {conn}: {val}"
                            elif ent and ent != "Story":
                                loop_content = f"[{cat_upper}] {ent}: {val}"
                            else:
                                loop_content = f"[{cat_upper}] {val}"
                                
                            st.session_state.loop_manager.add_to_loop(content=loop_content, category=cat_upper.lower(), session_id=current_session_id, memory_id=cp_id)
                    except Exception:
                        pass
                        
            thread_name_to_id = {}
            for th in threads_data:
                t_name = th.get("name", "")
                if not t_name: continue
                t_id = "thread_" + re.sub(r'[^a-z0-9]', '_', t_name.lower())[:20]
                thread_name_to_id[t_name] = t_id
                cp_ids = [local_to_real_cp[m_id] for m_id in th.get("memory_local_ids", []) if m_id in local_to_real_cp]
                mnemo_client.add_thread(thread_id=t_id, name=t_name, entity=th.get("entity", ""), thread_type=th.get("type", "plot_line"), session_id=current_session_id, point_ids=cp_ids)
                
            for kn in knots_data:
                k_name = kn.get("name", "")
                if not k_name: continue
                k_id = "knot_" + re.sub(r'[^a-z0-9]', '_', k_name.lower())[:20]
                t_ids = [thread_name_to_id.get(tn, "thread_" + re.sub(r'[^a-z0-9]', '_', tn.lower())[:20]) for tn in kn.get("thread_names", [])]
                mnemo_client.add_knot(knot_id=k_id, name=k_name, thread_ids=t_ids, pivot_type="collision", reason=kn.get("reason", ""), session_id=current_session_id)

    result_meta = {
        "cross_session_memories_used": context_meta.get("cross_session_memories_used", 0),
        "context_tokens": context_meta.get("context_tokens", 0),
        "mode": context_meta.get("mode", "full"),
        "extracted": extracted, "cost": msg_cost,
        "sessions_found": sessions_found,
        "gate_tier": context_meta.get("gate_tier", 1),
        "threads_injected": context_meta.get("threads_injected", 0),
        "knots_injected": context_meta.get("knots_injected", 0),
        "cp_results": context_meta.get("cp_results", 0),
    }
    return response, None, result_meta

def render_sidebar(mnemo_client, openrouter_key, hf_key):
    with st.sidebar:
        st.markdown("#### ⚙️ Settings")

        with st.expander("🔑 API Keys", expanded=False):
            or_key = st.text_input("OpenRouter API Key", value=DEFAULT_OPENROUTER_KEY, type="password")
            hf = st.text_input("HuggingFace Token", value=DEFAULT_HF_KEY, type="password")

        openrouter_key = or_key or DEFAULT_OPENROUTER_KEY
        hf_key = hf or DEFAULT_HF_KEY

        st.divider()
        st.markdown("#### 💬 Chat")

        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            start_new_chat()
            st.rerun()

        st.divider()
        render_sessions_panel(mnemo_client, hf_key)
        st.divider()

        with st.expander("⚙️ File Upload & Settings", expanded=False):
            render_file_upload(mnemo_client, openrouter_key)

        st.divider()
        st.markdown("#### 🧠 Memory")

        cross_session_enabled = st.toggle("Cross-Session Memory", value=True,
                                          help="Remember across chat sessions")
        auto_extract = st.toggle("Auto-Extract Memories", value=True,
                                 help="Automatically extract facts from conversations")
        st.session_state.auto_extract = auto_extract
        st.session_state.cross_session_enabled = cross_session_enabled

        use_loops = st.toggle("🔄 Metadata Loops (Save 80% tokens)", value=True,
                              help="Use token-efficient context injection")
        st.session_state.use_loops = use_loops

        isolate_sessions = st.toggle("🔒 Session Memory Isolation", value=False,
                                     help="Only use memories from the current session")
        st.session_state.isolate_sessions = isolate_sessions

        if "loop_manager" in st.session_state and use_loops:
            loop_stats = st.session_state.loop_manager.get_stats()
            st.caption(f"📊 {loop_stats['total_items']} memories | {loop_stats['total_metadata_tokens']} tokens")

        st.divider()
        st.markdown("#### 📝 Add Memory")
        render_memory_management(mnemo_client)
        st.divider()
        render_consolidation_panel(openrouter_key)

        st.divider()

        st.markdown("#### 💰 Usage")
        msg_count = st.session_state.get('message_count', 0)
        total_cost = st.session_state.get('total_cost', 0)
        st.caption(f"{msg_count} messages · ${total_cost:.4f}")
        st.caption(f"✍️ {WRITING_MODEL_ID.split('/')[-1]} · 🧠 {MEMORY_MODEL_ID.split('/')[-1]}")

    return openrouter_key, hf_key

def render_sessions_panel(mnemo_client, hf_key):
    col_title, col_refresh = st.columns([3, 1])
    with col_title:
        st.markdown("#### 📚 Sessions")
    with col_refresh:
        if st.button("🔄", key="refresh_sessions", help="Refresh from cloud"):
            try:
                storage = get_persistent_storage(hf_key, mnemo_client)
                sessions = storage.load_sessions(limit=MAX_SESSIONS_STORED)
                st.session_state.session_history = sessions
                st.rerun()
            except Exception:
                pass

    if "session_folders" not in st.session_state:
        st.session_state.session_folders = {"📁 Default": []}

    with st.expander("📁 Manage Folders", expanded=False):
        with st.popover("➕ Create Folder"):
            new_folder = st.text_input("New folder name", placeholder="e.g., Story Ideas")
            if st.button("Create", use_container_width=True):
                if new_folder and new_folder.strip():
                    folder_name = f"📁 {new_folder.strip()}"
                    if folder_name not in st.session_state.session_folders:
                        st.session_state.session_folders[folder_name] = []
                        st.success(f"Created {folder_name}")
                        st.rerun()
        folders = list(st.session_state.session_folders.keys())
        if len(folders) > 1:
            with st.popover("🗑️ Delete Folder"):
                folder_to_delete = st.selectbox("Delete folder",
                    [""] + [f for f in folders if f != "📁 Default"])
                if folder_to_delete and st.button("Delete", type="primary", use_container_width=True):
                    st.session_state.session_folders["📁 Default"].extend(
                        st.session_state.session_folders.get(folder_to_delete, []))
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
                                rename_session_dialog(session_id, title)
                        with col3:
                            if st.button("📁", key=f"move_{session_id}", help="Move to folder"):
                                move_session_dialog(session_id)
                        with col4:
                            if st.button("🗑️", key=f"del_{session_id}"):
                                delete_session(session_id)
                                st.rerun()
                    st.caption("")
        else:
            st.caption("No previous sessions")

def render_memory_management(mnemo_client):
    with st.expander("Add manually", expanded=False):
        memory_category = st.selectbox("Category",
            ["CHARACTER", "PLOT", "SETTING", "THEME", "STYLE", "TONE", "FACT"])
        memory_entity = st.text_input("Entity (character name, 'Story', etc.)",
            placeholder="e.g., Alistair, Story, Red Rose Society", value="Story")
        memory_content = st.text_area("Content",
            placeholder="e.g., Alistair's scenes should feel clinical and cold, never melodramatic",
            height=80)
        if st.button("💾 Save", use_container_width=True):
            if memory_content.strip():
                current_session = st.session_state.get("current_session_id", "")
                cp_id = store_as_cp(mnemo_client, entity=memory_entity.strip() or "Story",
                                    category=memory_category, content=memory_content.strip(),
                                    session_id=current_session, source="manual")
                if cp_id:
                    st.success(f"✅ Saved [{memory_category}] for {memory_entity}")
                    
                    cat = memory_category.upper()
                    ent = memory_entity.strip() or "Story"
                    val = memory_content.strip()
                    if ent and ent != "Story":
                        loop_content = f"[{cat}] {ent}: {val}"
                    else:
                        loop_content = f"[{cat}] {val}"
                        
                    st.session_state.loop_manager.add_to_loop(
                        content=loop_content, category=memory_category.lower(),
                        session_id=current_session, memory_id=cp_id)
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
        stats = mnemo_client.get_stats()
        n_blobs = stats.get('total_memories', 0)
        n_cps = stats.get('total_connection_points', 0)
        st.caption(f"ConnectionPoints: {n_cps} | Legacy blobs: {n_blobs} | Links: {stats.get('total_links', 0)}")
        points = mnemo_client.list_points(limit=15)
        for cp in points:
            col1, col2 = st.columns([5, 1])
            with col1:
                entity = cp.get("entity", "?")
                value = cp.get("value", "")[:60]
                cat = cp.get("category", "?")
                conn = f" → {cp.get('connects_to')}" if cp.get("connects_to") else ""
                st.caption(f"**{entity}{conn}** [{cat}] {value}...")
            with col2:
                if st.button("🗑️", key=f"del_cp_{cp.get('id', '')}"):
                    if mnemo_client.delete_point(cp.get("id")):
                        st.rerun()
        if n_cps > 15:
            st.caption(f"... showing first 15 of {n_cps}. Click 'View All' for complete list")
        st.markdown("---")
        if st.button("🧹 Clear ALL Memories", use_container_width=True):
            st.session_state.confirm_clear = True
        if st.session_state.get("confirm_clear"):
            st.warning("⚠️ Delete ALL memories?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, delete all"):
                    mnemo_client.clear()
                    st.session_state.confirm_clear = False
                    st.rerun()
            with col2:
                if st.button("Cancel"):
                    st.session_state.confirm_clear = False
                    st.rerun()

    if st.session_state.get("show_all_memories"):
        st.markdown("---")
        st.subheader("📖 All ConnectionPoints")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("❌ Close", use_container_width=True, key="close_all_mem"):
                st.session_state.show_all_memories = False
                st.rerun()
        with col2:
            search_query = st.text_input("🔍 Search", key="mem_search", placeholder="Filter by entity or value...")
        with col3:
            category_filter = st.selectbox("Category",
                ["All", "character", "relationship", "plot", "tone", "setting", "style", "fact"],
                key="cat_filter")
        all_points = mnemo_client.list_points(limit=500)
        if search_query:
            q = search_query.lower()
            all_points = [p for p in all_points if q in p.get("entity", "").lower()
                          or q in p.get("value", "").lower()
                          or q in p.get("connects_to", "").lower()]
        if category_filter != "All":
            all_points = [p for p in all_points if p.get("category") == category_filter]
        st.caption(f"Showing {len(all_points)} ConnectionPoints")
        for cp in all_points:
            entity = cp.get("entity", "?")
            value = cp.get("value", "")
            cat = cp.get("category", "?")
            conn = cp.get("connects_to", "")
            cp_id = cp.get("id", "")
            col1, col2, col3, col4 = st.columns([2, 1, 6, 1])
            with col1:
                conn_str = f" → {conn}" if conn else ""
                st.caption(f"**{entity}{conn_str}**")
            with col2:
                st.caption(f"[{cat}]")
            with col3:
                st.text(value[:200])
            with col4:
                if st.button("🗑️", key=f"del_full_{cp_id}"):
                    if mnemo_client.delete_point(cp_id):
                        st.rerun()

def render_file_upload(mnemo_client, openrouter_key):
    st.markdown("**📎 Upload File → Memory**")
    st.caption("Upload a file — extracts facts, deep context, style, tone via K2")
    uploaded_file = st.file_uploader("Upload file", type=["txt", "md", "csv", "json", "pdf", "docx"],
                                     label_visibility="collapsed")
    if uploaded_file is not None:
        if st.button("🧠 Extract Deep Context + Memories", use_container_width=True):
            with st.spinner("Reading file..."):
                content = extract_text_from_file(uploaded_file)
            if content and not content.startswith("["):
                # v6.9.1: Auto-scaling — let calculate_extraction_plan decide
                plan = calculate_extraction_plan(content)
                n_chunks = plan["n_chunks"]
                st.info(f"📄 {plan['words']:,} words → ~{plan['estimated_memories']} memories → "
                        f"{n_chunks} call{'s' if n_chunks > 1 else ''} @ max_tokens={plan['max_tokens_per_call']:,}")
                with st.spinner(f"K2 extracting ({n_chunks} call{'s' if n_chunks > 1 else ''})..."):
                    extraction_data, cost, _ = extract_memories_from_file(content, uploaded_file.name, openrouter_key)
                memories = extraction_data.get("memories", [])
                threads_data = extraction_data.get("threads", [])
                knots_data = extraction_data.get("knots", [])
                
                if memories:
                    with st.spinner("Structuring Graph (CPs, Threads, Knots)..."):
                        current_session = st.session_state.get("current_session_id", "")
                        stored = 0
                        local_to_real_cp = {}
                        
                        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                            futures = {}
                            for mem in memories:
                                local_id = mem.get("local_id", str(uuid.uuid4())[:8])
                                mem["local_id"] = local_id
                                cat = mem.get("category", "FACT").upper()
                                txt = mem.get("content", "")
                                entity = mem.get("entity", "Story")
                                connects_to = mem.get("connects_to", "")
                                if txt:
                                    future = executor.submit(store_as_cp, mnemo_client, entity=entity, category=cat, content=txt, session_id=current_session, source="file_upload", connects_to=connects_to, weight=0.8)
                                    futures[future] = mem
                                    
                            for future in concurrent.futures.as_completed(futures):
                                mem = futures[future]
                                try:
                                    cp_id = future.result()
                                    if cp_id:
                                        stored += 1
                                        local_to_real_cp[mem["local_id"]] = cp_id
                                        
                                        cat_upper = mem.get("category", "FACT").upper()
                                        ent = mem.get("entity", "Story")
                                        conn = mem.get("connects_to", "")
                                        val = mem.get("content", "")
                                        if conn:
                                            loop_content = f"[{cat_upper}] {ent} → {conn}: {val}"
                                        elif ent and ent != "Story":
                                            loop_content = f"[{cat_upper}] {ent}: {val}"
                                        else:
                                            loop_content = f"[{cat_upper}] {val}"
                                            
                                        st.session_state.loop_manager.add_to_loop(content=loop_content, category=cat_upper.lower(), session_id=current_session, memory_id=cp_id)
                                except Exception:
                                    pass
                                    
                        thread_name_to_id = {}
                        for th in threads_data:
                            t_name = th.get("name", "")
                            if not t_name: continue
                            t_id = "thread_" + re.sub(r'[^a-z0-9]', '_', t_name.lower())[:20]
                            thread_name_to_id[t_name] = t_id
                            cp_ids = [local_to_real_cp[m_id] for m_id in th.get("memory_local_ids", []) if m_id in local_to_real_cp]
                            mnemo_client.add_thread(thread_id=t_id, name=t_name, entity=th.get("entity", ""), thread_type=th.get("type", "plot_line"), session_id=current_session, point_ids=cp_ids)
                            
                        for kn in knots_data:
                            k_name = kn.get("name", "")
                            if not k_name: continue
                            k_id = "knot_" + re.sub(r'[^a-z0-9]', '_', k_name.lower())[:20]
                            t_ids = [thread_name_to_id.get(tn, "thread_" + re.sub(r'[^a-z0-9]', '_', tn.lower())[:20]) for tn in kn.get("thread_names", [])]
                            mnemo_client.add_knot(knot_id=k_id, name=k_name, thread_ids=t_ids, pivot_type="collision", reason=kn.get("reason", ""), session_id=current_session)

                    facts = [m for m in memories if m.get("category") in ("CHARACTER", "PLOT", "SETTING", "THEME", "FACT")]
                    context = [m for m in memories if m.get("category") in ("CONTEXT", "CLARIFICATION", "RELATIONSHIP", "INSTRUCTION")]
                    style = [m for m in memories if m.get("category") in ("PROSE_SAMPLE", "DIALOGUE_SAMPLE", "VOICE", "VOCABULARY")]
                    tone = [m for m in memories if m.get("category") in ("TONE",)]
                    st.success(f"✅ Stored {stored} memories, {len(threads_data)} threads, {len(knots_data)} knots")
                    st.caption(f"📊 {len(facts)} facts · {len(context)} context · {len(style)} style · {len(tone)} tone | Cost: ${cost:.4f}")
                    with st.expander("View extracted memories", expanded=False):
                        for mem in memories:
                            st.caption(f"**[{mem.get('category', 'FACT')}]** {mem.get('content', '')[:150]}")
                else:
                    st.warning("No memories extracted. Try a different file.")
            else:
                st.error("Could not read file content")

def render_consolidation_panel(openrouter_key):
    with st.expander("🧠 Memory Consolidation", expanded=False):
        st.caption("Generates **new** CONTEXT, RELATIONSHIP, and TONE entries from your existing facts. Never removes existing memories — purely additive.")
        last_consol = st.session_state.get("last_consolidation")
        if last_consol:
            st.caption(f"Last run: {last_consol[:16]}")

        try:
            mnemo_client = st.session_state.get("mnemo_client")
            if mnemo_client:
                stats = mnemo_client.get_stats()
                n_cps = stats.get("total_connection_points", 0)
                n_blobs = stats.get("total_memories", 0)
                cp_cats = stats.get("cp_by_category", {})
                fact_cats = sum(v for k, v in cp_cats.items() if k not in ("relationship", "tone"))
                context_cats = cp_cats.get("relationship", 0)
                st.caption(f"📊 {n_cps} CPs ({fact_cats} facts · {context_cats} relationships) · {n_blobs} legacy blobs")
        except Exception:
            pass

        if st.button("🧠 Generate Deep Context", use_container_width=True):
            with st.spinner("K2 analyzing memories... (generates new entries, never removes existing)"):
                result = st.session_state.context_engine.consolidate_memories(openrouter_key)
                if result.get("error"):
                    st.error(f"Error: {result['error']}")
                else:
                    st.session_state.last_consolidation = result["timestamp"]
                    created = result['created']
                    k2_returned = result.get('k2_returned', '?')
                    dedup_rejected = result.get('dedup_rejected', 0)
                    if created > 0:
                        st.success(f"✅ Created {created} new context entries! (existing memories untouched)")
                    else:
                        st.warning(f"K2 returned {k2_returned} entries but {dedup_rejected} were duplicates. Try again — K2 may generate different insights.")
                    st.caption(f"Analyzed: {result['memories_analyzed']} memories | K2 returned: {k2_returned} | Dedup rejected: {dedup_rejected} | Stored: {created} | Cost: ${result['cost']:.4f}")
                    if result.get("new_entries"):
                        with st.expander("New entries created"):
                            for entry in result["new_entries"]:
                                st.caption(f"[{entry['category']}] {entry['content']}")
                    st.session_state.loop_manager.load_from_mnemo(use_smart_extraction=False)

def render_chat(openrouter_key, mnemo_client):
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                col1, col2 = st.columns([8, 1])
                with col1:
                    if "metadata" in message:
                        meta = message["metadata"]
                        memory_info = []
                        if meta.get("threads_injected", 0) > 0:
                            memory_info.append(f"🧵 {meta['threads_injected']} threads")
                        if meta.get("knots_injected", 0) > 0:
                            memory_info.append(f"🪢 {meta['knots_injected']} knots")
                        if meta.get("cross_session_memories_used", 0) > 0:
                            memory_info.append(f"📚 {meta['cross_session_memories_used']} memories")
                        if meta.get("cp_results", 0) > 0:
                            memory_info.append(f"🔗 {meta['cp_results']} CPs")
                        mode = meta.get("mode", "")
                        if mode == "t1_direct":
                            memory_info.append(f"🔄 {meta.get('context_tokens', 0)} tokens")
                        elif mode == "t2_confirmed":
                            memory_info.append(f"🔄 t2 | {meta.get('context_tokens', 0)} tokens")
                        elif mode == "skip":
                            memory_info.append("⚡ fast")
                        if meta.get("extracted", 0) > 0:
                            memory_info.append(f"🧠 {meta['extracted']} extracted")
                        if meta.get("cost"):
                            memory_info.append(f"💰 ${meta['cost']:.4f}")
                        if memory_info:
                            st.caption(" | ".join(memory_info))
                with col2:
                    if st.button("📋", key=f"copy_{idx}", help="Copy response"):
                        copy_response_dialog(message["content"])

    if prompt := st.chat_input("Write a scene, ask a question, or just say hi..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner(""):
                response, error, result_meta = handle_message(prompt, openrouter_key, mnemo_client)
            if error:
                st.error(error)
                st.session_state.messages.pop()
            else:
                st.markdown(response)
                meta_parts = []
                if result_meta.get("sessions_found", 0) > 0:
                    meta_parts.append(f"📜 {result_meta['sessions_found']} past chats")
                if result_meta.get("threads_injected", 0) > 0:
                    meta_parts.append(f"🧵 {result_meta['threads_injected']} threads")
                if result_meta.get("knots_injected", 0) > 0:
                    meta_parts.append(f"🪢 {result_meta['knots_injected']} knots")
                if result_meta.get("cross_session_memories_used", 0) > 0:
                    meta_parts.append(f"📚 {result_meta['cross_session_memories_used']} memories")
                if result_meta.get("cp_results", 0) > 0:
                    meta_parts.append(f"🔗 {result_meta['cp_results']} CPs")
                mode = result_meta.get("mode", "")
                if mode == "t1_direct":
                    meta_parts.append(f"🔄 {result_meta.get('context_tokens', 0)} tokens")
                elif mode == "t2_confirmed":
                    meta_parts.append(f"🔄 t2 | {result_meta.get('context_tokens', 0)} tokens")
                elif mode == "skip":
                    meta_parts.append("⚡ fast")
                if result_meta.get("extracted", 0) > 0:
                    meta_parts.append(f"🧠 {result_meta['extracted']} extracted")
                meta_parts.append(f"💰 ${result_meta.get('cost', 0):.4f}")
                st.caption(" | ".join(meta_parts))
                st.session_state.messages.append({
                    "role": "assistant", "content": response, "metadata": result_meta,
                })
                save_current_session()

def main():
    st.set_page_config(page_title="4o with Memory", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap');
    .stApp { font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif; }
    .stApp > header { background: transparent !important; }
    #MainMenu, footer, .stDeployButton { display: none !important; }
    .stChatMessage { border-radius: 16px !important; padding: 1rem 1.25rem !important; margin-bottom: 0.5rem !important; border: 1px solid rgba(128, 128, 128, 0.08) !important; backdrop-filter: blur(8px); font-size: 0.95rem; line-height: 1.7; letter-spacing: 0.01em; }
    .stChatMessage[data-testid="chat-message-user"] { background: rgba(99, 102, 241, 0.06) !important; border-left: 3px solid rgba(99, 102, 241, 0.4) !important; }
    .stChatMessage[data-testid="chat-message-assistant"] { background: rgba(255, 255, 255, 0.02) !important; }
    .stChatInput > div { border-radius: 24px !important; border: 1px solid rgba(128, 128, 128, 0.15) !important; transition: border-color 0.2s, box-shadow 0.2s; }
    .stChatInput > div:focus-within { border-color: rgba(99, 102, 241, 0.5) !important; box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.08) !important; }
    .stChatInput textarea { font-family: 'DM Sans', sans-serif !important; font-size: 0.95rem !important; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, rgba(15, 15, 20, 0.97) 0%, rgba(20, 20, 30, 0.95) 100%) !important; border-right: 1px solid rgba(128, 128, 128, 0.08) !important; }
    section[data-testid="stSidebar"] .stMarkdown h1, section[data-testid="stSidebar"] .stMarkdown h2, section[data-testid="stSidebar"] .stMarkdown h3 { font-family: 'DM Sans', sans-serif; font-weight: 600; letter-spacing: -0.02em; }
    section[data-testid="stSidebar"] hr { border-color: rgba(128, 128, 128, 0.1) !important; margin: 0.75rem 0 !important; }
    .stButton > button { border-radius: 12px !important; font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important; font-size: 0.85rem !important; letter-spacing: 0.02em; transition: all 0.15s ease; border: 1px solid rgba(128, 128, 128, 0.12) !important; }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15); }
    .stButton > button[kind="primary"] { background: linear-gradient(135deg, #6366f1 0%, #818cf8 100%) !important; border: none !important; color: white !important; }
    .stButton > button[kind="primary"]:hover { background: linear-gradient(135deg, #5558e6 0%, #7578f0 100%) !important; }
    .stToggle label span { font-size: 0.85rem !important; }
    .streamlit-expanderHeader { font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important; font-size: 0.85rem !important; border-radius: 10px !important; }
    .stChatMessage .stCaption, .stChatMessage caption { font-family: 'JetBrains Mono', monospace !important; font-size: 0.7rem !important; opacity: 0.5; letter-spacing: 0.03em; transition: opacity 0.2s; }
    .stChatMessage:hover .stCaption, .stChatMessage:hover caption { opacity: 0.8; }
    .stSpinner > div { border-radius: 12px !important; }
    .stChatMessage pre { border-radius: 10px !important; font-family: 'JetBrains Mono', monospace !important; font-size: 0.85rem !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; font-size: 0.8rem; font-family: 'DM Sans', sans-serif; }
    .stAlert { border-radius: 12px !important; }
    .brand-header { display: flex; align-items: center; gap: 0.75rem; padding: 0.25rem 0 0.5rem; }
    .brand-header .brand-icon { font-size: 1.8rem; line-height: 1; }
    .brand-header .brand-text h1 { font-family: 'DM Sans', sans-serif; font-size: 1.5rem; font-weight: 600; letter-spacing: -0.03em; margin: 0; padding: 0; line-height: 1.2; }
    .brand-header .brand-text p { font-family: 'DM Sans', sans-serif; font-size: 0.75rem; opacity: 0.45; margin: 0.15rem 0 0; letter-spacing: 0.04em; text-transform: uppercase; font-weight: 500; }
    .status-bar { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; opacity: 0.35; letter-spacing: 0.05em; text-align: center; padding: 0.5rem 0; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="brand-header">
        <div class="brand-icon">🧠</div>
        <div class="brand-text">
            <h1>4o with Memory</h1>
            <p>GPT-4o writer · K2 memory curator · Mnemo v6.9.1</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

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

    mnemo_client = init_client(DEFAULT_HF_KEY)

    if "blobs_auto_converted" not in st.session_state:
        st.session_state.blobs_auto_converted = True
        try:
            converted = auto_convert_blobs(mnemo_client)
            if converted > 0:
                st.toast(f"🔄 Auto-migrated {converted} legacy memories → ConnectionPoints")
        except Exception:
            pass

    if "session_history_loaded" not in st.session_state:
        st.session_state.session_history_loaded = True
        st.session_state.session_history = []
        try:
            storage = get_persistent_storage(DEFAULT_HF_KEY, mnemo_client)
            if storage:
                sessions = storage.load_sessions(limit=MAX_SESSIONS_STORED)
                if sessions:
                    st.session_state.session_history = sessions
                if hasattr(storage, 'load_folder_state'):
                    st.session_state.session_folders = storage.load_folder_state()
        except Exception:
            pass

    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = generate_session_id()
    if "custom_titles" not in st.session_state:
        st.session_state.custom_titles = {}
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "loop_manager" not in st.session_state:
        st.session_state.loop_manager = LoopManager(mnemo_client=mnemo_client, openrouter_key=DEFAULT_OPENROUTER_KEY)
        st.session_state.loop_manager.load_from_mnemo(use_smart_extraction=False)
    if "smart_memory" not in st.session_state:
        st.session_state.smart_memory = SmartMemory()
    if "context_engine" not in st.session_state:
        st.session_state.context_engine = ContextEngine(mnemo_client=mnemo_client, openrouter_key=DEFAULT_OPENROUTER_KEY)
    
    if "context_manager" not in st.session_state:
        st.session_state.context_manager = ContextWindowManager(loop_manager=st.session_state.loop_manager)
    else:
        st.session_state.context_manager.set_loop_manager(st.session_state.loop_manager)
        
    if "signal_processor" not in st.session_state:
        st.session_state.signal_processor = SignalProcessor()
        try:
            st.session_state.signal_processor.update_threads_from_server(mnemo_client)
        except Exception:
            pass
            
    if "prefetch_engine" not in st.session_state:
        st.session_state.prefetch_engine = PrefetchEngine()

    openrouter_key, hf_key = render_sidebar(mnemo_client, DEFAULT_OPENROUTER_KEY, DEFAULT_HF_KEY)

    if mnemo_client.token != hf_key:
        mnemo_client.token = hf_key
        mnemo_client.session.headers.update({"Authorization": f"Bearer {hf_key}"})

    render_chat(openrouter_key, mnemo_client)

    st.markdown(f"""
    <div class="status-bar">
        4o with Memory v6.9.1 &nbsp;·&nbsp; GPT-4o + K2 + Mnemo &nbsp;·&nbsp; Threads & Knots &nbsp;·&nbsp; Graph Search
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

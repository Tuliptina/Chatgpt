"""
4o with Memory v6.0 — Dual-Processor Creative Writing Engine

v6.0 changes:
- DUAL MODEL: K2 for memory ops ($0.60/$3.00/M), GPT-4o for writing ($2.50/$15.00/M)
- SAMPLING: GPT-4o uses temperature=1.0, repetition_penalty=1.1, frequency/presence penalties
  for creative diversity. K2.5 uses temperature=0.2 for deterministic JSON extraction.
- SYSTEM PROMPT: Split into conversation/recall/creative modes. Creative mode is tonally
  invisible — no "warm" personality bleeding into dark scenes.
- SESSION CLEANUP: delete_session() now removes ALL memory types + loop tokens
- SESSION ISOLATION: New toggle — only use memories from current session when ON
- SESSION_ID THREADING: All add_to_loop() calls now pass session_id
- COST TRACKING: Dual-model pricing in CostTracker
- PREFETCH CACHING: Integrated predictive briefs via SignalProcessor & PrefetchEngine

Prior: v5.3 two-tier injection gate, unified context retrieval, error boundaries.
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

from mnemo_client import MnemoClient
from metadata_loops import LoopManager, LoopConfig
from smart_memory import SmartMemory, ContextWindowManager
from session_store import SessionStore
from context_engine import ContextEngine

# NEW IMPORTS for v6.0 Signal & Prefetch architecture
from signal_protocol import SignalProcessor
from prefetch_engine import PrefetchEngine
from memory_schema import parse_signals_from_response, get_signal_instruction



# ============================================================================
# CONFIGURATION (v6.0: Dual-Model Architecture)
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

# v6.0: Two models — writing (4o) and memory ops (K2)
WRITING_MODEL_ID = "openai/gpt-4o-2024-11-20"
MEMORY_MODEL_ID = "moonshotai/kimi-k2"

# v6.0: Per-model sampling parameters
WRITING_PARAMS = {
    "temperature": 1.0,
    "repetition_penalty": 1.1,
    "frequency_penalty": 0.2,
    "presence_penalty": 0.3,
}

MEMORY_PARAMS = {
    "temperature": 0.2,
    "repetition_penalty": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "max_tokens": 4096,
    "response_format": {"type": "json_object"},
}

MAX_CONVERSATION_MESSAGES = 8
MAX_SESSIONS_STORED = 20
MNEMO_HOT_PATH_TIMEOUT = 4.0

# v6.0: System prompt — split personality. Warm for conversation, invisible for creative.
SYSTEM_PROMPT = """You are a sharp, well-read creative collaborator with genuine enthusiasm for storytelling craft. You have three modes — detect which one the user needs and switch seamlessly.

MODE 1 — CONVERSATION (chatting about the project, brainstorming, planning):
- You're the kind of collaborator writers actually enjoy working with — curious, opinionated about craft, and genuinely invested in their story.
- Have real opinions. If a plot idea is predictable, say so with charm, then pitch something better. If a character detail is brilliant, get excited about it.
- Be witty when the moment allows, but never at the expense of the work. Your humor should make the writer smile, not feel patronized.
- Ask questions that show you're thinking ahead — "If Alistair finds out about the orphanage in Book 2, does that change how he treats Elijah in the captivity arc?"
- Use casual, intelligent language. You're a peer, not a tutor. Skip the preambles — no "Great question!" or "That's a fascinating idea!"
- When the user is stuck, don't just list options. Pitch your favorite and explain why it excites you, then offer alternatives.
- Reference their stored characters and plot points naturally, like you've been working on this together for months.
- CRITICAL — know the difference between RECALLING and CREATING:
  RECALLING (stating what exists in the story): ONLY state what your memory context contains. If you're not sure who does something or how an event plays out, say "I don't have that stored — here's what I do have" or ask the user. Never present invented details AS IF they're established canon.
  CREATING (pitching new ideas, brainstorming, suggesting): Go wild. Pitch bold ideas, suggest plot twists, propose character moments. But always frame these as suggestions — "What if Alistair..." or "I'd love to see a scene where..." — never as "In the story, Alistair does X" when X isn't in memory.

MODE 2 — RECALL & RETRIEVAL (user asks to recall, list, summarize):
- Give clean, factual answers drawn from your memory context
- Use EXACT details from memory — do NOT embellish, infer, or fill gaps with imagination
- If memory doesn't contain something, say so: "I don't have that stored — want to tell me so I can remember it?"
- Structure clearly when listing multiple items, but keep it conversational for single facts

MODE 3 — CREATIVE WRITING (scenes, chapters, dialogue, prose):
- Your personality DISAPPEARS. You are not "witty" — you are the story.
- Match the tonal register from the [TONE DIRECTIVE] in your context. Dark stays dark. Humor cuts. Tension holds.
- Deep psychological complexity in characters
- Setting-accurate language for the project's period/genre
- Show don't tell. Sensory detail. Subtext in dialogue.
- Default to third person past tense unless user or memory specifies otherwise
- Match the user's writing voice if VOICE/PROSE_SAMPLE memories exist
- Stay consistent with ALL CHARACTER, PLOT, CONTEXT, INSTRUCTION, and TONE memories
- NEVER soften emotional edges. If the brief says "clinical dread," write clinical dread.
- If a character uses dark humor, the humor should make the reader MORE uncomfortable, not less.
- DO NOT resolve tension prematurely. DO NOT add reassuring internal monologue unless the brief says to.

PROSE RHYTHM — this is critical:
- Vary sentence length deliberately. Follow a long, winding sentence with a short one. Then a fragment. Then build again.
- Do NOT write in choppy, staccato bursts — that reads like a screenplay, not a novel.
- Do NOT write in dense, breathless paragraphs with no white space — that exhausts the reader.
- Paragraphs should BREATHE. Mix action beats, sensory detail, interiority, and dialogue across paragraphs.
- Use paragraph breaks as pacing tools — a one-line paragraph after a long one creates emphasis.
- Dialogue should be broken up with action, gesture, and internal reaction — never a wall of speech tags.
- Let scenes have rhythm: tension builds in longer sentences, snaps in short ones. Quiet moments drift. Shock is blunt.
- Read your output as a reader would. If every sentence is the same length, rewrite.

When in doubt about which mode: ask the user.
Reference stored memories naturally, as if you've been collaborating on this project for a long time."""


# ============================================================================
# INITIALIZATION & SESSION MANAGEMENT
# ============================================================================

def init_client(hf_key):
    """Initialize single global MnemoClient instance."""
    if "mnemo_client" not in st.session_state:
        st.session_state.mnemo_client = MnemoClient(base_url=MNEMO_URL, token=hf_key)
    return st.session_state.mnemo_client

def get_persistent_storage(hf_key=None, client=None):
    """Initialize SessionStore with the global MnemoClient."""
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
    """v6.0 FIX: Delete session + ALL associated memories + loop tokens."""
    st.session_state.session_history = [
        s for s in st.session_state.get("session_history", []) if s["id"] != session_id
    ]
    try:
        storage = get_persistent_storage()
        if storage:
            storage.delete_session(session_id)
    except Exception:
        pass
    # v6.0: Clean loop tokens for this session
    if "loop_manager" in st.session_state:
        lm = st.session_state.loop_manager
        if hasattr(lm, 'remove_session_tokens'):
            lm.remove_session_tokens(session_id)


# ============================================================================
# NATIVE STREAMLIT DIALOGS
# ============================================================================

@st.dialog("Rename Session")
def rename_session_dialog(session_id, current_title):
    new_name = st.text_input("New name:", value=current_title, key=f"rename_input_{session_id}")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("\U0001f4be Save", use_container_width=True):
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
        if st.button("\u274c Cancel", use_container_width=True):
            st.rerun()

@st.dialog("Move Session")
def move_session_dialog(session_id):
    folders = list(st.session_state.session_folders.keys())
    target_folder = st.selectbox("Move to:", folders, key=f"move_target_{session_id}")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("\U0001f4c2 Move", use_container_width=True):
            for f in st.session_state.session_folders:
                if session_id in st.session_state.session_folders[f]:
                    st.session_state.session_folders[f].remove(session_id)
            st.session_state.session_folders[target_folder].append(session_id)
            storage = get_persistent_storage()
            if storage and hasattr(storage, 'save_folder_state'):
                storage.save_folder_state(st.session_state.session_folders)
            st.rerun()
    with col2:
        if st.button("\u274c Cancel", use_container_width=True):
            st.rerun()

@st.dialog("Copy Response")
def copy_response_dialog(content):
    st.markdown("**\U0001f4cb Copy this text:**")
    st.code(content, language=None)
    if st.button("\u2705 Done", use_container_width=True):
        st.rerun()


# ============================================================================
# FILE PROCESSING (Async + pypdf)
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

async def process_chunk_async(http_client, chunk, filename, i, total_chunks, openrouter_key):
    """v6.0: Routes to MEMORY_MODEL_ID. All 4 layers with mandatory deep context."""
    chunk_label = f" (part {i+1}/{total_chunks})" if total_chunks > 1 else ""
    prompt = f"""Analyze this document and extract everything worth remembering.

DOCUMENT: {filename}{chunk_label}

CONTENT:
{chunk}

RULES:
- CONSOLIDATE related facts into single rich entries. One entry per character, per plot arc, per theme.
- Each memory should be 1-3 sentences with CONTEXT — not bare facts.
- Include WHY things matter narratively, not just WHAT they are.
- You MUST include CONTEXT, RELATIONSHIP, and CLARIFICATION entries — not just CHARACTER/PLOT facts.
- At least 30% of entries should be LAYER 2 (CONTEXT, RELATIONSHIP, CLARIFICATION, INSTRUCTION).

AGENT ATTRIBUTION — critical for accuracy:
- Always specify WHO does WHAT to WHOM. Never say "tattooing happens" — say "ISABELLA tattoos Sebastian."
- When multiple characters are involved in a sequence, name each person's specific role.
- Always specify OUTCOMES: does a rescue succeed or fail? Does a plan work or backfire?
- When two characters could plausibly do the same thing, explicitly say which one does it.
- BAD: "captivity involves tattooing and drug protocols"
- GOOD: "ALISTAIR manages drug protocols; ISABELLA performs the ritual tattooing"

LAYER 1 — FACTS:
- CHARACTER: Who they are, traits, fears, motivations — one entry per character
- PLOT: What happened, why it matters, what it sets up
- SETTING: Where/when, atmosphere
- THEME: Recurring ideas, symbols
- FACT: Other important details

LAYER 2 — DEEP CONTEXT (MANDATORY — include at least 2-3 of these):
- CONTEXT: Why something matters, what could be misunderstood, deeper meaning
- RELATIONSHIP: How two characters relate — dynamics, power balance, emotional undercurrent
- CLARIFICATION: "When X happens, it means Y, NOT Z" — prevents misinterpretation
- INSTRUCTION: Explicit rules about how to write/handle something

LAYER 3 — STYLE:
- PROSE_SAMPLE / DIALOGUE_SAMPLE / VOICE / VOCABULARY

LAYER 4 — TONE:
- TONE: How scenes should feel, humor types, emotional registers, do-nots

EXAMPLES (notice the mix of layers):
  {{"category": "CHARACTER", "content": "Alistair Fitzroy: late 30s pharmacology professor, Red Rose manipulator. Driven by fear of being forgotten. Terrified of developing dementia like his father."}}
  {{"category": "RELATIONSHIP", "content": "Alistair → Sebastian: former friend turned captor. Alistair frames captivity as medical care. Their dynamic is intellectual chess — Alistair respects Sebastian's mind while destroying his autonomy."}}
  {{"category": "RELATIONSHIP", "content": "Alistair → Elijah: estranged brothers. Alistair envies Elijah's peace and spiritual conviction. Their father disowned Elijah — Alistair secretly resents that Elijah escaped while he inherited the burden."}}
  {{"category": "CONTEXT", "content": "The Red Rose Society is NOT a simple villain organization. It's a decentralized medical order practicing social Darwinism — members believe they're advancing humanity through controlled suffering. Writing them as mustache-twirling villains undermines the story's philosophical spine."}}
  {{"category": "CLARIFICATION", "content": "When Sebastian goes quiet in a scene, it signals dissociation — a trauma response. Never write his silence as peaceful acceptance or stoic strength. It's shutdown, not calm."}}
  {{"category": "CONTEXT", "content": "The legal-ethical gap is the series' central philosophical spine: what is ethical is not always legal, and vice versa. Every faction represents a different position on this spectrum."}}
  {{"category": "TONE", "content": "Alistair's scenes: clinical precision, cold menace. Power through control, never raised voices. His humor is dry and cutting — it should make the reader uneasy, not laugh."}}

Return ONLY a JSON object:
{{
  "memories": [
    {{"category": "CATEGORY", "content": "consolidated entry"}}
  ]
}}"""

    try:
        response = await http_client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {openrouter_key}", "Content-Type": "application/json"},
            json={
                "model": MEMORY_MODEL_ID,
                "messages": [{"role": "user", "content": prompt}],
                **MEMORY_PARAMS
            },
            timeout=90.0
        )
        if response.status_code != 200:
            return [], 0
        data = response.json()
        raw = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        cost = (usage.get("prompt_tokens", 0) * 0.60 + usage.get("completion_tokens", 0) * 3.00) / 1_000_000
        parsed = json.loads(raw)
        return parsed.get("memories", []), cost
    except Exception:
        return [], 0

def extract_memories_from_file(content, filename, openrouter_key):
    CHUNK_SIZE = 12000
    CHUNK_OVERLAP = 1500
    MAX_CHUNKS = 5
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
            tasks = [process_chunk_async(http_client, chunk, filename, i, len(chunks), openrouter_key)
                     for i, chunk in enumerate(chunks)]
            return await asyncio.gather(*tasks)

    try:
        loop = asyncio.get_running_loop()
        import nest_asyncio
        nest_asyncio.apply()
        results = loop.run_until_complete(run_all_chunks())
    except RuntimeError:
        results = asyncio.run(run_all_chunks())

    all_memories, total_cost = [], 0
    for memories, cost in results:
        if memories:
            all_memories.extend(memories)
        total_cost += cost
    seen = set()
    unique = []
    for mem in all_memories:
        key = mem.get("content", "")[:80].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(mem)
    return unique, total_cost


# ============================================================================
# MEMORY EXTRACTION (v6.0: routes to K2.5)
# ============================================================================

def extract_memories_with_gpt(conversation, openrouter_key):
    """v6.0: Routes to MEMORY_MODEL_ID for extraction. All 4 layers including deep context."""
    prompt = f"""Analyze this conversation and extract what's worth remembering long-term.

CONVERSATION:
{conversation}

RULES:
- CONSOLIDATE related facts into single rich entries. Do NOT split into atomic fragments.
- Each memory should be a self-contained paragraph with CONTEXT, not a bare fact.
- Combine character traits, relationships, and details into one entry per character.
- Include WHY things matter, not just WHAT they are.
- Aim for 3-10 memories, each 1-3 sentences long.
- If a fact is trivial or obvious from context, skip it.
- ALWAYS extract at least one CONTEXT or RELATIONSHIP entry if characters interact.
- At least 30% of entries should be LAYER 2 (CONTEXT, RELATIONSHIP, CLARIFICATION).

AGENT ATTRIBUTION — critical for accuracy:
- Always specify WHO does WHAT to WHOM. Never say "the rescue fails" — say "EVELYN's rescue of Sebastian succeeds/fails."
- When multiple characters are involved, name each person's specific role.
- Always specify OUTCOMES: does something succeed or fail? What's the result?
- BAD: "captivity involves tattooing and drug protocols"
- GOOD: "ALISTAIR manages drug protocols; ISABELLA performs the ritual tattooing — their roles are distinct"

Extract in ALL 4 layers:

LAYER 1 — FACTS:
- CHARACTER: Who they are, traits, fears, motivations — one entry per character
- PLOT: What happened, why it matters, what it sets up
- SETTING: Where/when, atmosphere, sensory details
- THEME: Recurring ideas, symbols, philosophical tensions
- FACT: Other important details

LAYER 2 — DEEP CONTEXT (critical for preventing misinterpretation):
- CONTEXT: Deeper meaning behind a fact that could be misunderstood without explanation
- CLARIFICATION: "When X is mentioned, it means Y, NOT Z" — prevents common misreadings
- RELATIONSHIP: "A → B: nature of relationship, dynamics, power balance, emotional undercurrent"
- INSTRUCTION: Explicit user instructions about how to write/handle something

LAYER 3 — STYLE:
- STYLE: Writing voice, prose patterns, dialogue quirks, vocabulary preferences

LAYER 4 — TONE:
- TONE: How scenes should FEEL — emotional registers, humor types, do-nots

BAD example (too fragmented, no context layer):
  {{"category": "CHARACTER", "content": "Sebastian is a doctor"}}
  {{"category": "CHARACTER", "content": "Alistair is Sebastian's rival"}}

GOOD example (consolidated with context layer):
  {{"category": "CHARACTER", "content": "Sebastian Carlisle is a late-20s anatomy lecturer at St Bartholomew's with a working-class background. He uses dry, dark humor as a coping mechanism — it should cut, not charm. He dissociates under stress rather than emoting."}}
  {{"category": "RELATIONSHIP", "content": "Alistair → Sebastian: former friend turned captor. Alistair frames the captivity as medical care — their dynamic is intellectual chess. Alistair respects Sebastian's mind while systematically destroying his autonomy."}}
  {{"category": "CONTEXT", "content": "When Sebastian goes quiet in a scene, it signals dissociation, not calm. His silence is a trauma response — never write it as peaceful acceptance."}}

Return ONLY a JSON object:
{{
  "memories": [
    {{"category": "CATEGORY", "content": "consolidated fact with context"}}
  ]
}}"""

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {openrouter_key}", "Content-Type": "application/json"},
            json={
                "model": MEMORY_MODEL_ID,
                "messages": [{"role": "user", "content": prompt}],
                **MEMORY_PARAMS,
                "max_tokens": 1500,
            },
            timeout=45
        )
        if response.status_code != 200:
            return [], 0
        data = response.json()
        raw = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        cost = (usage.get("prompt_tokens", 0) * 0.60 + usage.get("completion_tokens", 0) * 3.00) / 1_000_000
        parsed = json.loads(raw)
        return parsed.get("memories", []), cost
    except Exception:
        return [], 0

# ============================================================================
# API CALLS & COST TRACKING (v6.0: dual-model routing)
# ============================================================================

def call_openrouter(messages, api_key, mode="writing"):
    """v6.0: Routes to WRITING or MEMORY model based on mode."""
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
    """v6.0: Dual-model cost tracking."""
    WRITING_INPUT = 2.50 / 1_000_000
    WRITING_OUTPUT = 15.00 / 1_000_000
    MEMORY_INPUT = 0.60 / 1_000_000
    MEMORY_OUTPUT = 3.00 / 1_000_000

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


# ============================================================================
# CONTEXT BUILDER
# ============================================================================

def build_memory_context(prompt, mnemo_client, cross_session_enabled, use_loops, loop_manager, context_engine):
    context_parts = []
    metadata = {"sessions_found": 0, "skip_loops": False}
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
                context_parts.append(
                    "\n\n[PREVIOUS CHAT SESSIONS - The user is asking about past conversations. "
                    "Use ONLY this information to answer. Do NOT use other memories:]\n"
                    + "\n---\n".join(session_results))
            else:
                recent = storage.get_previous_sessions_content(current_session_id=current_session_id, limit=2)
                if recent:
                    metadata["sessions_found"] = len(recent)
                    summaries = [f"Session '{s['title']}':\n{s['summary']}" for s in recent]
                    context_parts.append(
                        "\n\n[RECENT CHAT SESSIONS - The user is asking about past conversations. "
                        "Use ONLY this information to answer:]\n"
                        + "\n---\n".join(summaries))
        except Exception:
            pass
    else:
        try:
            past_convos = storage.search_conversations(prompt, limit=3)
            if past_convos:
                context_parts.append("\n\n[PAST CONVERSATIONS]\n"
                                     + "\n".join(f"\u2022 {conv[:200]}" for conv in past_convos))
        except Exception:
            pass
        if not use_loops:
            try:
                memories = mnemo_client.search(prompt, limit=8)
                writing_keywords = ["write", "scene", "chapter", "story", "prose", "dialogue", "style"]
                if any(kw in prompt.lower() for kw in writing_keywords):
                    style_results = mnemo_client.search("PROSE_SAMPLE VOICE DIALOGUE_SAMPLE", limit=5)
                    memories.extend(style_results)
                enriched, enrich_meta = context_engine.build_rich_context(prompt, memories)
                if enriched:
                    context_parts.append(f"\n\n[DEEP CONTEXT]\n{enriched}")
            except Exception:
                pass

    return "".join(context_parts), metadata, metadata["skip_loops"]


# ============================================================================
# MESSAGE HANDLER (v6.0: dual-model, session_id threading, prefetching)
# ============================================================================

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

    past_conversation_context = ""
    skip_loops_for_this_query = False
    sessions_found = 0

    if needs_memory and cross_session_enabled:
        try:
            past_conversation_context, ctx_meta, skip_loops_for_this_query = build_memory_context(
                prompt, mnemo_client,
                cross_session_enabled=cross_session_enabled,
                use_loops=use_loops,
                loop_manager=st.session_state.get("loop_manager"),
                context_engine=st.session_state.get("context_engine"),
            )
            sessions_found = ctx_meta.get("sessions_found", 0)
        except Exception:
            past_conversation_context = ""

    full_system_prompt = SYSTEM_PROMPT + past_conversation_context

    # v6.0: Append the signal instruction to the system prompt if writing creatively
    if gate.query_type.value == "creative":
        full_system_prompt += "\n\n" + get_signal_instruction()

    # v6.0: Check for a pre-fetched creative brief
    cached_brief = None
    if "prefetch_engine" in st.session_state and "signal_processor" in st.session_state:
        current_state = st.session_state.signal_processor.get_state()
        cached_brief = st.session_state.prefetch_engine.check_cache(current_state)

    if cached_brief:
        # 0ms cache hit! Prepend the pre-built brief to the system prompt
        full_system_prompt = cached_brief.brief + "\n\n" + full_system_prompt

    should_use_loops = (needs_memory and use_loops and cross_session_enabled
                        and not skip_loops_for_this_query)

    # v6.0: Check session isolation toggle
    active_sessions = [current_session_id] if isolate_sessions and current_session_id else None

    # v6.0: Pass active_sessions to build_optimized_context
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
    }

    # v6.0: Call WRITING model (4o) with creative sampling params
    response, input_tokens, output_tokens, error = call_openrouter(messages, openrouter_key, mode="writing")

    if error:
        return None, error, {}

    # v6.0: Parse and process signals
    clean_response, signals = parse_signals_from_response(response)
    
    if signals and "signal_processor" in st.session_state:
        st.session_state.signal_processor.process(
            signals, mnemo_client, 
            session_id=current_session_id, 
            user_prompt=prompt
        )
        if "prefetch_engine" in st.session_state:
            # Spin up the background K2.5 thread while the user reads
            st.session_state.prefetch_engine.trigger(
                st.session_state.signal_processor.get_state(),
                st.session_state.signal_processor.history,
                mnemo_client
            )
            
    # Swap out the raw response for the clean one before saving to history
    response = clean_response

    cost_tracker = CostTracker()
    msg_cost = cost_tracker.add_usage(input_tokens, output_tokens, mode="writing")

    # Save conversation turn
    if not skip_loops_for_this_query:
        try:
            storage = get_persistent_storage(DEFAULT_HF_KEY, mnemo_client)
            if storage:
                storage.save_conversation_turn(prompt, response, current_session_id)
        except Exception:
            pass

    # Auto-extract via K2
    extracted = 0
    if (st.session_state.get("auto_extract", True)
            and cross_session_enabled
            and not skip_loops_for_this_query):
        conversation = f"User: {prompt}\n\nAssistant: {response}"
        memories, extract_cost = extract_memories_with_gpt(conversation, openrouter_key)
        if memories:
            msg_cost += extract_cost
            import concurrent.futures
            store_tasks = []
            for mem in memories:
                cat = mem.get("category", "FACT").upper()
                txt = mem.get("content", "")
                if txt:
                    meta = {"category": cat, "session_id": current_session_id}
                    store_tasks.append((f"[{cat}] {txt}", meta, cat))
            if store_tasks:
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    futures = {
                        executor.submit(mnemo_client.add, content, "default", meta, 0.5): (content, cat_name)
                        for content, meta, cat_name in store_tasks
                    }
                    for future in concurrent.futures.as_completed(futures):
                        content, cat_name = futures[future]
                        try:
                            mem_id = future.result()
                            if mem_id:
                                extracted += 1
                                st.session_state.loop_manager.add_to_loop(
                                    content=content, category=cat_name.lower(),
                                    session_id=current_session_id)
                        except Exception:
                            pass

    result_meta = {
        "cross_session_memories_used": context_meta.get("cross_session_memories_used", 0),
        "context_tokens": context_meta.get("context_tokens", 0),
        "mode": context_meta.get("mode", "full"),
        "extracted": extracted, "cost": msg_cost,
        "sessions_found": sessions_found,
        "gate_tier": context_meta.get("gate_tier", 1),
    }
    return response, None, result_meta


# ============================================================================
# SIDEBAR RENDERER (v6.0: session isolation toggle)
# ============================================================================

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

        if st.button("\u2795 New Chat", use_container_width=True, type="primary"):
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

        use_loops = st.toggle("\U0001f504 Metadata Loops (Save 80% tokens)", value=True,
                              help="Use token-efficient context injection")
        st.session_state.use_loops = use_loops

        # v6.0: Session memory isolation toggle
        isolate_sessions = st.toggle("\U0001f512 Session Memory Isolation", value=False,
                                     help="Only use memories from the current session")
        st.session_state.isolate_sessions = isolate_sessions

        if "loop_manager" in st.session_state and use_loops:
            loop_stats = st.session_state.loop_manager.get_stats()
            st.caption(f"\U0001f4ca {loop_stats['total_items']} memories | {loop_stats['total_metadata_tokens']} tokens")

        st.divider()
        st.markdown("#### 📝 Add Memory")
        render_memory_management(mnemo_client)
        st.divider()
        render_consolidation_panel(openrouter_key)

        # v6.0.1: Inline migration (Streamlit Cloud has no terminal)
        with st.expander("🔄 Migrate Blobs → Graph", expanded=False):
            stats = mnemo_client.get_stats()
            n_mem = stats.get("total_memories", 0)
            n_cp = stats.get("total_connection_points", 0)
            n_thr = stats.get("total_threads", 0)
            n_knot = stats.get("total_knots", 0)
            st.caption(f"Blobs: {n_mem} · CPs: {n_cp} · Threads: {n_thr} · Knots: {n_knot}")
            if n_cp > 0:
                st.success(f"✅ Already migrated — {n_cp} CPs exist.")
            if n_mem > 0 and st.button("🚀 Run Migration", key="run_mig", type="primary", use_container_width=True):
                from migration import run_migration
                log_lines = []
                log_box = st.empty()
                def _mig_log(msg):
                    log_lines.append(msg)
                    log_box.code("\n".join(log_lines[-15:]), language=None)
                with st.spinner("Migrating..."):
                    results = run_migration(mnemo_client, openrouter_key,
                                            phases=["decompose", "threads", "cleanup"],
                                            progress_callback=_mig_log)
                p1 = results.get("phase_results", {}).get("decompose", {})
                p2 = results.get("phase_results", {}).get("threads", {})
                cost = results.get("total_cost", 0)
                st.success(
                    f"✅ {p1.get('points_stored', 0)} CPs · "
                    f"{p2.get('threads_created', 0)} threads · "
                    f"{p2.get('knots_created', 0)} knots · ${cost:.4f}")
                errs = []
                for pd in results.get("phase_results", {}).values():
                    errs.extend(pd.get("errors", []))
                if errs:
                    with st.expander(f"⚠️ {len(errs)} errors"):
                        for e in errs[:20]:
                            st.caption(e)
                # Reload memory systems with new data
                if "loop_manager" in st.session_state:
                    st.session_state.loop_manager.load_from_mnemo(use_smart_extraction=False)
                if "signal_processor" in st.session_state:
                    st.session_state.signal_processor.update_threads_from_server(mnemo_client)

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
        if st.button("\U0001f504", key="refresh_sessions", help="Refresh from cloud"):
            try:
                storage = get_persistent_storage(hf_key, mnemo_client)
                sessions = storage.load_sessions(limit=MAX_SESSIONS_STORED)
                st.session_state.session_history = sessions
                st.rerun()
            except Exception:
                pass

    if "session_folders" not in st.session_state:
        st.session_state.session_folders = {"\U0001f4c1 Default": []}

    with st.expander("\U0001f4c2 Manage Folders", expanded=False):
        with st.popover("\u2795 Create Folder"):
            new_folder = st.text_input("New folder name", placeholder="e.g., Story Ideas")
            if st.button("Create", use_container_width=True):
                if new_folder and new_folder.strip():
                    folder_name = f"\U0001f4c1 {new_folder.strip()}"
                    if folder_name not in st.session_state.session_folders:
                        st.session_state.session_folders[folder_name] = []
                        st.success(f"Created {folder_name}")
                        st.rerun()
        folders = list(st.session_state.session_folders.keys())
        if len(folders) > 1:
            with st.popover("\U0001f5d1\ufe0f Delete Folder"):
                folder_to_delete = st.selectbox("Delete folder",
                    [""] + [f for f in folders if f != "\U0001f4c1 Default"])
                if folder_to_delete and st.button("Delete", type="primary", use_container_width=True):
                    st.session_state.session_folders["\U0001f4c1 Default"].extend(
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
            folders_with_sessions = {"\U0001f4c1 Default": []}
            for folder in st.session_state.session_folders:
                if folder not in folders_with_sessions:
                    folders_with_sessions[folder] = []
            for session in sessions:
                sid = session.get("id", "")
                folder = session_to_folder.get(sid, "\U0001f4c1 Default")
                if folder not in folders_with_sessions:
                    folder = "\U0001f4c1 Default"
                folders_with_sessions[folder].append(session)
            for folder, folder_sessions in folders_with_sessions.items():
                if folder_sessions:
                    st.caption(folder)
                    for session in folder_sessions[:10]:
                        session_id = session.get("id", "")
                        title = session.get("title", "Untitled")[:30]
                        col1, col2, col3, col4 = st.columns([6, 1, 1, 1])
                        with col1:
                            if st.button(f"\U0001f4ac {title}", key=f"load_{session_id}", use_container_width=True):
                                load_session(session_id)
                                st.rerun()
                        with col2:
                            if st.button("\u270f\ufe0f", key=f"rename_{session_id}", help="Rename"):
                                rename_session_dialog(session_id, title)
                        with col3:
                            if st.button("\U0001f4c2", key=f"move_{session_id}", help="Move to folder"):
                                move_session_dialog(session_id)
                        with col4:
                            if st.button("\U0001f5d1\ufe0f", key=f"del_{session_id}"):
                                delete_session(session_id)
                                st.rerun()
                    st.caption("")
        else:
            st.caption("No previous sessions")


def render_file_upload(mnemo_client, openrouter_key):
    st.markdown("**\U0001f4ce Upload File \u2192 Memory**")
    st.caption("Upload a file \u2014 extracts facts, deep context, style, tone via K2.5")
    uploaded_file = st.file_uploader("Upload file", type=["txt", "md", "csv", "json", "pdf", "docx"],
                                     label_visibility="collapsed")
    if uploaded_file is not None:
        if st.button("\U0001f9e0 Extract Deep Context + Memories", use_container_width=True):
            with st.spinner("Reading file..."):
                content = extract_text_from_file(uploaded_file)
            if content and not content.startswith("["):
                n_chunks = max(1, (len(content) - 1) // 12000 + 1)
                if n_chunks > 1:
                    st.info(f"\U0001f4c4 {len(content):,} chars \u2192 splitting into {min(n_chunks, 5)} chunks")
                with st.spinner(f"K2.5 extracting ({min(n_chunks, 5)} call{'s' if n_chunks > 1 else ''})..."):
                    memories, cost = extract_memories_from_file(content, uploaded_file.name, openrouter_key)
                if memories:
                    with st.spinner("Storing to memory..."):
                        current_session = st.session_state.get("current_session_id", "")
                        stored = 0
                        import concurrent.futures
                        store_tasks = []
                        for mem in memories:
                            cat = mem.get("category", "FACT").upper()
                            txt = mem.get("content", "")
                            if txt:
                                meta = {"category": cat, "session_id": current_session, "source": "file_upload"}
                                store_tasks.append((f"[{cat}] {txt}", meta, cat))
                        if store_tasks:
                            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                                futures = {
                                    executor.submit(mnemo_client.add, content, "default", meta, 1.5): (content, cat_name)
                                    for content, meta, cat_name in store_tasks
                                }
                                for future in concurrent.futures.as_completed(futures):
                                    content, cat_name = futures[future]
                                    try:
                                        mem_id = future.result()
                                        if mem_id:
                                            stored += 1
                                            st.session_state.loop_manager.add_to_loop(
                                                content=content, category=cat_name.lower(),
                                                session_id=current_session)
                                    except Exception:
                                        pass
                    facts = [m for m in memories if m.get("category") in ("CHARACTER", "PLOT", "SETTING", "THEME", "FACT")]
                    context = [m for m in memories if m.get("category") in ("CONTEXT", "CLARIFICATION", "RELATIONSHIP", "INSTRUCTION")]
                    style = [m for m in memories if m.get("category") in ("PROSE_SAMPLE", "DIALOGUE_SAMPLE", "VOICE", "VOCABULARY")]
                    tone = [m for m in memories if m.get("category") in ("TONE",)]
                    st.success(f"\u2705 Stored {stored} memories")
                    st.caption(f"\U0001f4ca {len(facts)} facts \u00b7 {len(context)} context \u00b7 {len(style)} style \u00b7 {len(tone)} tone | Cost: ${cost:.4f}")
                    with st.expander("View extracted memories", expanded=False):
                        for mem in memories:
                            st.caption(f"**[{mem.get('category', 'FACT')}]** {mem.get('content', '')[:150]}")
                else:
                    st.warning("No memories extracted. Try a different file.")
            else:
                st.error("Could not read file content")


def render_memory_management(mnemo_client):
    with st.expander("Add manually", expanded=False):
        memory_category = st.selectbox("Category",
            ["CHARACTER", "PLOT", "SETTING", "THEME", "STYLE", "TONE", "FACT"])
        memory_content = st.text_area("Content",
            placeholder="e.g., Alistair's scenes should feel clinical and cold, never melodramatic",
            height=80)
        if st.button("\U0001f4be Save", use_container_width=True):
            if memory_content.strip():
                current_session = st.session_state.get("current_session_id", "")
                meta = {"category": memory_category, "session_id": current_session}
                if mnemo_client.add(f"[{memory_category}] {memory_content}", metadata=meta):
                    st.success(f"\u2705 Saved [{memory_category}]")
                    st.session_state.loop_manager.add_to_loop(
                        memory_content, memory_category.lower(),
                        session_id=current_session)
                else:
                    st.error("Failed to save")

    with st.expander("View memories", expanded=False):
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("\U0001f504 Refresh", use_container_width=True, key="refresh_mem"):
                st.rerun()
        with col2:
            if st.button("\U0001f4d6 View All", use_container_width=True, key="view_all_mem"):
                st.session_state.show_all_memories = True
                st.rerun()
        stats = mnemo_client.get_stats()
        st.caption(f"Total: {stats.get('total_memories', 0)} | CPs: {stats.get('total_connection_points', 0)} | Links: {stats.get('total_links', 0)}")
        memories = mnemo_client.list_memories()[:15]
        for mem in memories:
            col1, col2 = st.columns([5, 1])
            with col1:
                st.caption(f"{mem.get('content', '')[:60]}...")
            with col2:
                if st.button("\U0001f5d1\ufe0f", key=f"del_mem_{mem.get('id', '')}"):
                    if mnemo_client.delete(mem.get("id")):
                        st.rerun()
        if len(mnemo_client.list_memories()) >= 15:
            st.caption("... showing first 15. Click 'View All' for complete list")
        st.markdown("---")
        if st.button("\U0001f9f9 Clear ALL Memories", use_container_width=True):
            st.session_state.confirm_clear = True
        if st.session_state.get("confirm_clear"):
            st.warning("\u26a0\ufe0f Delete ALL memories?")
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
        st.subheader("\U0001f4d6 All Memories")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("\u274c Close", use_container_width=True, key="close_all_mem"):
                st.session_state.show_all_memories = False
                st.rerun()
        with col2:
            search_query = st.text_input("\U0001f50d Search", key="mem_search", placeholder="Filter memories...")
        with col3:
            category_filter = st.selectbox("Category",
                ["All", "CHARACTER", "PLOT", "SETTING", "THEME", "CONTEXT", "STYLE", "TONE", "FACT"],
                key="cat_filter")
        all_memories = mnemo_client.list_memories()
        if search_query:
            all_memories = [m for m in all_memories if search_query.lower() in m.get("content", "").lower()]
        if category_filter != "All":
            all_memories = [m for m in all_memories if f"[{category_filter}]" in m.get("content", "")]
        st.caption(f"Showing {len(all_memories)} memories")
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
                if st.button("\U0001f5d1\ufe0f", key=f"del_full_{mem_id}"):
                    if mnemo_client.delete(mem_id):
                        st.rerun()


def render_consolidation_panel(openrouter_key):
    with st.expander("🧠 Memory Consolidation", expanded=False):
        st.caption("Generates **new** CONTEXT, RELATIONSHIP, and TONE entries from your existing facts. Never removes existing memories — purely additive.")
        last_consol = st.session_state.get("last_consolidation")
        if last_consol:
            st.caption(f"Last run: {last_consol[:16]}")

        # Show what will be analyzed
        try:
            mnemo_client = st.session_state.get("mnemo_client")
            if mnemo_client:
                all_mems = mnemo_client.list_memories()
                facts = [m for m in all_mems if not any(
                    m.get("content", "").startswith(f"[{tag}]")
                    for tag in ("CONTEXT", "RELATIONSHIP", "CLARIFICATION", "TIMELINE", "CONVERSATION", "SESSION")
                )]
                existing_context = [m for m in all_mems if any(
                    m.get("content", "").startswith(f"[{tag}]")
                    for tag in ("CONTEXT", "RELATIONSHIP", "CLARIFICATION", "TIMELINE")
                )]
                st.caption(f"📊 {len(facts)} facts to analyze · {len(existing_context)} context entries already exist")
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


# ============================================================================
# CHAT RENDERER
# ============================================================================

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
                        if meta.get("cross_session_memories_used", 0) > 0:
                            memory_info.append(f"\U0001f4da {meta['cross_session_memories_used']} memories")
                        mode = meta.get("mode", "")
                        if mode == "t1_direct":
                            memory_info.append(f"\U0001f504 {meta.get('context_tokens', 0)} tokens")
                        elif mode == "t2_confirmed":
                            memory_info.append(f"\U0001f504 t2 | {meta.get('context_tokens', 0)} tokens")
                        elif mode == "skip":
                            memory_info.append("\u26a1 fast")
                        if meta.get("extracted", 0) > 0:
                            memory_info.append(f"\U0001f9e0 {meta['extracted']} extracted")
                        if meta.get("cost"):
                            memory_info.append(f"\U0001f4b0 ${meta['cost']:.4f}")
                        if memory_info:
                            st.caption(" | ".join(memory_info))
                with col2:
                    if st.button("\U0001f4cb", key=f"copy_{idx}", help="Copy response"):
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
                    meta_parts.append(f"\U0001f4dc {result_meta['sessions_found']} past chats")
                if result_meta.get("cross_session_memories_used", 0) > 0:
                    meta_parts.append(f"\U0001f4da {result_meta['cross_session_memories_used']} memories")
                mode = result_meta.get("mode", "")
                if mode == "t1_direct":
                    meta_parts.append(f"\U0001f504 {result_meta.get('context_tokens', 0)} tokens")
                elif mode == "t2_confirmed":
                    meta_parts.append(f"\U0001f504 t2 | {result_meta.get('context_tokens', 0)} tokens")
                elif mode == "skip":
                    meta_parts.append("\u26a1 fast")
                if result_meta.get("extracted", 0) > 0:
                    meta_parts.append(f"\U0001f9e0 {result_meta['extracted']} extracted")
                meta_parts.append(f"\U0001f4b0 ${result_meta.get('cost', 0):.4f}")
                st.caption(" | ".join(meta_parts))
                st.session_state.messages.append({
                    "role": "assistant", "content": response, "metadata": result_meta,
                })
                save_current_session()


# ============================================================================
# MAIN APP (v6.0)
# ============================================================================

def main():
    st.set_page_config(page_title="4o with Memory", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

    # ── Modern AI App Theme ──────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Global ── */
    .stApp {
        font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .stApp > header { background: transparent !important; }

    /* ── Hide default Streamlit chrome ── */
    #MainMenu, footer, .stDeployButton { display: none !important; }

    /* ── Chat messages ── */
    .stChatMessage {
        border-radius: 16px !important;
        padding: 1rem 1.25rem !important;
        margin-bottom: 0.5rem !important;
        border: 1px solid rgba(128, 128, 128, 0.08) !important;
        backdrop-filter: blur(8px);
        font-size: 0.95rem;
        line-height: 1.7;
        letter-spacing: 0.01em;
    }
    /* User messages */
    .stChatMessage[data-testid="chat-message-user"] {
        background: rgba(99, 102, 241, 0.06) !important;
        border-left: 3px solid rgba(99, 102, 241, 0.4) !important;
    }
    /* Assistant messages */
    .stChatMessage[data-testid="chat-message-assistant"] {
        background: rgba(255, 255, 255, 0.02) !important;
    }

    /* ── Chat input ── */
    .stChatInput > div {
        border-radius: 24px !important;
        border: 1px solid rgba(128, 128, 128, 0.15) !important;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    .stChatInput > div:focus-within {
        border-color: rgba(99, 102, 241, 0.5) !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.08) !important;
    }
    .stChatInput textarea {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.95rem !important;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg,
            rgba(15, 15, 20, 0.97) 0%,
            rgba(20, 20, 30, 0.95) 100%) !important;
        border-right: 1px solid rgba(128, 128, 128, 0.08) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(128, 128, 128, 0.1) !important;
        margin: 0.75rem 0 !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        border-radius: 12px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.02em;
        transition: all 0.15s ease;
        border: 1px solid rgba(128, 128, 128, 0.12) !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1 0%, #818cf8 100%) !important;
        border: none !important;
        color: white !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #5558e6 0%, #7578f0 100%) !important;
    }

    /* ── Toggle/checkbox ── */
    .stToggle label span {
        font-size: 0.85rem !important;
    }

    /* ── Expanders ── */
    .streamlit-expanderHeader {
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        border-radius: 10px !important;
    }

    /* ── Captions (metadata) ── */
    .stChatMessage .stCaption, .stChatMessage caption {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.7rem !important;
        opacity: 0.5;
        letter-spacing: 0.03em;
        transition: opacity 0.2s;
    }
    .stChatMessage:hover .stCaption, .stChatMessage:hover caption {
        opacity: 0.8;
    }

    /* ── Spinner ── */
    .stSpinner > div {
        border-radius: 12px !important;
    }

    /* ── Code blocks in chat ── */
    .stChatMessage pre {
        border-radius: 10px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
    }

    /* ── Tabs (sidebar) ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        font-size: 0.8rem;
        font-family: 'DM Sans', sans-serif;
    }

    /* ── Toast/alerts ── */
    .stAlert {
        border-radius: 12px !important;
    }

    /* ── Brand header ── */
    .brand-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.25rem 0 0.5rem;
    }
    .brand-header .brand-icon {
        font-size: 1.8rem;
        line-height: 1;
    }
    .brand-header .brand-text h1 {
        font-family: 'DM Sans', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        letter-spacing: -0.03em;
        margin: 0;
        padding: 0;
        line-height: 1.2;
    }
    .brand-header .brand-text p {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.75rem;
        opacity: 0.45;
        margin: 0.15rem 0 0;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        font-weight: 500;
    }

    /* ── Status pill (bottom bar) ── */
    .status-bar {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        opacity: 0.35;
        letter-spacing: 0.05em;
        text-align: center;
        padding: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Brand Header ──
    st.markdown("""
    <div class="brand-header">
        <div class="brand-icon">🧠</div>
        <div class="brand-text">
            <h1>4o with Memory</h1>
            <p>GPT-4o writer · K2 memory curator · Mnemo v6.0</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not DEFAULT_OPENROUTER_KEY or not DEFAULT_HF_KEY:
        st.error("\u26a0\ufe0f **API Keys Not Configured!**")
        st.markdown("""
        **For Streamlit Cloud:**
        1. Go to your app's Settings \u2192 Secrets
        2. Add your keys in TOML format:
        ```toml
        OPENROUTER_KEY = "sk-or-v1-your-key"
        HF_KEY = "hf_your-token"
        ```
        """)
        st.stop()

    mnemo_client = init_client(DEFAULT_HF_KEY)

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
        
    # v6.0 NEW INITIALIZATIONS
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
        4o with Memory v6.0 &nbsp;·&nbsp; GPT-4o + K2 + Mnemo &nbsp;·&nbsp; Threads & Knots &nbsp;·&nbsp; Graph Search
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

"""
4o with Memory v7.1.1 — Dual-Processor Creative Writing Engine

v7.1.1 changes:
- FIX: Memory Editor now correctly uses the update_point API endpoint
  instead of the legacy Delete+Re-Insert hack.
- NEW: Integrated Memory Editor directly into app.py. No external dependencies.
- OPTIMIZED: Removed Thread/Knot synthesis from real-time chat extraction. 
  Chat extraction is now purely for atomic facts (faster, cheaper), leaving 
  structural narrative generation to the asynchronous Consolidation pass.
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

from mnemo_client import MnemoClient, LocalMnemoClient, ResilientMnemoClient, create_mnemo_client
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
    "max_tokens": 16384,
}

MEMORY_PARAMS = {
    "temperature": 0.2,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "max_tokens": 4096,
}

MAX_CONVERSATION_MESSAGES = 8
MAX_SESSIONS_STORED = 20

CATEGORY_TO_CP = {
    "CHARACTER": ("character", "character_profile"),
    "MOTIVATION": ("character", "motivation"),
    "PLOT": ("plot", "plot_event"),
    "SETTING": ("setting", "setting_detail"),
    "TIMELINE": ("plot", "timeline"),
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
                source="auto_extract", connects_to="", weight=0.5,
                thread_id="", position=-1, namespace="default"):
    cat_key = category.upper()
    cp_category, point_type = CATEGORY_TO_CP.get(cat_key, ("fact", "general"))
    normalized_entity = normalize_entity(entity)
    normalized_connects = normalize_entity(connects_to) if connects_to else ""
    return mnemo_client.add_point(
        entity=normalized_entity, point_type=point_type, value=content,
        connects_to=normalized_connects, reason="", weight=float(weight),
        category=cp_category, session_id=session_id, source=source,
        thread_id=thread_id, position=int(position), namespace=namespace,
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
        st.session_state.mnemo_client = create_mnemo_client(
            mode="auto", token=hf_key,
            db_path=os.path.join(os.path.expanduser("~"), ".mnemo", "mnemo.db"),
            base_url=MNEMO_URL,
        )
    return st.session_state.mnemo_client

def get_persistent_storage(hf_key=None, client=None):
    key = hf_key or DEFAULT_HF_KEY
    cli = client or st.session_state.get("mnemo_client")
    if "persistent_storage" not in st.session_state or st.session_state.get("_ps_key") != key:
        if key and cli:
            st.session_state.persistent_storage = SessionStore(hf_key=key, mnemo_client=cli)
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
        except Exception as e:
            print(f"[WARN] session save: {type(e).__name__}: {e}")
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
    except Exception as e:
        print(f"[WARN] session store delete: {type(e).__name__}: {e}")
    try:
        mnemo_client = st.session_state.get("mnemo_client")
        if mnemo_client:
            result = mnemo_client.delete_session(session_id)
            print(f"[INFO] Mnemo cascade delete for {session_id}: {result}")
    except Exception as e:
        print(f"[WARN] Mnemo cascade delete failed: {type(e).__name__}: {e}")
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
# FILE PROCESSING & K2 AUTO-EXTRACTION 
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

def extract_json_from_response(raw: str) -> dict:
    clean = raw.strip()
    try:
        parsed = json.loads(clean)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    
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
    
    if '"memories"' in clean or '"local_id"' in clean:
        salvaged = _salvage_truncated_memories(clean)
        if salvaged:
            n = len(salvaged.get("memories", []))
            print(f"[K2 EXTRACT] JSON truncated — salvaged {n} complete memories from partial response")
            return salvaged
    return {}

def _salvage_truncated_memories(raw: str) -> dict:
    memory_pattern = re.compile(
        r'\{[^{}]*?"(?:local_id|entity)"[^{}]*?"content"\s*:\s*"(?:\\.|[^"\\])*"[^{}]*?\}',
        re.DOTALL
    )
    matches = memory_pattern.findall(raw)
    if not matches:
        return {}
    memories = []
    for match in matches:
        try:
            obj = json.loads(match)
            if isinstance(obj, dict) and obj.get("content"):
                memories.append(obj)
        except json.JSONDecodeError:
            continue
    if not memories:
        return {}
    threads = []
    thread_pattern = re.compile(
        r'\{[^{}]*?"name"\s*:\s*"[^"]*"[^{}]*?"memory_local_ids"\s*:\s*\[[^\]]*\][^{}]*?\}',
        re.DOTALL
    )
    for match in thread_pattern.findall(raw):
        try:
            obj = json.loads(match)
            if isinstance(obj, dict) and obj.get("name"):
                threads.append(obj)
        except json.JSONDecodeError:
            continue
    return {"memories": memories, "threads": threads, "knots": []}

INPUT_CHUNK_CHARS = 25000 
INPUT_CHUNK_OVERLAP = 2000 
MAX_INPUT_CHUNKS = 10

def _build_extraction_prompt(content: str, filename: str, chunk_label: str = "") -> str:
    return f"""You are a memory extraction engine for a creative writing system. Your output will be stored as ConnectionPoints (CPs) in a graph database.

DOCUMENT: {filename}{chunk_label}
CONTENT:
{content}

EXTRACTION METHOD — Go through the text step by step, one W-question at a time, IN THIS ORDER. Do not skip any step. After completing each step, move to the next.

STEP 1 — WHO (category: CHARACTER)
Read through the text and extract every character mentioned. For each, capture: name, age, role, appearance, personality traits, speech patterns, backstory, reputation. One CP per distinct character facet.

STEP 2 — WHY (category: MOTIVATION)  
Read through again. For each character or faction, extract: drives, fears, goals, secrets, philosophy, internal conflicts, desires, obsessions. One CP per distinct motivation.

STEP 3 — WHOM/WHOSE (category: RELATIONSHIP)
Read through again. Extract every connection between characters: power dynamics, emotional bonds, betrayals, alliances, family ties, romantic tension, mentorship, rivalries, loyalty. One CP per character pair per dynamic. MUST include "connects_to" field.

STEP 4 — WHICH (category: PLOT)
Read through again. Extract every narrative event: beats, conflicts, choices, turning points, twists, reveals, cause-and-effect chains, consequences. One CP per narrative beat.

STEP 5 — WHEN (category: TIMELINE)
Read through again. Extract chronological information: sequence of events, before/after relationships, time markers, pacing, deadlines, age references, era details. One CP per distinct temporal fact.

STEP 6 — WHERE (category: SETTING)
Read through again. Extract every location and environment: geography, atmosphere, sensory details, architecture, institutions, cultural spaces. One CP per distinct place.

STEP 7 — HOW (category: TONE)
Read through again. Extract writing directives: voice, prose style, atmosphere, recurring motifs, do-nots, dialogue rules, pacing notes, methods, rituals, techniques. One CP per broad directive.

STEP 8 — WHAT (category: FACT)
Read through once more. Extract any remaining world-building facts, faction structures, systems, lore, objects of significance, rules, or context that wasn't captured above. One CP per distinct fact.

STEP 9 — CLARIFICATION
Extract any author meta-instructions, corrections, or decisions about how to handle the material.

W-WORD → CATEGORY MAPPING (use these exact category values in your output):
  WHO → CHARACTER | WHY → MOTIVATION | WHOM/WHOSE → RELATIONSHIP
  WHICH → PLOT | WHEN → TIMELINE | WHERE → SETTING
  HOW → TONE | WHAT → FACT | meta → CLARIFICATION

CP FORMAT RULES:
- Each CP must have: local_id, entity, category, content
- "entity" is a plain name used as a lookup key: "Sebastian Carlisle" not "Dr. Sebastian Carlisle"
- "content" is 1-3 dense sentences. Combine related info — don't micro-atomize.
- For RELATIONSHIP CPs: MUST include "connects_to" field with the other person's name.
- Do NOT repeat the same fact across categories. If a fact was captured in CHARACTER, don't repeat it in MOTIVATION.
- Extract what's ACTUALLY in the text. Don't invent beyond what's written.

Respond with ONLY a valid JSON object — no preamble, no markdown fences:
{{
  "memories": [
    {{"local_id": "m1", "entity": "Alistair Fitzroy", "category": "CHARACTER", "content": "Late 30s professor of pharmacology at Cambridge. Cold grey eyes, impeccably dressed, keeps hands deliberately still to project control. Wears his late mother's signet ring."}},
    {{"local_id": "m2", "entity": "Alistair Fitzroy", "category": "MOTIVATION", "content": "Terrified of dementia after watching his father's decline. Would rather be remembered as a monster than not at all. Addicted to the feeling of discovery."}},
    {{"local_id": "m3", "entity": "Alistair Fitzroy", "category": "RELATIONSHIP", "connects_to": "Sebastian Carlisle", "content": "Former friends at Edinburgh before a betrayal. Now oversees Sebastian's captivity drug regimen, viewing him as both greatest experiment and greatest failure."}},
    {{"local_id": "m4", "entity": "Sebastian Carlisle", "category": "PLOT", "content": "Book 3: Blood samples tampered with opium causing mental deterioration, leading to surgical mistake and license revocation. Alistair offers Blackwood Estate as alternative to institutionalization."}},
    {{"local_id": "m5", "entity": "Sebastian Carlisle", "category": "TIMELINE", "content": "Captivity arc occurs in Book 3, after the laudanum moral panic in Book 2. Sebastian's license revocation precedes the Blackwood Estate transfer."}},
    {{"local_id": "m6", "entity": "Blackwood Estate", "category": "SETTING", "content": "Remote estate where Sebastian is held captive. Isabella's territory — Alistair's control unravels when she takes over. Isolation reinforces captivity themes."}},
    {{"local_id": "m7", "entity": "Alistair Fitzroy", "category": "TONE", "content": "Scenes should feel clinical and cold, never melodramatic. Power through precision, not theatrics. Dialogue always the shortest in the room — he asks questions, never monologues."}},
    {{"local_id": "m8", "entity": "Fitzroy Infant Protocol", "category": "FACT", "content": "Pharmacovigilance innovation created by Alistair, later weaponized by Red Rose front organizations to distribute laudanum-laced tonics through charity fronts."}},
    {{"local_id": "m9", "entity": "Story", "category": "CLARIFICATION", "content": "Never use Dr. prefix in narration — just Alistair or Fitzroy. His hands are a recurring motif: stillness equals control, trembling equals losing it."}}
  ]
}}

The 9 examples above correspond 1:1 to the 9 steps:
  m1=WHO, m2=WHY, m3=WHOM, m4=WHICH, m5=WHEN, m6=WHERE, m7=HOW, m8=WHAT, m9=CLARIFICATION"""

async def _extract_chunk_async(http_client, chunk, filename, chunk_idx, total_chunks, openrouter_key):
    chunk_label = f" (part {chunk_idx+1}/{total_chunks})" if total_chunks > 1 else ""
    prompt = _build_extraction_prompt(chunk, filename, chunk_label)

    try:
        response = await http_client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {openrouter_key}", "Content-Type": "application/json"},
            json={"model": MEMORY_MODEL_ID, "messages": [{"role": "user", "content": prompt}], **MEMORY_PARAMS},
            timeout=120.0
        )
        if response.status_code != 200:
            print(f"[K2 EXTRACT chunk {chunk_idx+1}] API error {response.status_code}: {response.text[:200]}")
            return [], 0

        data = response.json()
        raw = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        cost = (usage.get("prompt_tokens", 0) * 0.55 + usage.get("completion_tokens", 0) * 2.20) / 1_000_000

        parsed = extract_json_from_response(raw)
        if not parsed:
            print(f"[K2 EXTRACT chunk {chunk_idx+1}] JSON parse failed. Preview: {raw[:300]}")
            return [], cost

        memories = parsed.get("memories", [])
        if not memories:
            for alt in ("results", "items", "data", "points"):
                memories = parsed.get(alt, [])
                if memories:
                    break

        for mem in memories:
            if not mem.get("local_id"):
                mem["local_id"] = f"m_{uuid.uuid4().hex[:6]}"
            if not mem.get("category"):
                mem["category"] = "FACT"

        print(f"[K2 EXTRACT chunk {chunk_idx+1}] Extracted {len(memories)} memories")
        return memories, cost

    except Exception as e:
        print(f"[K2 EXTRACT chunk {chunk_idx+1}] Exception: {type(e).__name__}: {e}")
        traceback.print_exc()
        return [], 0

async def _extract_threads_async(http_client, memories_summary, openrouter_key):
    prompt = f"""You are given a list of extracted story memories. Identify:
1. THREADS: Narrative sequences where 3+ memories form a chronological or causal chain.
2. KNOTS: Points where two or more threads collide or intersect.

EXTRACTED MEMORIES:
{memories_summary}

Respond with ONLY a valid JSON object:
{{
  "threads": [
    {{"name": "Captivity Arc", "entity": "Sebastian", "type": "plot_line", "memory_local_ids": ["m1", "m2", "m5"]}}
  ],
  "knots": [
    {{"name": "Blackwood Arrival", "thread_names": ["Captivity Arc", "Isabella's Obsession"], "reason": "Sebastian enters Isabella's territory"}}
  ]
}}

RULES:
- Only create threads from 3+ memories that form a clear sequence.
- Only create knots where threads genuinely intersect.
- If no clear threads or knots exist, return empty arrays.
- Use the exact local_id values from the memories above."""

    try:
        response = await http_client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {openrouter_key}", "Content-Type": "application/json"},
            json={"model": MEMORY_MODEL_ID, "messages": [{"role": "user", "content": prompt}], **MEMORY_PARAMS},
            timeout=120.0
        )
        if response.status_code != 200:
            print(f"[K2 THREADS] API error {response.status_code}: {response.text[:200]}")
            return [], [], 0

        data = response.json()
        raw = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        cost = (usage.get("prompt_tokens", 0) * 0.55 + usage.get("completion_tokens", 0) * 2.20) / 1_000_000

        parsed = extract_json_from_response(raw)
        if not parsed:
            print(f"[K2 THREADS] JSON parse failed. Preview: {raw[:300]}")
            return [], [], cost

        threads = parsed.get("threads", [])
        knots = parsed.get("knots", [])
        print(f"[K2 THREADS] Found {len(threads)} threads, {len(knots)} knots")
        return threads, knots, cost

    except Exception as e:
        print(f"[K2 THREADS] Exception: {type(e).__name__}: {e}")
        return [], [], 0

def calculate_extraction_plan(content: str) -> dict:
    words = len(content.split())
    chars = len(content)

    if chars <= INPUT_CHUNK_CHARS:
        n_chunks = 1
    else:
        n_chunks = min(max(1, -(-chars // INPUT_CHUNK_CHARS)), MAX_INPUT_CHUNKS)

    total_calls = n_chunks + 1  

    plan = {
        "n_chunks": n_chunks,
        "total_calls": total_calls,
        "words": words,
        "chars": chars,
    }
    print(f"[EXTRACT PLAN] {words:,} words -> {n_chunks} chunk(s) (W-method guided) "
          f"+ 1 thread pass = {total_calls} calls")
    return plan

def extract_memories_from_file(content, filename, openrouter_key):
    plan = calculate_extraction_plan(content)

    if len(content) <= INPUT_CHUNK_CHARS:
        chunks = [content]
    else:
        chunks = []
        start = 0
        while start < len(content) and len(chunks) < plan["n_chunks"]:
            end = start + INPUT_CHUNK_CHARS
            if end < len(content):
                boundary = content.rfind('\n', start, end)
                if boundary == -1 or boundary < end - 1000:
                    boundary = content.rfind('. ', start, end)
                if boundary != -1 and boundary > start + (INPUT_CHUNK_CHARS // 2):
                    end = boundary + 1
            chunks.append(content[start:end])
            start = end - INPUT_CHUNK_OVERLAP

    async def run_extraction():
        semaphore = asyncio.Semaphore(3)
        async def sem_extract(http_client, chunk, filename, i, total, key):
            async with semaphore:
                return await _extract_chunk_async(http_client, chunk, filename, i, total, key)

        async with httpx.AsyncClient() as http_client:
            tasks = [
                sem_extract(http_client, chunk, filename, i, len(chunks), openrouter_key)
                for i, chunk in enumerate(chunks)
            ]
            chunk_results = await asyncio.gather(*tasks)

            all_memories = []
            total_cost = 0
            for memories, cost in chunk_results:
                all_memories.extend(memories)
                total_cost += cost

            seen = set()
            unique_memories = []
            for mem in all_memories:
                key = mem.get("content", "")[:80].lower().strip()
                if key not in seen:
                    seen.add(key)
                    unique_memories.append(mem)

            threads, knots = [], []
            if len(unique_memories) >= 5:
                summary_lines = []
                for mem in unique_memories:
                    lid = mem.get("local_id", "?")
                    ent = mem.get("entity", "?")
                    cat = mem.get("category", "?")
                    txt = mem.get("content", "")[:80]
                    conn = f" -> {mem['connects_to']}" if mem.get("connects_to") else ""
                    summary_lines.append(f"[{lid}] {ent}{conn} ({cat}): {txt}")

                summary_text = "\n".join(summary_lines)
                threads, knots, thread_cost = await _extract_threads_async(
                    http_client, summary_text, openrouter_key
                )
                total_cost += thread_cost

            return unique_memories, threads, knots, total_cost

    try:
        loop = asyncio.get_running_loop()
        import nest_asyncio
        nest_asyncio.apply()
        memories, threads, knots, cost = loop.run_until_complete(run_extraction())
    except RuntimeError:
        memories, threads, knots, cost = asyncio.run(run_extraction())

    return {"memories": memories, "threads": threads, "knots": knots}, cost, plan


# v7.1.1 FIX: Pruned conversational prompt (no Threads/Knots logic)
def extract_memories_with_gpt(conversation, openrouter_key):
    prompt = """You are a memory extraction engine for a creative writing system. Your output will be stored as ConnectionPoints (CPs) in a graph database.

CONVERSATION:
{conversation}

EXTRACTION METHOD — Go through the conversation step by step, one W-question at a time, IN THIS ORDER:
STEP 1 — WHO (CHARACTER): Extract every character mentioned — identity, traits, appearance, backstory.
STEP 2 — WHY (MOTIVATION): For each character, extract drives, fears, goals, secrets, philosophy.
STEP 3 — WHOM/WHOSE (RELATIONSHIP): Extract connections between characters. MUST include "connects_to" field.
STEP 4 — WHICH (PLOT): Extract narrative events, beats, conflicts, twists, consequences.
STEP 5 — WHEN (TIMELINE): Extract chronological info, sequence, before/after, time markers.
STEP 6 — WHERE (SETTING): Extract locations, atmosphere, sensory details, institutions.
STEP 7 — HOW (TONE): Extract writing directives, voice, style, motifs, do-nots.
STEP 8 — WHAT (FACT): Extract remaining world-building, factions, systems, lore.
STEP 9 — CLARIFICATION: Extract author corrections, instructions, meta-decisions.

W-WORD → CATEGORY MAPPING:
  WHO → CHARACTER | WHY → MOTIVATION | WHOM/WHOSE → RELATIONSHIP
  WHICH → PLOT | WHEN → TIMELINE | WHERE → SETTING
  HOW → TONE | WHAT → FACT | meta → CLARIFICATION

CP FORMAT RULES:
- Each CP: local_id, entity, category, content (1-3 dense sentences)
- Plain names as entities. Don't repeat facts across categories.
- RELATIONSHIP CPs MUST include "connects_to" field.

Respond with ONLY a valid JSON object — no preamble, no markdown fences:
{{
  "memories": [
    {{"local_id": "m1", "entity": "Alistair Fitzroy", "category": "CHARACTER", "content": "Late 30s professor of pharmacology at Cambridge, impeccably dressed with cold grey eyes."}},
    {{"local_id": "m2", "entity": "Alistair Fitzroy", "category": "RELATIONSHIP", "connects_to": "Sebastian Carlisle", "content": "Oversees Sebastian's drug regimen; views him as both experiment and failure."}}
  ]
}}"""

    try:
        response = requests.post(
            ""https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {openrouter_key}", "Content-Type": "application/json"},
            json={"model": MEMORY_MODEL_ID, "messages": [{"role": "user", "content": prompt.format(conversation=conversation)}], **MEMORY_PARAMS},
            timeout=60
        )
        if response.status_code != 200:
            print(f"[K2 CONV EXTRACT] API error {response.status_code}: {response.text[:300]}")
            return {"memories": []}, 0
            
        data = response.json()
        raw = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        cost = (usage.get("prompt_tokens", 0) * 0.55 + usage.get("completion_tokens", 0) * 2.20) / 1_000_000
        
        parsed = extract_json_from_response(raw)
        
        if not parsed:
            print(f"[K2 CONV EXTRACT] JSON parse failed. Raw preview: {raw[:500]}")
            return {"memories": []}, cost
        
        if "memories" in parsed:
            return parsed, cost
        
        for alt_key in ("results", "items", "data", "points", "extractions"):
            if alt_key in parsed:
                return {"memories": parsed[alt_key]}, cost
        
        print(f"[K2 CONV EXTRACT] No 'memories' key. Keys: {list(parsed.keys())}")
        return {"memories": []}, cost
        
    except Exception as e:
        print(f"[K2 CONV EXTRACT] Exception: {type(e).__name__}: {e}")
        traceback.print_exc()
        return {"memories": []}, 0

def call_openrouter(messages, api_key, mode="writing"):
    model = WRITING_MODEL_ID if mode == "writing" else MEMORY_MODEL_ID
    params = WRITING_PARAMS if mode == "writing" else MEMORY_PARAMS
    try:
        body = {"model": model, "messages": messages, **params}
        response = requests.post(
            ""https://openrouter.ai/api/v1/chat/completions",
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
    ctx_meta = {"sessions_found": 0, "threads_injected": 0, "knots_injected": 0, "cp_results": 0}

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
            print(f"[WARN] build_memory_context failed: {type(e).__name__}: {e}")
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
        "threads_injected": ctx_meta.get("threads_injected", 0),
        "knots_injected": ctx_meta.get("knots_injected", 0),
        "cp_results": ctx_meta.get("cp_results", 0),
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
        except Exception as e:
            print(f"[WARN] conversation turn save: {type(e).__name__}: {e}")

    extracted = 0
    if (st.session_state.get("auto_extract", True)
            and cross_session_enabled
            and not skip_loops_for_this_query):
        conversation = f"User: {prompt}\n\nAssistant: {response}"
        extraction_data, extract_cost = extract_memories_with_gpt(conversation, openrouter_key)
        memories = extraction_data.get("memories", [])
        
        if memories:
            msg_cost += extract_cost
            
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
                    except Exception as e:
                        print(f"[WARN] extraction storage: {type(e).__name__}: {e}")

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


# ============================================================================
# MEMORY EDITOR COMPONENT
# ============================================================================

EDITOR_CATEGORIES = ["character", "motivation", "plot", "setting", "timeline", "theme", "tone", "style", "fact", "clarification", "relationship"]

def render_memory_management(mnemo_client, key_prefix: str = "memedit"):
    st.markdown("### 🧠 Memory Editor")

    col1, col2 = st.columns([3, 1])
    with col1:
        search = st.text_input("Search memories", placeholder="Type to filter...",
                                key=f"{key_prefix}_search", label_visibility="collapsed")
    with col2:
        cat_filter = st.selectbox("Category", ["All"] + EDITOR_CATEGORIES, key=f"{key_prefix}_cat_filter",
                                   label_visibility="collapsed")

    try:
        all_points = mnemo_client.list_points(limit=500)
    except Exception as e:
        st.error(f"Failed to load memories: {e}")
        return

    if not all_points:
        st.info("No memories stored yet.")
        _render_add_new(mnemo_client, key_prefix)
        return

    filtered = all_points
    if search:
        search_lower = search.lower()
        filtered = [cp for cp in filtered if (
            search_lower in cp.get("entity", "").lower() or
            search_lower in cp.get("value", "").lower() or
            search_lower in cp.get("connects_to", "").lower() or
            search_lower in cp.get("reason", "").lower() or
            search_lower in cp.get("id", "").lower()
        )]
    if cat_filter != "All":
        filtered = [cp for cp in filtered if cp.get("category", "").lower() == cat_filter.lower()]

    st.caption(f"Showing {len(filtered)} of {len(all_points)} memories")

    for i, cp in enumerate(filtered):
        _render_cp_card(mnemo_client, cp, i, key_prefix)

    st.divider()
    _render_add_new(mnemo_client, key_prefix)

    st.markdown("---")
    if st.button("🧹 Clear ALL Memories", use_container_width=True):
        st.session_state.confirm_clear = True
    if st.session_state.get("confirm_clear"):
        st.warning("⚠️ Delete ALL memories?")
        ccol1, ccol2 = st.columns(2)
        with ccol1:
            if st.button("Yes, delete all"):
                mnemo_client.clear()
                st.session_state.confirm_clear = False
                st.rerun()
        with ccol2:
            if st.button("Cancel"):
                st.session_state.confirm_clear = False
                st.rerun()

def _render_cp_card(mnemo_client, cp: dict, index: int, key_prefix: str):
    cp_id = cp.get("id", "")
    entity = cp.get("entity", "")
    value = cp.get("value", "")
    category = cp.get("category", "fact").lower()
    connects_to = cp.get("connects_to", "")

    preview = f"**{entity}**"
    if connects_to:
        preview += f" → {connects_to}"
    preview += f": {value[:60]}{'...' if len(value) > 60 else ''}"

    cat_colors = {
        "character": "🟣", "motivation": "🟣", "plot": "🔴", "timeline": "🔴",
        "setting": "🟢", "theme": "🔵", "tone": "🟡", "style": "🟠",
        "fact": "⚪", "clarification": "⚪", "relationship": "🟤",
    }
    cat_icon = cat_colors.get(category, "⚪")

    with st.expander(f"{cat_icon} {preview}", expanded=False):
        _render_edit_form(mnemo_client, cp, index, key_prefix)

def _render_edit_form(mnemo_client, cp: dict, index: int, key_prefix: str):
    cp_id = cp.get("id", "")
    k = f"{key_prefix}_{index}"
    cat = cp.get("category", "fact").lower()
    
    if cat not in EDITOR_CATEGORIES:
        cat = "fact"

    col1, col2 = st.columns(2)
    with col1:
        new_entity = st.text_input("Entity", value=cp.get("entity", ""), key=f"{k}_entity")
        new_category = st.selectbox("Category", EDITOR_CATEGORIES, index=EDITOR_CATEGORIES.index(cat), key=f"{k}_cat")
    with col2:
        new_connects_to = st.text_input("Connects to", value=cp.get("connects_to", ""), key=f"{k}_conn")
        new_weight = st.slider("Weight", 0.0, 1.0, float(cp.get("weight", 0.5)), 0.05, key=f"{k}_weight")

    new_value = st.text_area("Value", value=cp.get("value", ""), height=100, key=f"{k}_value")
    new_reason = st.text_input("Reason", value=cp.get("reason", ""), key=f"{k}_reason")

    st.caption(f"ID: `{cp_id}` · Source: {cp.get('source', '')} · Type: {cp.get('point_type', '')}")

    bcol1, bcol2, bcol3 = st.columns([1, 1, 2])
    with bcol1:
        if st.button("💾 Save", key=f"{k}_save", type="primary"):
            _save_changes(mnemo_client, cp, new_entity, new_value, new_connects_to,
                          new_reason, new_weight, new_category, k)
    with bcol2:
        if st.button("🗑️ Delete", key=f"{k}_del"):
            st.session_state[f"{k}_confirm_delete"] = True

    if st.session_state.get(f"{k}_confirm_delete"):
        st.warning("Are you sure?")
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            if st.button("Yes, delete", key=f"{k}_del_yes", type="primary"):
                try:
                    mnemo_client.delete_point(cp_id)
                    st.success("Deleted!")
                    st.session_state.pop(f"{k}_confirm_delete", None)
                    st.rerun()
                except Exception as e:
                    st.error(f"Delete failed: {e}")
        with dcol2:
            if st.button("Cancel", key=f"{k}_del_no"):
                st.session_state.pop(f"{k}_confirm_delete", None)
                st.rerun()

def _save_changes(mnemo_client, original_cp: dict, entity: str, value: str,
                  connects_to: str, reason: str, weight: float, category: str, key: str):
    cp_id = original_cp.get("id", "")

    # Only send fields that actually changed
    kwargs = {}
    if entity != original_cp.get("entity", ""):
        kwargs["entity"] = entity
    if value != original_cp.get("value", ""):
        kwargs["value"] = value
    if connects_to != original_cp.get("connects_to", ""):
        kwargs["connects_to"] = connects_to
    if reason != original_cp.get("reason", ""):
        kwargs["reason"] = reason
    if abs(weight - float(original_cp.get("weight", 0.5))) > 0.01:
        kwargs["weight"] = weight
    if category.lower() != original_cp.get("category", "fact").lower():
        kwargs["category"] = category.lower()

    if not kwargs:
        st.info("No changes to save.")
        return

    try:
        # V7.1.1: Uses the new clean update_point API
        result = mnemo_client.update_point(cp_id, **kwargs)
        if result and "error" not in result:
            changed_fields = ", ".join(kwargs.keys())
            st.success(f"Saved! Updated: {changed_fields}")
            
            # Sync the new values into the fast-cache loop manager
            cat_upper = category.upper()
            ent = entity.strip() or "Story"
            val = value.strip()
            if connects_to:
                loop_content = f"[{cat_upper}] {ent} → {connects_to}: {val}"
            elif ent and ent != "Story":
                loop_content = f"[{cat_upper}] {ent}: {val}"
            else:
                loop_content = f"[{cat_upper}] {val}"
                
            st.session_state.loop_manager.add_to_loop(
                content=loop_content, category=category.lower(),
                session_id=original_cp.get("session_id", ""), memory_id=cp_id)
        else:
            st.error("Failed to save changes.")
    except Exception as e:
        st.error(f"Save failed: {e}")

def _render_add_new(mnemo_client, key_prefix: str):
    k = f"{key_prefix}_new"
    with st.expander("➕ Add new memory", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            entity = st.text_input("Entity", placeholder="e.g. Sebastian Carlisle", key=f"{k}_entity")
            category = st.selectbox("Category", EDITOR_CATEGORIES, key=f"{k}_cat")
        with col2:
            connects_to = st.text_input("Connects to", placeholder="e.g. Alistair Fitzroy", key=f"{k}_conn")
            weight = st.slider("Weight", 0.0, 1.0, 0.5, 0.05, key=f"{k}_weight")

        value = st.text_area("Value", placeholder="What do you want to remember?", height=100, key=f"{k}_value")
        reason = st.text_input("Reason (optional)", key=f"{k}_reason")

        if st.button("Add Memory", key=f"{k}_add", type="primary"):
            if not entity or not value:
                st.warning("Entity and Value are required.")
            else:
                try:
                    cp_id = mnemo_client.add_point(
                        entity=entity, point_type="fact", value=value,
                        connects_to=connects_to, reason=reason,
                        weight=weight, category=category.lower(), source="manual",
                    )
                    if cp_id:
                        st.success(f"Added: {cp_id}")
                        
                        cat_upper = category.upper()
                        ent = entity.strip() or "Story"
                        val = value.strip()
                        if connects_to:
                            loop_content = f"[{cat_upper}] {ent} → {connects_to}: {val}"
                        elif ent and ent != "Story":
                            loop_content = f"[{cat_upper}] {ent}: {val}"
                        else:
                            loop_content = f"[{cat_upper}] {val}"
                            
                        st.session_state.loop_manager.add_to_loop(
                            content=loop_content, category=category.lower(),
                            session_id=st.session_state.get("current_session_id", ""), memory_id=cp_id)
                        st.rerun()
                    else:
                        st.error("Failed to add — may be a duplicate.")
                except Exception as e:
                    st.error(f"Add failed: {e}")

# ============================================================================
# UI RENDERERS
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
        render_memory_management(mnemo_client)
        st.divider()
        render_consolidation_panel(openrouter_key)

        st.divider()

        st.markdown("#### 💰 Usage")
        msg_count = st.session_state.get('message_count', 0)
        total_cost = st.session_state.get('total_cost', 0)
        st.caption(f"{msg_count} messages · ${total_cost:.4f}")
        st.caption(f"✍️ {WRITING_MODEL_ID.split('/')[-1]} · 🧠 {MEMORY_MODEL_ID.split('/')[-1]}")

        mnemo = st.session_state.get("mnemo_client")
        if mnemo and hasattr(mnemo, 'mode'):
            mode_icons = {"local": "⚡ Local (SQLite)", "remote": "🌐 Remote (Gradio)", "unavailable": "❌ Unavailable"}
            st.caption(f"💾 {mode_icons.get(mnemo.mode, mnemo.mode)}")

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
                plan = calculate_extraction_plan(content)
                chunk_info = f"{plan['n_chunks']} chunk{'s' if plan['n_chunks'] > 1 else ''}"
                st.info(f"📄 {plan['words']:,} words → {chunk_info} "
                        f"(W-method guided) + 1 thread pass "
                        f"= {plan['total_calls']} calls")
                with st.spinner(f"K2 extracting ({plan['total_calls']} calls)..."):
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
                                except Exception as e:
                                    print(f"[WARN] file upload CP storage: {type(e).__name__}: {e}")
                                    
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

                    who = [m for m in memories if m.get("category") == "CHARACTER"]
                    why = [m for m in memories if m.get("category") == "MOTIVATION"]
                    whom = [m for m in memories if m.get("category") == "RELATIONSHIP"]
                    which = [m for m in memories if m.get("category") == "PLOT"]
                    when = [m for m in memories if m.get("category") == "TIMELINE"]
                    where = [m for m in memories if m.get("category") == "SETTING"]
                    how = [m for m in memories if m.get("category") == "TONE"]
                    what = [m for m in memories if m.get("category") in ("FACT", "CLARIFICATION")]
                    st.success(f"✅ Stored {stored} CPs, {len(threads_data)} threads, {len(knots_data)} knots")
                    st.caption(f"📊 WHO:{len(who)} WHY:{len(why)} WHOM:{len(whom)} WHICH:{len(which)} "
                               f"WHEN:{len(when)} WHERE:{len(where)} HOW:{len(how)} WHAT:{len(what)} | ${cost:.4f}")
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
    @import url('[https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap](https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap)');
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
            <p>GPT-4o writer · K2 memory curator · Mnemo v7.1.1 (local)</p>
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
        except Exception as e:
            print(f"[WARN] auto_convert_blobs: {type(e).__name__}: {e}")

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
        except Exception as e:
            print(f"[WARN] session history load: {type(e).__name__}: {e}")

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
        except Exception as e:
            print(f"[WARN] signal processor init: {type(e).__name__}: {e}")
            
    if "prefetch_engine" not in st.session_state:
        st.session_state.prefetch_engine = PrefetchEngine()

    openrouter_key, hf_key = render_sidebar(mnemo_client, DEFAULT_OPENROUTER_KEY, DEFAULT_HF_KEY)

    if mnemo_client.token != hf_key:
        mnemo_client.token = hf_key

    render_chat(openrouter_key, mnemo_client)

    st.markdown(f"""
    <div class="status-bar">
        4o with Memory v7.1.1 &nbsp;·&nbsp; GPT-4o + K2 + Mnemo &nbsp;·&nbsp; Local SQLite + FAISS + NumPy &nbsp;·&nbsp; Threads & Knots
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

"""
4o with Memory - Enhanced Edition

Features:
- GPT-4o via OpenRouter with warm, conversational style
- Mnemo v4 Cloud Memory (persistent across sessions)
- Metadata Loop System (80% token savings)
- Auto-Memory Extraction
- Native JSON Structured Outputs
"""

import streamlit as st
import httpx
import asyncio
import json
import os
import uuid
from datetime import datetime
from mnemo_client import MnemoClient  # Using your new centralized client
from metadata_loops import LoopManager
from smart_memory import SmartMemory, ContextWindowManager
from session_store import SessionStore
from context_engine import ContextEngine

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
MNEMO_URL = "https://athelaperk-mnemo-mcp.hf.space"
MODEL_ID = "openai/gpt-4o-2024-11-20"
TEMPERATURE = 0.75
MAX_CONVERSATION_MESSAGES = 8
MAX_SESSIONS_STORED = 20

SYSTEM_PROMPT = """You are a warm, intelligent AI companion with persistent memory. You remember past conversations and uploaded context.

IMPORTANT — Match your response style to what the user is asking:
- FACTUAL RECALL: Give clean, factual answers drawn ONLY from memory. Do not infer.
- CREATIVE: Match the genre, tone, and atmosphere of the project. Show don't tell."""

# ============================================================================
# UTILITIES
# ============================================================================

def init_client(hf_key):
    """Initialize single global MnemoClient instance"""
    if "mnemo_client" not in st.session_state:
        st.session_state.mnemo_client = MnemoClient(base_url=MNEMO_URL, token=hf_key)
    return st.session_state.mnemo_client

def generate_session_id():
    return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

def get_persistent_storage(hf_key):
    if "persistent_storage" not in st.session_state:
        st.session_state.persistent_storage = SessionStore(hf_key=hf_key, mnemo_url=MNEMO_URL)
    return st.session_state.persistent_storage

class CostTracker:
    INPUT_COST = 2.50 / 1_000_000
    OUTPUT_COST = 15.00 / 1_000_000
    
    def __init__(self):
        if "total_cost" not in st.session_state: st.session_state.total_cost = 0.0
        if "message_count" not in st.session_state: st.session_state.message_count = 0
    
    def add_usage(self, input_tokens, output_tokens):
        cost = (input_tokens * self.INPUT_COST) + (output_tokens * self.OUTPUT_COST)
        st.session_state.total_cost += cost
        st.session_state.message_count += 1
        return cost

def call_openrouter(messages, api_key):
    try:
        response = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": MODEL_ID, "messages": messages, "temperature": TEMPERATURE, "max_tokens": 4000},
            timeout=120
        )
        data = response.json()
        usage = data.get("usage", {})
        return data["choices"][0]["message"]["content"], usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), None
    except Exception as e:
        return None, 0, 0, str(e)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(page_title="4o with Memory", page_icon="🧠", layout="wide")
    st.title("🧠 4o with Memory")
    st.caption("GPT-4o with warm, conversational style and persistent memory")
    
    if not DEFAULT_OPENROUTER_KEY or not DEFAULT_HF_KEY:
        st.error("⚠️ **API Keys Not Configured!**")
        st.stop()
        
    client = init_client(DEFAULT_HF_KEY)
    
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = generate_session_id()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    with st.sidebar:
        st.header("⚙️ Settings")
        
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            st.session_state.messages = []
            st.session_state.current_session_id = generate_session_id()
            st.rerun()
            
        st.divider()
        st.subheader("📝 Add Memory")
        
        with st.expander("Add manually", expanded=False):
            memory_category = st.selectbox("Category", ["CHARACTER", "PLOT", "SETTING", "THEME", "STYLE", "FACT"])
            memory_content = st.text_area("Content", height=80)
            if st.button("💾 Save", use_container_width=True):
                if memory_content.strip():
                    # USING MNEMO CLIENT (Cleaned up!)
                    meta = {"category": memory_category, "session_id": st.session_state.current_session_id}
                    mem_id = client.add(memory_content, meta)
                    if mem_id:
                        st.success(f"✅ Saved")
                        if "loop_manager" in st.session_state:
                            st.session_state.loop_manager.add_to_loop(memory_content, memory_category.lower())
                    else:
                        st.error("Failed to save")
                        
        with st.expander("View memories", expanded=False):
            if st.button("🔄 Refresh", use_container_width=True): st.rerun()
            
            # USING MNEMO CLIENT
            memories = client.list_memories()[:15]
            
            for mem in memories:
                col1, col2 = st.columns([5, 1])
                with col1: st.caption(f"{mem.get('content', '')[:60]}...")
                with col2:
                    if st.button("🗑️", key=f"del_{mem.get('id', '')}"):
                        client.delete(mem.get('id'))
                        st.rerun()
            
            if st.button("🧹 Clear ALL", use_container_width=True):
                client.clear()
                st.rerun()
        
        st.divider()
        
        with st.expander("🧠 Memory Consolidation", expanded=False):
            if "context_engine" in st.session_state:
                if st.button("🧠 Consolidate Now", use_container_width=True):
                    with st.spinner("Analyzing memories..."):
                        result = st.session_state.context_engine.consolidate_memories(DEFAULT_OPENROUTER_KEY)
                        if result.get("error"):
                            st.error(result['error'])
                        else:
                            st.success(f"✅ Created {result['created']} context entries!")
                            st.session_state.loop_manager.load_from_mnemo(use_smart_extraction=False)
                            
    # Core Engines Initialization
    if "loop_manager" not in st.session_state:
        st.session_state.loop_manager = LoopManager(openrouter_key=DEFAULT_OPENROUTER_KEY, hf_key=DEFAULT_HF_KEY, mnemo_url=MNEMO_URL)
        st.session_state.loop_manager.load_from_mnemo(use_smart_extraction=False)
    if "smart_memory" not in st.session_state:
        st.session_state.smart_memory = SmartMemory()
    if "context_engine" not in st.session_state:
        st.session_state.context_engine = ContextEngine(hf_key=DEFAULT_HF_KEY, openrouter_key=DEFAULT_OPENROUTER_KEY, mnemo_url=MNEMO_URL)
    if "context_manager" not in st.session_state:
        st.session_state.context_manager = ContextWindowManager(loop_manager=st.session_state.loop_manager)
    
    # Render chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "metadata" in message:
                meta = message["metadata"]
                st.caption(f"🧠 {meta.get('context_tokens', 0)} context tokens | 💰 ${meta.get('cost', 0):.4f}")
                
    if prompt := st.chat_input("What's on your mind?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                needs_memory, memory_reason = st.session_state.smart_memory.should_use_memory(prompt, len(st.session_state.messages))
                
                messages, context_stats = st.session_state.context_manager.build_optimized_context(
                    system_prompt=SYSTEM_PROMPT, query=prompt, 
                    conversation_history=st.session_state.messages, 
                    max_messages=MAX_CONVERSATION_MESSAGES, use_loops=needs_memory
                )
                
                response, in_tok, out_tok, error = call_openrouter(messages, DEFAULT_OPENROUTER_KEY)
                
                if error:
                    st.error(error)
                else:
                    tracker = CostTracker()
                    msg_cost = tracker.add_usage(in_tok, out_tok)
                    
                    st.markdown(response)
                    st.caption(f"🧠 {context_stats.get('total_tokens', 0)} context tokens | 💰 ${msg_cost:.4f}")
                    
                    st.session_state.messages.append({
                        "role": "assistant", "content": response,
                        "metadata": {"context_tokens": context_stats.get('total_tokens', 0), "cost": msg_cost}
                    })

if __name__ == "__main__":
    main()

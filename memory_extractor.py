"""
Auto Memory Extraction Module

Uses GPT-4o to automatically extract important facts from conversations
and store them in Mnemo memory system.
"""

import os
import requests
import json
from typing import List, Dict, Optional


class MemoryExtractor:
    """
    Extracts memories from conversations using GPT-4o and stores in Mnemo.
    """
    
    def __init__(self, openrouter_key: str, mnemo_url: str, hf_key: str):
        self.openrouter_key = openrouter_key
        self.mnemo_url = mnemo_url.rstrip('/')
        self.hf_key = hf_key
        self.mnemo_headers = {
            "Authorization": f"Bearer {hf_key}",
            "Content-Type": "application/json"
        }
    
    def extract_memories(self, conversation: str) -> List[Dict]:
        """
        Use GPT-4o to extract important facts from conversation.
        Returns list of {category, content} dicts.
        """
        
        extraction_prompt = f"""Analyze this conversation and extract IMPORTANT FACTS that should be remembered.

CONVERSATION:
{conversation}

Extract facts in these categories:
- CHARACTER: Names, traits, relationships, backstories, motivations
- PLOT: Events, twists, conflicts, resolutions
- SETTING: Locations, time periods, world-building
- THEME: Recurring themes, symbols, motifs
- STYLE: Writing preferences, tone, format requests
- FACT: Any other important information

Rules:
1. Only extract NEW information not already obvious
2. Be specific and detailed
3. Each fact should be self-contained
4. Skip generic/obvious statements

Return ONLY a JSON object containing an array called "memories". Example format:
{{
  "memories": [
    {{"category": "CHARACTER", "content": "description"}}
  ]
}}

If nothing important to extract, return: {{"memories": []}}"""

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openrouter_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "openai/gpt-4o-2024-11-20",
                    "messages": [
                        {
                            "role": "system", 
                            "content": "You extract important facts from creative writing conversations. Return ONLY a valid JSON object. No explanations."
                        },
                        {"role": "user", "content": extraction_prompt}
                    ],
                    "temperature": 0.2,
                    "max_tokens": 1000,
                    "response_format": {"type": "json_object"} # NATIVE JSON MODE
                },
                timeout=30
            )
            
            if response.status_code != 200:
                return []
            
            data = response.json()
            raw = data["choices"][0]["message"]["content"]
            
            # Direct JSON loading - no more regex or string splitting needed!
            parsed = json.loads(raw)
            memories = parsed.get("memories", [])
            return memories if isinstance(memories, list) else []
            
        except Exception as e:
            print(f"Extraction error: {e}")
            return []
    
    def store_memory(self, content: str, category: str = "GENERAL", 
                     metadata: dict = None) -> Optional[str]:
        """Store a single memory in Mnemo."""
        try:
            payload = {
                "content": f"[{category}] {content}",
                "metadata": metadata or {"category": category}
            }
            response = requests.post(
                f"{self.mnemo_url}/add",
                headers=self.mnemo_headers,
                json=payload,
                timeout=10
            )
            if response.status_code == 200:
                return response.json().get("memory_id")
            return None
        except Exception:
            return None
    
    def store_memories(self, memories: List[Dict]) -> int:
        """Store multiple memories in Mnemo. Returns count stored."""
        stored = 0
        for mem in memories:
            category = mem.get("category", "GENERAL")
            content = mem.get("content", "")
            if content:
                if self.store_memory(content, category, {"auto_extracted": True}):
                    stored += 1
        return stored
    
    def extract_and_store(self, conversation: str) -> Dict:
        """
        Full pipeline: extract memories from conversation and store in Mnemo.
        Returns stats dict.
        """
        memories = self.extract_memories(conversation)
        stored = self.store_memories(memories) if memories else 0
        
        return {
            "extracted": len(memories),
            "stored": stored,
            "memories": memories
        }
    
    def manual_add(self, content: str, category: str = "GENERAL") -> bool:
        """Manually add a memory."""
        return self.store_memory(content, category, {"manual": True}) is not None
    
    def list_memories(self) -> List[Dict]:
        """List all memories from Mnemo."""
        try:
            response = requests.get(
                f"{self.mnemo_url}/list",
                headers=self.mnemo_headers,
                timeout=10
            )
            if response.status_code == 200:
                return response.json().get("memories", [])
            return []
        except Exception:
            return []
    
    def clear_memories(self) -> bool:
        """Clear all memories from Mnemo."""
        try:
            response = requests.post(
                f"{self.mnemo_url}/clear",
                headers=self.mnemo_headers,
                json={"confirm": True},
                timeout=10
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def get_stats(self) -> Dict:
        """Get Mnemo stats."""
        try:
            response = requests.get(
                f"{self.mnemo_url}/stats",
                headers=self.mnemo_headers,
                timeout=5
            )
            if response.status_code == 200:
                return response.json().get("stats", {})
            return {}
        except Exception:
            return {}


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    # Test the extractor
    OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY", "")
    HF_KEY = os.environ.get("HF_KEY", "")
    MNEMO_URL = "https://athelaperk-mnemo-mcp.hf.space"
    
    extractor = MemoryExtractor(OPENROUTER_KEY, MNEMO_URL, HF_KEY)
    
    print("=" * 60)
    print("TESTING AUTO MEMORY EXTRACTION")
    print("=" * 60)
    
    # Test conversation
    test_conv = """
User: I want to add a new character - Dr. Clara Hendricks, a rogue physicist who secretly helps an underground resistance. She studied at MIT and has radical views on government transparency.
"""
    
    print(f"Test conversation length: {len(test_conv)} chars")
    
    # Removed the invalid session_id keyword argument from the test call
    memories = extractor.extract_memories(test_conv)
    print(f"Extracted {len(memories)} memories")
    for mem in memories:
        print(f"  - {mem}")

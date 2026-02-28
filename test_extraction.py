#!/usr/bin/env python3
"""Test auto memory extraction feature"""

import os
import requests
import json

OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY", "")
HF_KEY = os.environ.get("HF_KEY", "")
MNEMO_URL = "https://athelaperk-mnemo-mcp.hf.space"

mnemo_headers = {
    "Authorization": f"Bearer {HF_KEY}",
    "Content-Type": "application/json"
}


def extract_memories_with_gpt(conversation):
    """Use GPT-4o to extract important facts from conversation."""
    
    prompt = """Analyze this conversation and extract IMPORTANT FACTS to remember.

CONVERSATION:
""" + conversation + """

Extract facts in these categories:
- CHARACTER: Names, traits, relationships
- PLOT: Events, twists, conflicts  
- SETTING: Locations, time periods
- THEME: Themes, symbols
- STYLE: Writing preferences

Return ONLY a valid JSON object containing an array called "memories". Example format:
{"memories": [{"category": "CHARACTER", "content": "description"}]}

If nothing to extract, return: {"memories": []}"""

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "openai/gpt-4o-2024-11-20",
                "messages": [
                    {"role": "system", "content": "Extract facts as a JSON object. No explanations."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 1000,
                "response_format": {"type": "json_object"}  # NATIVE JSON MODE
            },
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"API Error: {response.status_code} - {response.text}")
            return [], 0.0
            
        data = response.json()
        usage = data.get("usage", {})
        cost = (usage.get("prompt_tokens", 0) * 2.5 + usage.get("completion_tokens", 0) * 15) / 1_000_000
        
        raw = data["choices"][0]["message"]["content"]
        
        # Direct JSON loading - no more regex or string splitting needed!
        parsed = json.loads(raw)
        memories = parsed.get("memories", [])
        return memories if isinstance(memories, list) else [], cost
        
    except Exception as e:
        print(f"Extraction error: {e}")
        return [], 0.0


def store_in_mnemo(memories):
    """Store extracted memories in Mnemo."""
    stored = 0
    for mem in memories:
        cat = mem.get("category", "GENERAL")
        content = mem.get("content", "")
        if content:
            response = requests.post(
                f"{MNEMO_URL}/add",
                headers=mnemo_headers,
                json={
                    "content": f"[{cat}] {content}",
                    "metadata": {"category": cat, "auto_extracted": True}
                },
                timeout=10
            )
            if response.status_code == 200:
                stored += 1
    return stored


def manual_add(content, category="GENERAL"):
    """Manually add a memory."""
    response = requests.post(
        f"{MNEMO_URL}/add",
        headers=mnemo_headers,
        json={
            "content": f"[{category}] {content}",
            "metadata": {"category": category, "manual": True}
        },
        timeout=10
    )
    return response.status_code == 200


def get_stats():
    """Get Mnemo stats."""
    response = requests.get(f"{MNEMO_URL}/stats", headers=mnemo_headers, timeout=5)
    return response.json().get("stats", {}) if response.status_code == 200 else {}


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING AUTO MEMORY EXTRACTION")
    print("=" * 60)
    
    # Test conversation about adding a new character
    test_conv = "User: I want to add a new character - Dr. Clara Hendricks, a rogue scientist who secretly helps an underground resistance network. She studied physics at MIT under controversial professors and has radical views on government surveillance. She is in her early 40s, divorced, and uses her tech startup fortune to fund safe houses for whistleblowers.\n\nAssistant: That sounds like a fascinating character! Dr. Clara Hendricks would fit perfectly as a bridge between the resistance and the establishment. Her MIT connections could link her to both sides. Want me to develop her backstory further?\n\nUser: Yes, and I also want to establish that she has a complicated relationship with Marcus - they respect each other but disagree on tactics. Clara believes in working within the system while Marcus wants to tear it down."

    print("\nTest conversation:")
    print(test_conv[:200] + "...")
    
    # Step 1: Extract memories using GPT-4o
    print("\n" + "-" * 60)
    print("Step 1: Extracting memories with GPT-4o...")
    print("-" * 60)
    
    memories, cost = extract_memories_with_gpt(test_conv)
    
    print(f"Extraction cost: ${cost:.4f}")
    print(f"Memories extracted: {len(memories)}")
    
    for i, mem in enumerate(memories):
        content = mem.get("content", "")[:80]
        print(f"  {i+1}. [{mem.get('category')}] {content}...")
    
    # Step 2: Store in Mnemo
    print("\n" + "-" * 60)
    print("Step 2: Storing in Mnemo...")
    print("-" * 60)
    
    stored = store_in_mnemo(memories)
    print(f"Memories stored: {stored}/{len(memories)}")
    
    # Step 3: Test manual add
    print("\n" + "-" * 60)
    print("Step 3: Testing manual memory add...")
    print("-" * 60)
    
    success = manual_add("Clara and Marcus have ideological tension - reform vs revolution", "PLOT")
    print(f"Manual add: {'Success' if success else 'Failed'}")
    
    # Step 4: Check stats
    print("\n" + "-" * 60)
    print("Step 4: Mnemo stats...")
    print("-" * 60)
    
    stats = get_stats()
    print(f"Total memories: {stats.get('total_memories', 0)}")
    print(f"Total adds: {stats.get('adds', 0)}")
    print(f"Neural links: {stats.get('total_links', 0)}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE!")
    print("=" * 60)

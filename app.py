"""
4o with Memory v6.2 Patch — Entity Normalization + Retrieval Fix

Apply these changes to your Streamlit app.py:

1. Add normalize_entity() function (new)
2. Replace store_as_cp() to use normalization
3. Replace extract_memories_with_gpt() prompt (better entity naming rules)
4. Replace process_chunk_async() prompt (same fix for file uploads)

Also deploy the updated mnemo_core.py to AthelaPerk/mnemo Space.
"""

import re


# ==========================================================================
# FIX 1: Add this function AFTER the CATEGORY_TO_CP dict (around line 97)
# ==========================================================================

def normalize_entity(raw_entity: str) -> str:
    """v6.2: Normalize entity name for consistent EntityIndex storage.
    
    Fixes K2 extraction issues that prevent graph_search from matching:
    1. Strip category prefixes: "Clarification: Sebastian's Role" → "Sebastian"
    2. Strip honorifics: "Dr. Sebastian Carlisle" → "Sebastian Carlisle"
    3. Strip articles: "The Midnight Salon" → "Midnight Salon"
    4. Detect non-entities: "whispered prayers" → "Story" (fallback)
    5. Collapse possessives: "Sebastian's Role" → "Sebastian"
    
    Without this, K2 produces entity names like "Dr. Sebastian Carlisle"
    which the EntityIndex stores as "dr. sebastian carlisle" — but queries
    like "Tell me about Sebastian" extract "Sebastian" and get zero hits.
    """
    entity = raw_entity.strip()
    if not entity:
        return "Story"
    
    # 1. Strip category prefixes (K2 sometimes leaks these into entity field)
    category_prefixes = [
        "Clarification:", "Tone:", "Context:", "Relationship:",
        "Timeline:", "Setting:", "Theme:", "Instruction:",
        "Style:", "Voice:", "Fact:",
    ]
    for prefix in category_prefixes:
        if entity.startswith(prefix):
            entity = entity[len(prefix):].strip()
    
    # 2. Strip honorifics and titles
    title_prefixes = [
        "Dr. ", "Dr ", "Reverend ", "Rev. ", "Rev ",
        "Professor ", "Prof. ", "Prof ", "Lord ", "Lady ",
        "Sir ", "Dame ", "Mr. ", "Mrs. ", "Ms. ", "Miss ",
    ]
    for prefix in title_prefixes:
        if entity.startswith(prefix):
            entity = entity[len(prefix):].strip()
            break
    
    # 3. Strip leading articles
    for article in ["The ", "the ", "A ", "a ", "An ", "an "]:
        if entity.startswith(article) and len(entity) > len(article) + 3:
            entity = entity[len(article):].strip()
            break
    
    # 4. Detect non-entities: must have at least one capitalized word
    words = entity.split()
    has_proper_noun = any(w[0].isupper() for w in words if len(w) > 0)
    if not has_proper_noun:
        return "Story"
    
    # 5. Collapse possessives: "Sebastian's Role" → "Sebastian"
    poss_match = re.match(r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'s\s+\w+", entity)
    if poss_match:
        entity = poss_match.group(1)
    
    return entity if entity else "Story"


# ==========================================================================
# FIX 2: Replace store_as_cp() — adds normalize_entity() call
# ==========================================================================

def store_as_cp(mnemo_client, entity, category, content, session_id="",
                source="auto_extract", connects_to="", weight=0.5):
    """v6.2: Store a memory as a ConnectionPoint with entity normalization."""
    cat_key = category.upper()
    # Import from module-level CATEGORY_TO_CP dict
    cp_category, point_type = CATEGORY_TO_CP.get(cat_key, ("fact", "general"))
    
    # v6.2: Normalize entity name before storage
    normalized_entity = normalize_entity(entity)
    normalized_connects = normalize_entity(connects_to) if connects_to else ""
    
    return mnemo_client.add_point(
        entity=normalized_entity, point_type=point_type, value=content,
        connects_to=normalized_connects, reason="", weight=float(weight),
        category=cp_category, session_id=session_id, source=source,
    )


# ==========================================================================
# FIX 3: Updated extraction prompt for extract_memories_with_gpt()
#         (conversational auto-extract)
# ==========================================================================

CONVERSATION_EXTRACTION_PROMPT = """Extract structured memories from this conversation by asking the W questions about everything mentioned.

CONVERSATION:
{conversation}

METHOD — For every character, event, or topic mentioned, ask:
  WHO is this? → CHARACTER entry
  WHAT happened? → PLOT entry 
  WHOM do they affect/relate to? → RELATIONSHIP entry 
  WHY does it matter? → CONTEXT entry 
  WHERE does it happen? → SETTING entry 
  WHEN in the timeline? → PLOT entry 
  HOW should it feel? → TONE entry 
  WHOSE rule or instruction? → CLARIFICATION entry 

ENTRY FORMAT:
  {{"entity": "Name", "category": "CATEGORY", "content": "1-3 sentence answer"}}
  Add "connects_to": "OtherName" only for RELATIONSHIP entries.

CRITICAL ENTITY NAMING RULES:
  - Use the character's plain name: "Sebastian Carlisle", NOT "Dr. Sebastian Carlisle"
  - NO titles or honorifics: "Elijah Cartwright", NOT "Reverend Elijah Cartwright"  
  - NO articles: "Midnight Salon", NOT "The Midnight Salon"
  - NO category prefixes: entity is "Sebastian", NOT "Clarification: Sebastian's Role"
  - For organizations: "Red Rose Society", "Progressive Women's Society"
  - For themes/concepts without a named entity: use "Story" as the entity
  - The entity field is a NAME for indexing. The content field holds the description.

CATEGORIES: CHARACTER, PLOT, SETTING, THEME, FACT, CONTEXT, CLARIFICATION, RELATIONSHIP, INSTRUCTION, STYLE, TONE

Return ONLY a JSON object:
{{
  "memories": [
    {{"entity": "Name", "category": "CATEGORY", "content": "information"}}
  ]
}}"""


# ==========================================================================
# FIX 4: Updated extraction prompt for process_chunk_async()
#         (file upload extraction)
# ==========================================================================

FILE_EXTRACTION_PROMPT = """Extract structured memories from this document by asking the W questions about everything mentioned.

DOCUMENT: {filename}{chunk_label}

CONTENT:
{chunk}

METHOD — For every character, event, faction, place, or rule mentioned, ask:
  WHO is this? → CHARACTER entry (traits, fears, motivations, background)
  WHAT happened / what do they do? → PLOT entry (events, schemes, outcomes)
  WHOM do they connect to? → RELATIONSHIP entry (dynamics, power balance, emotional undercurrent)
  WHY does it matter? → CONTEXT entry (deeper meaning, narrative function, what could be misread)
  WHERE does it happen? → SETTING entry (location, atmosphere, sensory detail)
  WHEN in the timeline? → PLOT entry (sequence, book number, what precedes/follows)
  HOW should it feel on the page? → TONE entry (register, humor type, do-nots)
  WHOSE interpretation could go wrong? → CLARIFICATION entry (prevent specific misreadings)

CRITICAL ENTITY NAMING RULES:
  - Use the character's plain name: "Sebastian Carlisle", NOT "Dr. Sebastian Carlisle"
  - NO titles or honorifics: "Elijah Cartwright", NOT "Reverend Elijah Cartwright"  
  - NO articles: "Midnight Salon", NOT "The Midnight Salon"
  - NO category prefixes in entity: entity is "Sebastian", NOT "Clarification: Sebastian"
  - For factions/organizations: "Red Rose Society", "Progressive Women's Society"
  - For themes/concepts without a named entity: use "Story" as the entity
  - The entity field is a LOOKUP KEY. Keep it short and consistent.

ENTRY FORMAT:
  {{"entity": "Name", "category": "CATEGORY", "content": "1-3 sentence answer"}}
  Add "connects_to": "OtherName" only for RELATIONSHIP entries.

CATEGORIES: CHARACTER, PLOT, SETTING, THEME, FACT, CONTEXT, CLARIFICATION, RELATIONSHIP, INSTRUCTION, STYLE, TONE

Return ONLY a JSON object:
{{
  "memories": [
    {{"entity": "Name", "category": "CATEGORY", "content": "information"}}
  ]
}}"""

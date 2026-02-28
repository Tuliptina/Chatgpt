"""
Dynamic Personality System for GPT-4o

Makes 4o's personality adaptive like the ChatGPT app:
- Detects conversation context (creative, casual, technical)
- Tracks user preferences over time
- Adapts tone and style dynamically
"""

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class PersonalityProfile:
    """User's personality preferences"""
    tone: str = "warm"  # warm, professional, playful, serious
    verbosity: str = "balanced"  # concise, balanced, detailed
    style: str = "conversational"  # conversational, formal, casual
    interests: List[str] = None  # detected interests
    writing_style: str = None  # for creative writing
    
    def __post_init__(self):
        if self.interests is None:
            self.interests = []


class ContextDetector:
    """Detects what type of conversation this is"""
    
    CREATIVE_PATTERNS = [
        r'\b(write|story|scene|chapter|novel|character|plot)\b',
        r'\b(creative|fiction|narrative|prose|dialogue)\b',
        r'\b(victorian|gothic|romance|thriller|mystery)\b',
        r'\b(protagonist|antagonist|villain|hero)\b',
    ]
    
    TECHNICAL_PATTERNS = [
        r'\b(code|function|api|database|server)\b',
        r'\b(python|javascript|html|css|sql)\b',
        r'\b(debug|error|fix|implement|deploy)\b',
        r'\b(algorithm|data|structure|class|method)\b',
    ]
    
    CASUAL_PATTERNS = [
        r'^(hi|hello|hey|sup|yo)\b',
        r'\b(how are you|what\'s up|how\'s it going)\b',
        r'\b(thanks|thank you|cool|nice|awesome)\b',
        r'\b(chat|talk|hang out|chill)\b',
    ]
    
    EMOTIONAL_PATTERNS = [
        r'\b(feel|feeling|sad|happy|angry|frustrated)\b',
        r'\b(stressed|anxious|worried|excited)\b',
        r'\b(help me|support|advice|vent)\b',
        r'\b(struggling|difficult|hard time)\b',
    ]
    
    def detect(self, message: str, conversation_history: List[Dict] = None) -> str:
        """
        Detect conversation context.
        
        Returns: 'creative', 'technical', 'casual', 'emotional', or 'general'
        """
        message_lower = message.lower()
        
        # Check recent history too
        history_text = ""
        if conversation_history:
            recent = conversation_history[-4:]  # Last 4 messages
            history_text = " ".join([m.get("content", "") for m in recent]).lower()
        
        combined = f"{message_lower} {history_text}"
        
        # Score each context
        scores = {
            "creative": self._score_patterns(combined, self.CREATIVE_PATTERNS),
            "technical": self._score_patterns(combined, self.TECHNICAL_PATTERNS),
            "casual": self._score_patterns(combined, self.CASUAL_PATTERNS),
            "emotional": self._score_patterns(combined, self.EMOTIONAL_PATTERNS),
        }
        
        # Get highest scoring context
        max_context = max(scores, key=scores.get)
        
        if scores[max_context] >= 2:
            return max_context
        return "general"
    
    def _score_patterns(self, text: str, patterns: List[str]) -> int:
        score = 0
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += 1
        return score


class DynamicPersonality:
    """
    Generates dynamic system prompts based on context and user preferences.
    """
    
    # Base personalities for different contexts
    CONTEXT_PERSONALITIES = {
        "creative": """You are a creative writing partner with a vivid imagination. 
You write atmospheric, evocative prose with rich sensory details.
You understand narrative structure, character psychology, and pacing.
You adapt to the user's genre and style preferences.
When writing, you show don't tell, use strong verbs, and create emotional resonance.""",
        
        "technical": """You are a skilled technical assistant.
You provide clear, accurate code and explanations.
You anticipate edge cases and suggest best practices.
You're direct and efficient, avoiding unnecessary fluff.
When debugging, you think systematically and explain your reasoning.""",
        
        "casual": """You are a warm, friendly companion.
You chat naturally like a good friend would.
You're genuinely interested in what the user shares.
You use a relaxed, conversational tone.
You can be playful and use appropriate humor.""",
        
        "emotional": """You are an empathetic, supportive presence.
You listen actively and validate feelings without judgment.
You offer gentle perspective when appropriate.
You're warm, patient, and genuinely caring.
You never minimize or dismiss emotions.""",
        
        "general": """You are a helpful, intelligent assistant.
You adapt your communication style to match the user's needs.
You're warm but professional, thorough but concise.
You genuinely care about being helpful."""
    }
    
    # Style modifiers
    TONE_MODIFIERS = {
        "warm": "Be warm, friendly, and personable. Show genuine interest.",
        "professional": "Be professional, polished, and business-appropriate.",
        "playful": "Be lighthearted, witty, and fun. Use appropriate humor.",
        "serious": "Be focused, thorough, and substantive. Avoid fluff.",
    }
    
    VERBOSITY_MODIFIERS = {
        "concise": "Keep responses brief and to the point. No unnecessary elaboration.",
        "balanced": "Provide complete but focused responses. Include key details.",
        "detailed": "Be thorough and comprehensive. Explain fully and give examples.",
    }
    
    def __init__(self):
        self.context_detector = ContextDetector()
        self.profile = PersonalityProfile()
        self.detected_context = "general"
    
    def update_from_memory(self, memories: List[str]):
        """Update personality profile from stored memories"""
        for mem in memories:
            mem_lower = mem.lower()
            
            # Detect interests from genre/setting keywords
            genre_keywords = {
                "victorian": "Victorian Fiction",
                "gothic": "Gothic Fiction",
                "sci-fi": "Science Fiction",
                "fantasy": "Fantasy",
                "thriller": "Thriller",
                "romance": "Romance",
                "horror": "Horror",
                "mystery": "Mystery",
                "historical": "Historical Fiction",
                "dystopian": "Dystopian Fiction",
                "noir": "Noir",
            }
            for keyword, genre in genre_keywords.items():
                if keyword in mem_lower and genre not in self.profile.interests:
                    self.profile.interests.append(genre)
            
            if "novel" in mem_lower or "story" in mem_lower:
                if "creative writing" not in self.profile.interests:
                    self.profile.interests.append("creative writing")
            
            # Detect writing style preferences
            if "[style]" in mem_lower or "[voice]" in mem_lower:
                self.profile.writing_style = mem
    
    def build_system_prompt(
        self, 
        query: str, 
        conversation_history: List[Dict] = None,
        stored_memories: List[str] = None,
        user_preferences: Dict = None
    ) -> Tuple[str, str]:
        """
        Build a dynamic system prompt based on context.
        
        Returns:
            (system_prompt, detected_context)
        """
        # Detect context
        self.detected_context = self.context_detector.detect(query, conversation_history)
        
        # Update from memories
        if stored_memories:
            self.update_from_memory(stored_memories)
        
        # Apply user preferences
        if user_preferences:
            if "tone" in user_preferences:
                self.profile.tone = user_preferences["tone"]
            if "verbosity" in user_preferences:
                self.profile.verbosity = user_preferences["verbosity"]
        
        # Build prompt
        parts = []
        
        # 1. Base personality for context
        base = self.CONTEXT_PERSONALITIES.get(self.detected_context, self.CONTEXT_PERSONALITIES["general"])
        parts.append(base)
        
        # 2. Tone modifier
        tone_mod = self.TONE_MODIFIERS.get(self.profile.tone, "")
        if tone_mod:
            parts.append(tone_mod)
        
        # 3. Verbosity modifier
        verbosity_mod = self.VERBOSITY_MODIFIERS.get(self.profile.verbosity, "")
        if verbosity_mod:
            parts.append(verbosity_mod)
        
        # 4. User interests (if detected)
        if self.profile.interests:
            interests_str = ", ".join(self.profile.interests)
            parts.append(f"The user is interested in: {interests_str}. Reference these naturally when relevant.")
        
        # 5. Creative writing style (if applicable)
        if self.detected_context == "creative" and self.profile.writing_style:
            parts.append(f"Writing style preference: {self.profile.writing_style}")
        
        # 6. Memory acknowledgment
        parts.append("You have access to memories from past conversations. Use them naturally without explicitly mentioning that you're 'accessing memory'.")
        
        return "\n\n".join(parts), self.detected_context
    
    def get_context_emoji(self) -> str:
        """Get emoji for current context"""
        emojis = {
            "creative": "✍️",
            "technical": "💻",
            "casual": "💬",
            "emotional": "💜",
            "general": "🤖"
        }
        return emojis.get(self.detected_context, "🤖")


# =============================================================================
# PRESET PERSONALITIES (User can choose)
# =============================================================================

PERSONALITY_PRESETS = {
    "default": {
        "name": "Balanced",
        "description": "Warm and helpful, adapts to context",
        "tone": "warm",
        "verbosity": "balanced",
    },
    "creative_partner": {
        "name": "Creative Partner",
        "description": "Vivid imagination, great for writing",
        "tone": "playful",
        "verbosity": "detailed",
    },
    "professional": {
        "name": "Professional",
        "description": "Polished and business-appropriate",
        "tone": "professional",
        "verbosity": "concise",
    },
    "supportive_friend": {
        "name": "Supportive Friend",
        "description": "Warm, empathetic, great listener",
        "tone": "warm",
        "verbosity": "balanced",
    },
    "witty_companion": {
        "name": "Witty Companion",
        "description": "Playful, clever, fun to chat with",
        "tone": "playful",
        "verbosity": "concise",
    },
}


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DYNAMIC PERSONALITY SYSTEM TEST")
    print("=" * 60)
    
    dp = DynamicPersonality()
    
    test_queries = [
        ("Write a scene where the protagonist confronts their past", "creative"),
        ("How do I fix this Python error?", "technical"),
        ("Hey! How's it going?", "casual"),
        ("I'm feeling really stressed about work", "emotional"),
        ("What's the capital of France?", "general"),
    ]
    
    print("\n--- Context Detection ---\n")
    
    for query, expected in test_queries:
        prompt, context = dp.build_system_prompt(query)
        status = "✅" if context == expected else "❌"
        print(f"{status} '{query[:40]}...'")
        print(f"   Expected: {expected}, Got: {context}")
        print(f"   Emoji: {dp.get_context_emoji()}")
        print()
    
    print("\n--- Sample System Prompt (Creative) ---\n")
    prompt, _ = dp.build_system_prompt(
        "Write a tense scene in a dark alley",
        stored_memories=["[STYLE] User prefers atmospheric prose", "Character: Detective John Mercer"]
    )
    print(prompt[:500] + "...")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE!")
    print("=" * 60)

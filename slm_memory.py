"""
SLM-Enhanced Memory System for 4o with Memory

Implements key features from the SLM Blockchain AI Memory System:
1. Three-Tier Memory Hierarchy (Working → Token → Semantic)
2. Memory Decay & Priority Scoring
3. Neural Link Types with Strength Evolution
4. Folder-Based Organization
5. Promotion/Demotion Algorithms
6. Memory Utility Prediction

Based on: SLM Blockchain AI Memory System Architecture
"""

import os

import requests
import json
import math
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum


# =============================================================================
# CONFIGURATION - SLM Parameters
# =============================================================================

class SLMConfig:
    """SLM System Parameters from the architecture document"""
    
    # Memory Tier Sizes
    WORKING_MEMORY_MAX_ITEMS = 32  # Active items
    TOKEN_MEMORY_LOOP_SIZE = 100  # Default loop capacity
    TOKEN_MEMORY_MAX_LOOPS = 250  # Expandable limit
    
    # Decay Rates
    WORKING_MEMORY_DECAY_RATE = 0.95  # Per minute
    PRIORITY_THRESHOLD = 0.2  # Eviction threshold
    
    # Promotion/Demotion Thresholds
    PROMOTION_THRESHOLD = 0.65
    DEMOTION_THRESHOLD = 0.75
    WORKING_MEMORY_CAPACITY_TRIGGER = 0.80
    
    # Similarity Thresholds
    SIMILARITY_STRICT = 0.92  # Entity matching
    SIMILARITY_STANDARD = 0.65  # Semantic search
    SIMILARITY_PERMISSIVE = 0.55  # Exploratory
    
    # Neural Link Parameters
    LINK_CREATION_THRESHOLD = 0.68
    LINK_MERGE_THRESHOLD = 0.80
    LINK_PRUNE_THRESHOLD = 0.30
    LINK_PRUNE_AGE_DAYS = 60
    MAX_PATH_DEPTH = 4
    PATH_STRENGTH_DECAY = 0.9
    
    # Context Window
    STANDARD_CONTEXT_TOKENS = 8000
    EXPANDED_CONTEXT_TOKENS = 32000
    EXPANSION_THRESHOLD = 0.8


# =============================================================================
# ENUMS & DATA CLASSES
# =============================================================================

class MemoryTier(Enum):
    """Three-tier memory hierarchy"""
    WORKING = "working"    # Active, immediate access
    TOKEN = "token"        # Compressed, millisecond access
    SEMANTIC = "semantic"  # Full knowledge, persistent


class LinkType(Enum):
    """Eight neural link types from SLM architecture"""
    DIRECT_REFERENCE = "direct_reference"      # Explicit reference
    SEMANTIC_SIMILARITY = "semantic_similarity" # Vector similarity
    TEMPORAL_SEQUENCE = "temporal_sequence"    # Time-based
    CAUSAL = "causal"                          # Cause-effect
    HIERARCHICAL = "hierarchical"              # Parent-child
    ASSOCIATIVE = "associative"                # Co-occurrence
    CROSS_DOMAIN = "cross_domain"              # Different domains
    CONTRADICTORY = "contradictory"            # Conflicts


@dataclass
class LinkTypeConfig:
    """Configuration for each link type"""
    creation_threshold: float
    default_strength: float
    decay_rate: float  # Per day


# Link type configurations from SLM spec
LINK_CONFIGS = {
    LinkType.DIRECT_REFERENCE: LinkTypeConfig(0.70, 0.90, 0.005),
    LinkType.SEMANTIC_SIMILARITY: LinkTypeConfig(0.68, 0.75, 0.010),
    LinkType.TEMPORAL_SEQUENCE: LinkTypeConfig(0.60, 0.80, 0.015),
    LinkType.CAUSAL: LinkTypeConfig(0.75, 0.85, 0.008),
    LinkType.HIERARCHICAL: LinkTypeConfig(0.72, 0.85, 0.005),
    LinkType.ASSOCIATIVE: LinkTypeConfig(0.65, 0.70, 0.020),
    LinkType.CROSS_DOMAIN: LinkTypeConfig(0.80, 0.70, 0.012),
    LinkType.CONTRADICTORY: LinkTypeConfig(0.85, 0.80, 0.025),
}


class MemoryCategory(Enum):
    """Memory categories for folder organization"""
    CHARACTER = "character"
    PLOT = "plot"
    SETTING = "setting"
    THEME = "theme"
    STYLE = "style"
    FACT = "fact"
    PREFERENCE = "preference"
    GENERAL = "general"


@dataclass
class MemoryItem:
    """Enhanced memory item with SLM features"""
    id: str
    content: str
    tier: MemoryTier = MemoryTier.SEMANTIC
    category: MemoryCategory = MemoryCategory.GENERAL
    folder: str = "/"
    
    # Scoring
    priority: float = 0.5
    importance: float = 0.5
    relevance_score: float = 0.5
    
    # Access tracking
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)
    
    # Metadata
    embedding: List[float] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    source: str = "auto"  # auto, manual, extracted
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "tier": self.tier.value,
            "category": self.category.value,
            "folder": self.folder,
            "priority": self.priority,
            "importance": self.importance,
            "relevance_score": self.relevance_score,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat(),
            "created_at": self.created_at.isoformat(),
            "keywords": self.keywords,
            "source": self.source
        }


@dataclass
class NeuralLink:
    """Neural link between memory items"""
    source_id: str
    target_id: str
    link_type: LinkType
    strength: float
    created_at: datetime = field(default_factory=datetime.now)
    last_traversed: datetime = field(default_factory=datetime.now)
    traversal_count: int = 0
    
    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "link_type": self.link_type.value,
            "strength": self.strength,
            "created_at": self.created_at.isoformat(),
            "last_traversed": self.last_traversed.isoformat(),
            "traversal_count": self.traversal_count
        }


@dataclass
class MemoryFolder:
    """Folder for organizing memories"""
    path: str
    name: str
    description: str = ""
    color: str = "#808080"
    parent: str = "/"
    created_at: datetime = field(default_factory=datetime.now)
    memory_count: int = 0
    
    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "name": self.name,
            "description": self.description,
            "color": self.color,
            "parent": self.parent,
            "created_at": self.created_at.isoformat(),
            "memory_count": self.memory_count
        }


# =============================================================================
# SLM MEMORY SCORING SYSTEM
# =============================================================================

class MemoryScorer:
    """
    Implements SLM promotion/demotion scoring algorithms.
    
    Promotion Score (Token → Working):
    PromotionScore = (QueryRelevance * 0.6) + (AccessFrequency * 0.3) + (RecencyScore * 0.1)
    
    Demotion Score (Working → Token):
    DemotionScore = (1 - QueryRelevance) * 0.5 + (1 - AccessFrequency) * 0.3 + (Age/MAX_AGE) * 0.2
    """
    
    MAX_AGE_MINUTES = 60  # Max age for working memory
    
    @staticmethod
    def calculate_recency_score(last_accessed: datetime) -> float:
        """Calculate recency score (1.0 = just accessed, 0.0 = very old)"""
        age_minutes = (datetime.now() - last_accessed).total_seconds() / 60
        return max(0.0, 1.0 - (age_minutes / MemoryScorer.MAX_AGE_MINUTES))
    
    @staticmethod
    def calculate_access_frequency(access_count: int, age_hours: float) -> float:
        """Calculate normalized access frequency"""
        if age_hours < 0.1:
            age_hours = 0.1
        frequency = access_count / age_hours
        # Normalize to 0-1 range (assume max 10 accesses/hour is very high)
        return min(1.0, frequency / 10.0)
    
    @staticmethod
    def calculate_promotion_score(
        query_relevance: float,
        access_count: int,
        created_at: datetime,
        last_accessed: datetime
    ) -> float:
        """
        Calculate promotion score for Token → Working memory.
        Returns score between 0 and 1.
        """
        age_hours = (datetime.now() - created_at).total_seconds() / 3600
        
        access_frequency = MemoryScorer.calculate_access_frequency(access_count, age_hours)
        recency_score = MemoryScorer.calculate_recency_score(last_accessed)
        
        promotion_score = (
            query_relevance * 0.6 +
            access_frequency * 0.3 +
            recency_score * 0.1
        )
        return min(1.0, max(0.0, promotion_score))
    
    @staticmethod
    def calculate_demotion_score(
        query_relevance: float,
        access_count: int,
        created_at: datetime,
        age_in_working: float  # Minutes
    ) -> float:
        """
        Calculate demotion score for Working → Token memory.
        Returns score between 0 and 1.
        """
        age_hours = (datetime.now() - created_at).total_seconds() / 3600
        access_frequency = MemoryScorer.calculate_access_frequency(access_count, age_hours)
        age_ratio = min(1.0, age_in_working / MemoryScorer.MAX_AGE_MINUTES)
        
        demotion_score = (
            (1 - query_relevance) * 0.5 +
            (1 - access_frequency) * 0.3 +
            age_ratio * 0.2
        )
        return min(1.0, max(0.0, demotion_score))
    
    @staticmethod
    def calculate_priority_decay(
        current_priority: float,
        minutes_elapsed: float
    ) -> float:
        """Apply priority decay over time"""
        decay = SLMConfig.WORKING_MEMORY_DECAY_RATE ** minutes_elapsed
        return current_priority * decay
    
    @staticmethod
    def calculate_memory_utility(
        relevance: float,
        importance: float,
        access_count: int,
        recency: float
    ) -> float:
        """
        Calculate overall memory utility for retrieval ranking.
        Higher = more useful to retrieve.
        """
        # Weighted combination
        utility = (
            relevance * 0.4 +
            importance * 0.3 +
            min(1.0, access_count / 10) * 0.2 +
            recency * 0.1
        )
        return utility


# =============================================================================
# NEURAL LINK MANAGER
# =============================================================================

class NeuralLinkManager:
    """
    Manages neural links between memories with SLM features:
    - 8 link types with different properties
    - Link strength evolution and decay
    - Path finding with strength decay
    - Automatic pruning
    """
    
    def __init__(self):
        self.links: Dict[str, NeuralLink] = {}
    
    def _link_key(self, source_id: str, target_id: str, link_type: LinkType) -> str:
        return f"{source_id}::{target_id}::{link_type.value}"
    
    def calculate_link_score(
        self,
        vector_similarity: float,
        co_occurrence: float,
        domain_relatedness: float
    ) -> float:
        """
        Calculate link creation score.
        LinkScore = (VectorSimilarity * 0.6) + (CoOccurrence * 0.25) + (DomainRelatedness * 0.15)
        """
        return (
            vector_similarity * 0.6 +
            co_occurrence * 0.25 +
            domain_relatedness * 0.15
        )
    
    def should_create_link(
        self,
        link_type: LinkType,
        vector_similarity: float,
        co_occurrence: float = 0.5,
        domain_relatedness: float = 0.5
    ) -> Tuple[bool, float]:
        """Check if a link should be created and return the score"""
        config = LINK_CONFIGS[link_type]
        score = self.calculate_link_score(vector_similarity, co_occurrence, domain_relatedness)
        return score >= config.creation_threshold, score
    
    def create_link(
        self,
        source_id: str,
        target_id: str,
        link_type: LinkType,
        initial_strength: float = None
    ) -> NeuralLink:
        """Create a new neural link"""
        config = LINK_CONFIGS[link_type]
        strength = initial_strength if initial_strength else config.default_strength
        
        link = NeuralLink(
            source_id=source_id,
            target_id=target_id,
            link_type=link_type,
            strength=strength
        )
        
        key = self._link_key(source_id, target_id, link_type)
        self.links[key] = link
        return link
    
    def traverse_link(self, source_id: str, target_id: str, link_type: LinkType) -> Optional[NeuralLink]:
        """Traverse a link, updating strength and tracking"""
        key = self._link_key(source_id, target_id, link_type)
        link = self.links.get(key)
        
        if link:
            link.last_traversed = datetime.now()
            link.traversal_count += 1
            # Boost strength on traversal (max daily boost = 0.05)
            usage_boost = min(0.05, 0.01 * link.traversal_count)
            link.strength = min(1.0, link.strength + usage_boost)
        
        return link
    
    def apply_decay(self):
        """Apply daily decay to all links"""
        for link in self.links.values():
            config = LINK_CONFIGS[link.link_type]
            days_since_traversal = (datetime.now() - link.last_traversed).days
            
            # NewStrength = CurrentStrength * (1 - DecayRate)^days
            decay_factor = (1 - config.decay_rate) ** days_since_traversal
            link.strength *= decay_factor
    
    def get_prune_candidates(self) -> List[NeuralLink]:
        """Get links that should be pruned (strength < 0.3, unused 60+ days)"""
        candidates = []
        cutoff = datetime.now() - timedelta(days=SLMConfig.LINK_PRUNE_AGE_DAYS)
        
        for link in self.links.values():
            if link.strength < SLMConfig.LINK_PRUNE_THRESHOLD and link.last_traversed < cutoff:
                candidates.append(link)
        
        return candidates
    
    def find_paths(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = None
    ) -> List[List[NeuralLink]]:
        """
        Find paths between two memories.
        Path strength = product of link strengths * decay per hop
        """
        max_depth = max_depth or SLMConfig.MAX_PATH_DEPTH
        paths = []
        
        def dfs(current: str, target: str, path: List[NeuralLink], depth: int):
            if depth > max_depth:
                return
            if current == target:
                paths.append(path.copy())
                return
            
            for link in self.links.values():
                if link.source_id == current and link not in path:
                    path.append(link)
                    dfs(link.target_id, target, path, depth + 1)
                    path.pop()
        
        dfs(source_id, target_id, [], 0)
        return paths
    
    def calculate_path_strength(self, path: List[NeuralLink]) -> float:
        """Calculate combined path strength with decay per hop"""
        if not path:
            return 0.0
        
        strength = 1.0
        for i, link in enumerate(path):
            hop_decay = SLMConfig.PATH_STRENGTH_DECAY ** i
            strength *= link.strength * hop_decay
        
        return strength
    
    def get_links_for_memory(self, memory_id: str) -> List[NeuralLink]:
        """Get all links connected to a memory"""
        return [
            link for link in self.links.values()
            if link.source_id == memory_id or link.target_id == memory_id
        ]
    
    def get_stats(self) -> Dict:
        """Get link statistics"""
        by_type = {}
        for link_type in LinkType:
            by_type[link_type.value] = sum(
                1 for link in self.links.values() 
                if link.link_type == link_type
            )
        
        return {
            "total_links": len(self.links),
            "by_type": by_type,
            "avg_strength": sum(l.strength for l in self.links.values()) / max(1, len(self.links))
        }


# =============================================================================
# SLM ENHANCED MEMORY MANAGER
# =============================================================================

class SLMMemoryManager:
    """
    Enhanced Memory Manager implementing SLM architecture features.
    
    Integrates with existing Mnemo server while adding:
    - Three-tier memory hierarchy
    - Promotion/demotion with scoring
    - Neural link management
    - Folder organization
    - Memory decay and cleanup
    """
    
    def __init__(
        self,
        mnemo_url: str = "https://athelaperk-mnemo-mcp.hf.space",
        hf_key: str = None,
        openrouter_key: str = None
    ):
        self.mnemo_url = mnemo_url.rstrip('/')
        self.hf_key = hf_key
        self.openrouter_key = openrouter_key
        self.headers = {
            "Authorization": f"Bearer {hf_key}",
            "Content-Type": "application/json"
        }
        
        # Local state
        self.working_memory: Dict[str, MemoryItem] = {}
        self.token_memory: Dict[str, MemoryItem] = {}
        self.link_manager = NeuralLinkManager()
        self.folders: Dict[str, MemoryFolder] = {
            "/": MemoryFolder(path="/", name="Root", description="Root folder")
        }
        
        # Initialize default folders by category
        for category in MemoryCategory:
            folder_path = f"/{category.value}"
            self.folders[folder_path] = MemoryFolder(
                path=folder_path,
                name=category.value.title(),
                description=f"Memories related to {category.value}"
            )
    
    # =========================================================================
    # MNEMO INTEGRATION
    # =========================================================================
    
    def _mnemo_add(self, content: str, metadata: dict = None) -> Optional[str]:
        """Add memory to Mnemo server (semantic tier)"""
        try:
            response = requests.post(
                f"{self.mnemo_url}/add",
                headers=self.headers,
                json={"content": content, "metadata": metadata or {}},
                timeout=10
            )
            if response.status_code == 200:
                return response.json().get("memory_id")
            return None
        except Exception:
            return None
    
    def _mnemo_search(self, query: str, limit: int = 5) -> List[Dict]:
        """Search Mnemo server"""
        try:
            response = requests.post(
                f"{self.mnemo_url}/search",
                headers=self.headers,
                json={"query": query, "limit": limit},
                timeout=10
            )
            if response.status_code == 200:
                return response.json().get("results", [])
            return []
        except Exception:
            return []
    
    def _mnemo_get_context(self, query: str) -> str:
        """Get formatted context from Mnemo"""
        try:
            response = requests.post(
                f"{self.mnemo_url}/get_context",
                headers=self.headers,
                json={"query": query},
                timeout=10
            )
            if response.status_code == 200:
                return response.json().get("context", "")
            return ""
        except Exception:
            return ""
    
    def _mnemo_list(self) -> List[Dict]:
        """List all Mnemo memories"""
        try:
            response = requests.get(
                f"{self.mnemo_url}/list",
                headers=self.headers,
                timeout=10
            )
            if response.status_code == 200:
                return response.json().get("memories", [])
            return []
        except Exception:
            return []
    
    def _mnemo_delete(self, memory_id: str) -> bool:
        """Delete from Mnemo"""
        try:
            response = requests.delete(
                f"{self.mnemo_url}/delete/{memory_id}",
                headers=self.headers,
                timeout=10
            )
            return response.status_code == 200
        except Exception:
            return False
    
    # =========================================================================
    # MEMORY OPERATIONS
    # =========================================================================
    
    def add_memory(
        self,
        content: str,
        category: MemoryCategory = MemoryCategory.GENERAL,
        folder: str = None,
        importance: float = 0.5,
        source: str = "auto"
    ) -> Optional[MemoryItem]:
        """
        Add a new memory to the system.
        Stores in semantic tier (Mnemo) and creates local tracking.
        """
        # Default folder based on category
        if folder is None:
            folder = f"/{category.value}"
        
        # Store in Mnemo (semantic tier)
        metadata = {
            "category": category.value,
            "folder": folder,
            "importance": importance,
            "source": source
        }
        
        memory_id = self._mnemo_add(f"[{category.value.upper()}] {content}", metadata)
        
        if not memory_id:
            return None
        
        # Create local memory item
        memory = MemoryItem(
            id=memory_id,
            content=content,
            tier=MemoryTier.SEMANTIC,
            category=category,
            folder=folder,
            importance=importance,
            source=source
        )
        
        # Update folder count
        if folder in self.folders:
            self.folders[folder].memory_count += 1
        
        return memory
    
    def promote_to_working(self, memory: MemoryItem) -> bool:
        """Promote memory from token/semantic to working memory"""
        if len(self.working_memory) >= SLMConfig.WORKING_MEMORY_MAX_ITEMS:
            # Need to demote something first
            self._demote_lowest_priority()
        
        memory.tier = MemoryTier.WORKING
        memory.priority = 1.0  # Fresh in working memory
        memory.last_accessed = datetime.now()
        memory.access_count += 1
        
        self.working_memory[memory.id] = memory
        
        # Remove from token memory if present
        if memory.id in self.token_memory:
            del self.token_memory[memory.id]
        
        return True
    
    def demote_to_token(self, memory: MemoryItem) -> bool:
        """Demote memory from working to token memory"""
        memory.tier = MemoryTier.TOKEN
        
        self.token_memory[memory.id] = memory
        
        if memory.id in self.working_memory:
            del self.working_memory[memory.id]
        
        return True
    
    def _demote_lowest_priority(self):
        """Demote the lowest priority item from working memory"""
        if not self.working_memory:
            return
        
        # Find lowest priority
        lowest = min(self.working_memory.values(), key=lambda m: m.priority)
        self.demote_to_token(lowest)
    
    def apply_decay_cycle(self):
        """Apply decay to all working memory items"""
        items_to_demote = []
        
        for memory in self.working_memory.values():
            minutes_since_access = (datetime.now() - memory.last_accessed).total_seconds() / 60
            memory.priority = MemoryScorer.calculate_priority_decay(
                memory.priority, 
                minutes_since_access
            )
            
            if memory.priority < SLMConfig.PRIORITY_THRESHOLD:
                items_to_demote.append(memory)
        
        for memory in items_to_demote:
            self.demote_to_token(memory)
    
    # =========================================================================
    # RETRIEVAL WITH SLM SCORING
    # =========================================================================
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        include_working: bool = True,
        include_token: bool = True,
        include_semantic: bool = True
    ) -> List[Tuple[MemoryItem, float]]:
        """
        Retrieve memories with SLM utility scoring.
        Returns list of (memory, utility_score) tuples.
        """
        results = []
        
        # Get from Mnemo (semantic tier)
        if include_semantic:
            semantic_results = self._mnemo_search(query, limit=top_k * 2)
            for result in semantic_results:
                # Create memory item from Mnemo result
                memory = MemoryItem(
                    id=result.get("id", ""),
                    content=result.get("content", ""),
                    tier=MemoryTier.SEMANTIC,
                    access_count=result.get("access_count", 0)
                )
                
                relevance = result.get("score", 0.5)
                utility = MemoryScorer.calculate_memory_utility(
                    relevance=relevance,
                    importance=memory.importance,
                    access_count=memory.access_count,
                    recency=0.5  # Default for semantic tier
                )
                
                results.append((memory, utility))
        
        # Add working memory items (highest priority)
        if include_working:
            for memory in self.working_memory.values():
                recency = MemoryScorer.calculate_recency_score(memory.last_accessed)
                utility = MemoryScorer.calculate_memory_utility(
                    relevance=memory.relevance_score,
                    importance=memory.importance,
                    access_count=memory.access_count,
                    recency=recency
                )
                # Boost working memory items
                utility = min(1.0, utility * 1.2)
                results.append((memory, utility))
        
        # Add token memory items
        if include_token:
            for memory in self.token_memory.values():
                recency = MemoryScorer.calculate_recency_score(memory.last_accessed)
                utility = MemoryScorer.calculate_memory_utility(
                    relevance=memory.relevance_score,
                    importance=memory.importance,
                    access_count=memory.access_count,
                    recency=recency
                )
                results.append((memory, utility))
        
        # Sort by utility and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def get_context(self, query: str) -> str:
        """Get formatted context for LLM, combining all tiers"""
        parts = []
        
        # Working memory (most relevant)
        if self.working_memory:
            working_items = list(self.working_memory.values())[:5]
            if working_items:
                parts.append("[ACTIVE CONTEXT]")
                for mem in working_items:
                    parts.append(f"• {mem.content}")
        
        # Get from Mnemo
        mnemo_context = self._mnemo_get_context(query)
        if mnemo_context:
            parts.append(mnemo_context)
        
        return "\n".join(parts)
    
    # =========================================================================
    # FOLDER MANAGEMENT
    # =========================================================================
    
    def create_folder(
        self,
        name: str,
        parent: str = "/",
        description: str = "",
        color: str = "#808080"
    ) -> MemoryFolder:
        """Create a new folder"""
        path = f"{parent.rstrip('/')}/{name}"
        
        folder = MemoryFolder(
            path=path,
            name=name,
            description=description,
            color=color,
            parent=parent
        )
        
        self.folders[path] = folder
        return folder
    
    def move_memory_to_folder(self, memory_id: str, folder_path: str) -> bool:
        """Move a memory to a different folder"""
        # Update in working memory
        if memory_id in self.working_memory:
            old_folder = self.working_memory[memory_id].folder
            self.working_memory[memory_id].folder = folder_path
            
            # Update folder counts
            if old_folder in self.folders:
                self.folders[old_folder].memory_count -= 1
            if folder_path in self.folders:
                self.folders[folder_path].memory_count += 1
            
            return True
        
        # Update in token memory
        if memory_id in self.token_memory:
            old_folder = self.token_memory[memory_id].folder
            self.token_memory[memory_id].folder = folder_path
            
            if old_folder in self.folders:
                self.folders[old_folder].memory_count -= 1
            if folder_path in self.folders:
                self.folders[folder_path].memory_count += 1
            
            return True
        
        return False
    
    def get_memories_in_folder(self, folder_path: str) -> List[MemoryItem]:
        """Get all memories in a folder"""
        memories = []
        
        for memory in self.working_memory.values():
            if memory.folder == folder_path:
                memories.append(memory)
        
        for memory in self.token_memory.values():
            if memory.folder == folder_path:
                memories.append(memory)
        
        return memories
    
    def list_folders(self) -> List[MemoryFolder]:
        """List all folders"""
        return list(self.folders.values())
    
    # =========================================================================
    # NEURAL LINKS
    # =========================================================================
    
    def create_link(
        self,
        source_id: str,
        target_id: str,
        link_type: LinkType,
        strength: float = None
    ) -> Optional[NeuralLink]:
        """Create a neural link between memories"""
        return self.link_manager.create_link(source_id, target_id, link_type, strength)
    
    def find_related(self, memory_id: str, max_depth: int = 2) -> List[Tuple[str, float]]:
        """Find related memories through neural links"""
        related = []
        visited = set([memory_id])
        
        def explore(current_id: str, depth: int, cumulative_strength: float):
            if depth > max_depth:
                return
            
            links = self.link_manager.get_links_for_memory(current_id)
            for link in links:
                target = link.target_id if link.source_id == current_id else link.source_id
                
                if target not in visited:
                    visited.add(target)
                    path_strength = cumulative_strength * link.strength * SLMConfig.PATH_STRENGTH_DECAY
                    related.append((target, path_strength))
                    explore(target, depth + 1, path_strength)
        
        explore(memory_id, 0, 1.0)
        related.sort(key=lambda x: x[1], reverse=True)
        return related
    
    # =========================================================================
    # STATS & MAINTENANCE
    # =========================================================================
    
    def get_stats(self) -> Dict:
        """Get comprehensive system statistics"""
        mnemo_memories = self._mnemo_list()
        
        return {
            "tiers": {
                "working": len(self.working_memory),
                "token": len(self.token_memory),
                "semantic": len(mnemo_memories)
            },
            "folders": {
                "count": len(self.folders),
                "by_path": {f.path: f.memory_count for f in self.folders.values()}
            },
            "links": self.link_manager.get_stats(),
            "config": {
                "working_memory_max": SLMConfig.WORKING_MEMORY_MAX_ITEMS,
                "promotion_threshold": SLMConfig.PROMOTION_THRESHOLD,
                "demotion_threshold": SLMConfig.DEMOTION_THRESHOLD
            }
        }
    
    def run_maintenance(self):
        """Run maintenance tasks: decay, pruning, cleanup"""
        # Apply decay to working memory
        self.apply_decay_cycle()
        
        # Apply decay to neural links
        self.link_manager.apply_decay()
        
        # Get prune candidates
        prune_candidates = self.link_manager.get_prune_candidates()
        
        return {
            "working_memory_after_decay": len(self.working_memory),
            "demoted_to_token": len(self.token_memory),
            "links_prunable": len(prune_candidates)
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_slm_manager(
    mnemo_url: str = "https://athelaperk-mnemo-mcp.hf.space",
    hf_key: str = None,
    openrouter_key: str = None
) -> SLMMemoryManager:
    """Factory function to create SLM Memory Manager"""
    return SLMMemoryManager(
        mnemo_url=mnemo_url,
        hf_key=hf_key,
        openrouter_key=openrouter_key
    )


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SLM ENHANCED MEMORY SYSTEM - TEST")
    print("=" * 60)
    
    # Test scoring
    print("\n--- Testing Memory Scorer ---")
    
    # Promotion score
    promo_score = MemoryScorer.calculate_promotion_score(
        query_relevance=0.8,
        access_count=5,
        created_at=datetime.now() - timedelta(hours=2),
        last_accessed=datetime.now() - timedelta(minutes=5)
    )
    print(f"Promotion score (high relevance, recent): {promo_score:.3f}")
    print(f"Should promote: {promo_score >= SLMConfig.PROMOTION_THRESHOLD}")
    
    # Demotion score
    demo_score = MemoryScorer.calculate_demotion_score(
        query_relevance=0.2,
        access_count=1,
        created_at=datetime.now() - timedelta(hours=24),
        age_in_working=45  # minutes
    )
    print(f"Demotion score (low relevance, old): {demo_score:.3f}")
    print(f"Should demote: {demo_score >= SLMConfig.DEMOTION_THRESHOLD}")
    
    # Test link manager
    print("\n--- Testing Neural Links ---")
    link_mgr = NeuralLinkManager()
    
    # Check if should create link
    should_create, score = link_mgr.should_create_link(
        LinkType.SEMANTIC_SIMILARITY,
        vector_similarity=0.75,
        co_occurrence=0.6
    )
    print(f"Link score: {score:.3f}, Should create: {should_create}")
    
    # Create some links
    link1 = link_mgr.create_link("mem_001", "mem_002", LinkType.SEMANTIC_SIMILARITY)
    link2 = link_mgr.create_link("mem_002", "mem_003", LinkType.HIERARCHICAL)
    print(f"Created {len(link_mgr.links)} links")
    print(f"Link stats: {link_mgr.get_stats()}")
    
    print("\n--- Testing SLM Manager ---")
    manager = SLMMemoryManager(
        hf_key=os.environ.get("HF_KEY", "")
    )
    
    print(f"Folders: {[f.name for f in manager.list_folders()]}")
    print(f"Stats: {manager.get_stats()}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE!")
    print("=" * 60)

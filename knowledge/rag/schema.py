"""
NEXUS Memory Layer — RAG Schema Design
Defines the data structures for all five memory types:
  Working → Short-term → Episodic → Semantic → Procedural
"""
from __future__ import annotations

import uuid
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ─── Memory Type Taxonomy ────────────────────────────────────────────────────

class MemoryType(str, Enum):
    WORKING    = "working"     # In-process context (not persisted to DB)
    SHORT_TERM = "short_term"  # Redis TTL 48h — session logs
    EPISODIC   = "episodic"    # Task execution records (vector + metadata)
    SEMANTIC   = "semantic"    # Knowledge facts, documents, summaries
    PROCEDURAL = "procedural"  # Skill definitions, how-to knowledge


class DocumentType(str, Enum):
    DOCUMENT      = "document"       # Raw uploaded file (PDF, DOCX…)
    SUMMARY       = "summary"        # Agent-generated summary
    TASK_RECORD   = "task_record"    # Episodic: what happened, outcome
    FACT          = "fact"           # Atomic knowledge unit
    CONVERSATION  = "conversation"   # Chat history chunk
    SKILL         = "skill"          # SKILL.md content
    CODE_SNIPPET  = "code_snippet"   # Reusable code fragment
    DECISION      = "decision"       # Recorded decision + rationale


class PrivacyTier(str, Enum):
    PRIVATE  = "PRIVATE"
    INTERNAL = "INTERNAL"
    PUBLIC   = "PUBLIC"


# ─── Core Memory Record ──────────────────────────────────────────────────────

@dataclass
class MemoryRecord:
    """
    Universal record stored in the vector database.
    The 'content' field is embedded; everything else is filterable metadata.
    """
    # Identity
    doc_id:         str = field(default_factory=lambda: str(uuid.uuid4()))
    memory_type:    MemoryType = MemoryType.SEMANTIC
    doc_type:       DocumentType = DocumentType.FACT

    # Content (what gets embedded)
    content:        str = ""
    title:          Optional[str] = None

    # Provenance
    source:         str = ""        # file path, URL, "agent:rag_agent", etc.
    agent_id:       Optional[str] = None
    task_id:        Optional[str] = None
    session_id:     Optional[str] = None

    # Classification
    domain:         str = "general"    # research, engineering, operations…
    tags:           list[str] = field(default_factory=list)
    language:       str = "zh-TW"

    # Access control
    privacy_tier:   PrivacyTier = PrivacyTier.INTERNAL
    owner_id:       Optional[str] = None    # user or agent that owns this record
    readable_by:    list[str] = field(default_factory=list)   # agent_ids

    # Quality signals
    quality_score:  float = 0.5     # 0.0–1.0, updated by Critic Agent
    confidence:     float = 0.5     # model confidence when creating this
    access_count:   int = 0
    decay_factor:   float = 1.0     # multiplied by relevance score; decays over time

    # Temporal
    created_at:     float = field(default_factory=time.time)
    updated_at:     float = field(default_factory=time.time)
    expires_at:     Optional[float] = None    # None = permanent

    # Relationships (stored as doc_ids)
    parent_id:      Optional[str] = None
    related_ids:    list[str] = field(default_factory=list)
    chunk_index:    Optional[int] = None     # if this is a chunk of a larger doc
    total_chunks:   Optional[int] = None

    # LLM metadata
    embedding_model: str = "bge-m3"          # model used to embed
    llm_created_by:  Optional[str] = None    # which LLM generated this content

    def to_metadata_dict(self) -> dict[str, Any]:
        """Serialize to flat dict for ChromaDB / Qdrant metadata."""
        return {
            "doc_id":          self.doc_id,
            "memory_type":     self.memory_type.value,
            "doc_type":        self.doc_type.value,
            "title":           self.title or "",
            "source":          self.source,
            "agent_id":        self.agent_id or "",
            "task_id":         self.task_id or "",
            "session_id":      self.session_id or "",
            "domain":          self.domain,
            "tags":            ",".join(self.tags),   # flat string for metadata filters
            "language":        self.language,
            "privacy_tier":    self.privacy_tier.value,
            "owner_id":        self.owner_id or "",
            "quality_score":   self.quality_score,
            "confidence":      self.confidence,
            "access_count":    self.access_count,
            "decay_factor":    self.decay_factor,
            "created_at":      self.created_at,
            "updated_at":      self.updated_at,
            "expires_at":      self.expires_at or 0.0,
            "parent_id":       self.parent_id or "",
            "chunk_index":     self.chunk_index if self.chunk_index is not None else -1,
            "total_chunks":    self.total_chunks if self.total_chunks is not None else -1,
            "embedding_model": self.embedding_model,
            "llm_created_by":  self.llm_created_by or "",
        }


# ─── Specialized Record Types ─────────────────────────────────────────────────

@dataclass
class EpisodicRecord(MemoryRecord):
    """Stores the full execution trace of a completed task."""
    memory_type:      MemoryType = MemoryType.EPISODIC
    doc_type:         DocumentType = DocumentType.TASK_RECORD

    # Task execution details
    agent_chain:      list[str] = field(default_factory=list)  # agents used in order
    workflow_pattern: str = "sequential"    # sequential|parallel|feedback_loop
    llm_used:         str = ""
    duration_ms:      int = 0
    success:          bool = True
    error_message:    Optional[str] = None
    cost_usd:         float = 0.0
    tokens_used:      int = 0
    retry_count:      int = 0

    def to_metadata_dict(self) -> dict[str, Any]:
        base = super().to_metadata_dict()
        base.update({
            "agent_chain":      ",".join(self.agent_chain),
            "workflow_pattern": self.workflow_pattern,
            "llm_used":         self.llm_used,
            "duration_ms":      self.duration_ms,
            "success":          int(self.success),
            "cost_usd":         self.cost_usd,
            "tokens_used":      self.tokens_used,
            "retry_count":      self.retry_count,
        })
        return base


@dataclass
class AgentPersonaRecord(MemoryRecord):
    """
    Versioned system prompt for an agent.
    Used for prompt management and A/B optimization.
    """
    memory_type:       MemoryType = MemoryType.PROCEDURAL
    doc_type:          DocumentType = DocumentType.SKILL

    agent_name:        str = ""
    prompt_version:    str = "1.0.0"
    avg_quality_score: float = 0.0      # average Critic score across tasks using this version
    task_count:        int = 0
    is_active:         bool = True


# ─── Collection Definitions ──────────────────────────────────────────────────

COLLECTION_CONFIGS: dict[str, dict] = {

    "knowledge_base": {
        # Primary semantic knowledge store
        "description": "Domain knowledge, documents, facts, summaries",
        "memory_types": [MemoryType.SEMANTIC],
        "embedding_model": "bge-m3",          # local, 768-dim, multilingual
        "embedding_dim": 768,
        "distance_metric": "cosine",
        "hnsw_config": {
            "m": 16,              # connections per layer (higher = better recall, more RAM)
            "ef_construction": 200,
        },
        "index_payload_fields": [           # fields to index for fast filtering
            "domain", "doc_type", "privacy_tier", "language",
            "quality_score", "created_at", "agent_id",
        ],
    },

    "episodic_memory": {
        # Task execution history for agent self-reflection
        "description": "What agents did, how long, outcome, cost",
        "memory_types": [MemoryType.EPISODIC],
        "embedding_model": "bge-m3",
        "embedding_dim": 768,
        "distance_metric": "cosine",
        "retention_days": 90,    # Auto-expire after 90 days
        "index_payload_fields": [
            "agent_chain", "success", "llm_used",
            "domain", "duration_ms", "cost_usd",
        ],
    },

    "agent_personas": {
        # System prompt versioning + performance tracking
        "description": "Agent identity files with quality tracking",
        "memory_types": [MemoryType.PROCEDURAL],
        "embedding_model": "bge-m3",
        "embedding_dim": 768,
        "distance_metric": "cosine",
        "index_payload_fields": [
            "agent_name", "prompt_version", "is_active",
            "avg_quality_score", "task_count",
        ],
    },

    "conversation_history": {
        # Multi-session conversation memory
        "description": "Chat history chunks for context retrieval",
        "memory_types": [MemoryType.SHORT_TERM, MemoryType.EPISODIC],
        "embedding_model": "bge-m3",
        "embedding_dim": 768,
        "distance_metric": "cosine",
        "retention_days": 30,
        "index_payload_fields": [
            "session_id", "agent_id", "created_at", "language",
        ],
    },
}


# ─── Chunking Strategy ──────────────────────────────────────────────────────

CHUNKING_STRATEGIES: dict[str, dict] = {

    "document": {
        # Long documents: semantic chunking with overlap
        "chunk_size": 1000,      # characters
        "chunk_overlap": 200,
        "strategy": "recursive_text",
        "separators": ["\n\n", "\n", "。", ".", " "],
    },

    "conversation": {
        # Conversations: preserve turn boundaries
        "chunk_size": 500,
        "chunk_overlap": 100,
        "strategy": "by_turn",
    },

    "code_snippet": {
        # Code: preserve function/class boundaries
        "chunk_size": 2000,
        "chunk_overlap": 100,
        "strategy": "by_function",
    },

    "task_record": {
        # Task records: store as single unit (no chunking)
        "chunk_size": 999_999,
        "chunk_overlap": 0,
        "strategy": "none",
    },
}


# ─── Retrieval Modes ────────────────────────────────────────────────────────

class RetrievalMode(str, Enum):
    SEMANTIC  = "semantic"    # Pure vector similarity
    KEYWORD   = "keyword"     # BM25 keyword search
    HYBRID    = "hybrid"      # Semantic + keyword (recommended default)
    GRAPH     = "graph"       # Knowledge graph traversal
    TEMPORAL  = "temporal"    # Recency-weighted retrieval


@dataclass
class RetrievalConfig:
    """Configure how memory is retrieved for a given context."""
    mode:           RetrievalMode = RetrievalMode.HYBRID
    top_k:          int = 5
    score_threshold: float = 0.6
    filters:        dict[str, Any] = field(default_factory=dict)
    # Hybrid weights
    semantic_weight: float = 0.7
    keyword_weight:  float = 0.3
    # Temporal decay: relevance *= exp(-decay_rate * days_old)
    apply_temporal_decay: bool = True
    temporal_decay_rate:  float = 0.01   # per day


# ─── Knowledge Graph Schema ──────────────────────────────────────────────────

GRAPH_RELATIONSHIP_TYPES = {
    "DERIVED_FROM":    "Content was derived from this source",
    "RELATES_TO":      "Semantically related content",
    "CONTRADICTS":     "This content contradicts the target",
    "SUPPORTS":        "This content supports/validates the target",
    "SUMMARIZES":      "This is a summary of the target",
    "MENTIONS":        "This content mentions this entity",
    "USED_BY_AGENT":   "This was retrieved/used by this agent",
    "CREATED_DURING":  "Created during this task execution",
}

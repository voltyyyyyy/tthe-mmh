"""Provider-neutral implementation of the Meta-Memory Harness paper."""

from .embeddings import EmbeddingConfig, EmbeddingEngine, EmbeddingError, build_embedding_engine
from .engine import MetaMemoryEngine, deterministic_embedding
from .store import SQLiteStore
from .types import (
    MMHConfig,
    Patch,
    PatchOperation,
    PatchStatus,
    Precedent,
    Rule,
    RuleTier,
    ValidationEvidence,
)

__all__ = [
    "MMHConfig",
    "MetaMemoryEngine",
    "Patch",
    "PatchOperation",
    "PatchStatus",
    "Precedent",
    "Rule",
    "RuleTier",
    "SQLiteStore",
    "ValidationEvidence",
    "deterministic_embedding",
    "EmbeddingConfig",
    "EmbeddingEngine",
    "EmbeddingError",
    "build_embedding_engine",
]

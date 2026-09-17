"""Embedding engines for the Meta-Memory Harness.

The default engine remains dependency-free deterministic hashing so offline
tests and demos work without credentials.  Production users can switch to a
semantic provider:

* ``openai`` -> any OpenAI-compatible ``/v1/embeddings`` endpoint
* ``sentence-transformers`` -> local CPU/GPU model, optional dependency
* ``hash`` -> deterministic fallback used by the offline tests
"""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .engine import deterministic_embedding


class EmbeddingError(RuntimeError):
    """Raised when an embedding provider cannot be constructed or called."""


@dataclass(slots=True)
class EmbeddingConfig:
    provider: str = "hash"                 # hash | openai | sentence-transformers
    model: str = "text-embedding-3-small"  # OpenAI-compatible embedding model
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    dimensions: int | None = None
    request_timeout: float = 60.0
    batch_size: int = 64
    normalize: bool = True
    local_model: str = "BAAI/bge-m3"
    local_device: str | None = None
    local_trust_remote_code: bool = False
    hash_dimensions: int = 64


class EmbeddingEngine:
    """Callable, caching embedding engine accepted by ``MetaMemoryEngine``.

    ``EmbeddingEngine(...)`` can be passed directly as the
    ``embedding_provider`` argument.  It also exposes ``embed`` and
    ``embed_batch`` for callers that want the provider API directly.
    """

    def __init__(self, config: EmbeddingConfig | None = None, *, client: Any | None = None) -> None:
        self.config = config or EmbeddingConfig()
        self._client = client
        self._lock = threading.RLock()
        self._cache: dict[str, list[float]] = {}
        self._known_dimensions: int | None = self.config.dimensions

    @classmethod
    def from_env(cls) -> "EmbeddingEngine":
        provider = os.environ.get("MMH_EMBEDDING_PROVIDER", "hash").strip().lower()
        if provider in {"st", "sentence", "sentence_transformers", "sentence-transformers"}:
            provider = "sentence-transformers"
        dimensions_raw = os.environ.get("MMH_EMBEDDING_DIMENSIONS", "").strip()
        dimensions = int(dimensions_raw) if dimensions_raw else None
        return cls(EmbeddingConfig(
            provider=provider,
            model=os.environ.get("MMH_EMBEDDING_MODEL", "text-embedding-3-small"),
            base_url=os.environ.get("MMH_EMBEDDING_BASE_URL", "https://api.openai.com/v1"),
            api_key_env=os.environ.get("MMH_EMBEDDING_API_KEY_ENV", "OPENAI_API_KEY"),
            dimensions=dimensions,
            request_timeout=float(os.environ.get("MMH_EMBEDDING_TIMEOUT", "60")),
            batch_size=int(os.environ.get("MMH_EMBEDDING_BATCH_SIZE", "64")),
            local_model=os.environ.get("MMH_EMBEDDING_LOCAL_MODEL", "BAAI/bge-m3"),
            local_device=os.environ.get("MMH_EMBEDDING_DEVICE") or None,
            local_trust_remote_code=os.environ.get("MMH_EMBEDDING_TRUST_REMOTE_CODE", "").lower() in {"1", "true", "yes"},
        ))

    def __call__(self, text: str) -> list[float]:
        return self.embed(text)

    def embed(self, text: str) -> list[float]:
        if text in self._cache:
            return list(self._cache[text])
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        items = [str(text) for text in texts]
        if not items:
            return []
        with self._lock:
            missing = [text for text in items if text not in self._cache]
        if missing:
            vectors = self._embed_missing(missing)
            with self._lock:
                for text, vector in zip(missing, vectors):
                    self._cache[text] = vector
        with self._lock:
            return [list(self._cache[text]) for text in items]

    def _embed_missing(self, texts: list[str]) -> list[list[float]]:
        provider = self.config.provider.strip().lower()
        if provider in {"st", "sentence", "sentence_transformers", "sentence-transformers"}:
            vectors = self._embed_sentence_transformers(texts)
        elif provider == "openai":
            vectors = self._embed_openai(texts)
        elif provider == "hash":
            vectors = [deterministic_embedding(text, self.config.hash_dimensions) for text in texts]
        else:
            raise EmbeddingError(f"unknown embedding provider: {self.config.provider!r}")
        if len(vectors) != len(texts):
            raise EmbeddingError("embedding provider returned the wrong number of vectors")
        for vector in vectors:
            if not vector or not all(math.isfinite(float(v)) for v in vector):
                raise EmbeddingError("embedding provider returned empty or non-finite vector")
            if not any(float(v) != 0 for v in vector):
                raise EmbeddingError("embedding provider returned zero vector")
            if self._known_dimensions is None:
                self._known_dimensions = len(vector)
            if len(vector) != self._known_dimensions:
                raise EmbeddingError("embedding dimension changed or disagrees with configuration")
        return [self._normalize(vector) for vector in vectors]

    def _normalize(self, vector: Sequence[float]) -> list[float]:
        values = [float(value) for value in vector]
        if not self.config.normalize:
            return values
        norm = math.sqrt(sum(value * value for value in values))
        return [value / norm for value in values] if norm else values

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        client = self._client
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - dependency is in requirements
                raise EmbeddingError("openai is not installed; `pip install openai>=1.0`") from exc
            key = os.environ.get(self.config.api_key_env, "")
            if not key:
                raise EmbeddingError(
                    f"env ${self.config.api_key_env} is empty; cannot use the openai embedding provider"
                )
            client = OpenAI(
                base_url=self.config.base_url,
                api_key=key,
                timeout=self.config.request_timeout,
            )
            self._client = client

        vectors: list[list[float]] = []
        for start in range(0, len(texts), max(1, self.config.batch_size)):
            batch = texts[start:start + max(1, self.config.batch_size)]
            create_kwargs: dict[str, Any] = {"model": self.config.model, "input": batch}
            if self.config.dimensions is not None:
                create_kwargs["dimensions"] = self.config.dimensions
            response = client.embeddings.create(**create_kwargs)
            data = sorted(response.data, key=lambda item: getattr(item, "index", 0))
            if len(data) != len(batch):
                raise EmbeddingError("embedding endpoint returned the wrong number of vectors")
            vectors.extend([list(item.embedding) for item in data])
        return vectors

    def _embed_sentence_transformers(self, texts: list[str]) -> list[list[float]]:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError(
                "sentence-transformers is not installed; `pip install sentence-transformers`"
            ) from exc
        model = getattr(self, "_st_model", None)
        if model is None:
            kwargs: dict[str, Any] = {"trust_remote_code": self.config.local_trust_remote_code}
            if self.config.local_device:
                kwargs["device"] = self.config.local_device
            model = SentenceTransformer(self.config.local_model, **kwargs)
            self._st_model = model
        encoded = model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
        return [list(map(float, row)) for row in encoded]


def build_embedding_engine(config: EmbeddingConfig | None = None) -> EmbeddingEngine:
    """Construct an embedding engine, honoring ``MMH_EMBEDDING_*`` env vars."""
    return EmbeddingEngine(config) if config is not None else EmbeddingEngine.from_env()


__all__ = ["EmbeddingConfig", "EmbeddingEngine", "EmbeddingError", "build_embedding_engine"]

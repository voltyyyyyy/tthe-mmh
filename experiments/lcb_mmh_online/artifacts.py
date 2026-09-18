"""Content-addressed immutable artifact store for the online-MMH experiment."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .types import ArtifactRef, utc_now


class ArtifactError(RuntimeError):
    """Raised when an artifact is missing, mutated, or cannot be written."""


class ArtifactStore:
    """Run-scoped, content-addressed storage for parent and child harnesses.

    Retrieval is by SHA-256, never by generated module name.  This prevents a
    re-run or ``--fresh`` candidate cleanup from silently changing an artifact
    that a persisted application refers to.
    """

    def __init__(self, root: str | Path, run_id: str) -> None:
        self.root = Path(root)
        self.run_id = str(run_id)
        self.objects = self.root / "objects"
        self.objects.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _digest(payload: bytes) -> str:
        return hashlib.sha256(payload).hexdigest()

    def put_bytes(self, payload: bytes, *, media_type: str = "text/x-python") -> ArtifactRef:
        digest = self._digest(payload)
        relative = f"objects/{digest}"
        destination = self.root / relative
        if destination.exists():
            existing = destination.read_bytes()
            if existing != payload:
                raise ArtifactError(f"content-addressed collision for {digest}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".tmp")
            temporary.write_bytes(payload)
            temporary.replace(destination)
        return ArtifactRef(digest, len(payload), relative, media_type)

    def put_text(self, text: str, *, media_type: str = "text/x-python") -> ArtifactRef:
        return self.put_bytes(str(text).encode("utf-8"), media_type=media_type)

    def put_file(self, path: str | Path, *, media_type: str = "text/x-python") -> ArtifactRef:
        source = Path(path)
        if not source.exists():
            raise ArtifactError(f"artifact source does not exist: {source}")
        return self.put_bytes(source.read_bytes(), media_type=media_type)

    def read_bytes(self, ref: ArtifactRef | str | Path) -> bytes:
        if isinstance(ref, ArtifactRef):
            path = self.root / ref.relative_path
            expected = ref.sha256
        else:
            path = self.root / ref
            expected = path.name
        if not path.exists():
            raise ArtifactError(f"artifact missing: {path}")
        payload = path.read_bytes()
        actual = self._digest(payload)
        if actual != expected:
            raise ArtifactError(f"artifact mutated: {path} expected={expected} actual={actual}")
        return payload

    def read_text(self, ref: ArtifactRef | str | Path) -> str:
        return self.read_bytes(ref).decode("utf-8")

    def freeze_text(self, text: str, *, media_type: str = "text/x-python") -> ArtifactRef:
        return self.put_text(text, media_type=media_type)

    def verify(self, ref: ArtifactRef) -> bool:
        try:
            self.read_bytes(ref)
        except ArtifactError:
            return False
        return True

    def manifest(self, refs: dict[str, ArtifactRef]) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "generated_at": utc_now(),
            "artifacts": {name: ref.to_dict() for name, ref in refs.items()},
        }

    def export_json(self, path: str | Path, refs: dict[str, ArtifactRef]) -> None:
        Path(path).write_text(json.dumps(self.manifest(refs), indent=2, sort_keys=True), encoding="utf-8")

    def clone_to(self, destination: str | Path) -> None:
        target = Path(destination)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(self.root, target)

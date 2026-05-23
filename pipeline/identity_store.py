from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .stages.vector_search import IdentitySearchStage


@dataclass(frozen=True)
class IdentityStoreConfig:
    bucket_name: str
    match_threshold: float
    local_root: str = "local_embeddings"
    embedding_model: str = "ms1mv3_arcface"


class LocalFaissIdentityStore:
    """Loads the local embedding snapshot used to build an in-memory FAISS index."""

    def __init__(self, cfg: IdentityStoreConfig):
        self.cfg = cfg

    @property
    def embeddings_path(self) -> Path:
        return self._bucket_dir / f"{self.cfg.embedding_model}_embeddings.npy"

    @property
    def employee_ids_path(self) -> Path:
        return self._bucket_dir / f"{self.cfg.embedding_model}_employee_ids.pkl"

    @property
    def _bucket_dir(self) -> Path:
        return Path(self.cfg.local_root) / self.cfg.bucket_name

    def load(self) -> IdentitySearchStage:
        embeddings, employee_ids = self._load_snapshot()
        if embeddings is None or employee_ids is None:
            return IdentitySearchStage.build_empty(self.cfg.match_threshold)

        return IdentitySearchStage(
            embeddings,
            employee_ids,
            self.cfg.match_threshold,
        )

    def _load_snapshot(self) -> tuple[np.ndarray | None, list[str] | None]:
        if not self.embeddings_path.exists() or not self.employee_ids_path.exists():
            print(
                "Identity embeddings not found, using empty index: "
                f"{self.embeddings_path} / {self.employee_ids_path}"
            )
            return None, None

        try:
            embeddings = np.load(self.embeddings_path).astype(np.float32)
            with open(self.employee_ids_path, "rb") as file:
                employee_ids = [str(item) for item in pickle.load(file)]
        except (OSError, ValueError, pickle.PickleError, EOFError) as exc:
            print(f"Cannot load identity embeddings, using empty index: {exc}")
            return None, None

        if embeddings.ndim != 2:
            print(f"Invalid identity embeddings shape {embeddings.shape}, using empty index")
            return None, None
        if len(embeddings) != len(employee_ids):
            print(
                "Identity embeddings and employee IDs length mismatch, using empty index: "
                f"{len(embeddings)} embeddings vs {len(employee_ids)} IDs"
            )
            return None, None

        return embeddings, employee_ids

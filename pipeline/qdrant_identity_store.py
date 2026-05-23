from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .stages.vector_search import SearchMatch


@dataclass(frozen=True)
class QdrantIdentityStoreConfig:
    url: str
    api_key: str
    collection: str
    bucket_name: str
    match_threshold: float


class QdrantIdentitySearchStage:
    def __init__(self, cfg: QdrantIdentityStoreConfig):
        self.cfg = cfg
        self.client = self._build_client()
        self._ensure_collection()

    def _build_client(self):
        from qdrant_client import QdrantClient

        api_key = self.cfg.api_key or None
        return QdrantClient(url=self.cfg.url, api_key=api_key, timeout=10.0)

    def _ensure_collection(self) -> None:
        from qdrant_client import models

        if self._collection_exists():
            self._validate_collection()
            self._create_payload_indexes()
            return

        self.client.create_collection(
            collection_name=self.cfg.collection,
            vectors_config=models.VectorParams(
                size=512,
                distance=models.Distance.COSINE,
                on_disk=True,
            ),
            hnsw_config=models.HnswConfigDiff(
                m=16,
                ef_construct=128,
                on_disk=True,
            ),
        )
        self._create_payload_indexes()

    def _collection_exists(self) -> bool:
        try:
            return bool(self.client.collection_exists(self.cfg.collection))
        except AttributeError:
            try:
                self.client.get_collection(self.cfg.collection)
                return True
            except Exception:
                return False

    def _validate_collection(self) -> None:
        info = self.client.get_collection(self.cfg.collection)
        vectors_config = getattr(getattr(getattr(info, "config", None), "params", None), "vectors", None)
        if isinstance(vectors_config, dict):
            vectors_config = next(iter(vectors_config.values()), None)
        size = getattr(vectors_config, "size", None)
        distance = getattr(vectors_config, "distance", None)

        if size is not None and int(size) != 512:
            raise ValueError(
                f"Qdrant collection {self.cfg.collection!r} has vector size {size}, "
                "expected 512"
            )
        if distance is not None:
            actual = getattr(distance, "value", str(distance))
            if str(actual).lower() != "cosine":
                raise ValueError(
                    f"Qdrant collection {self.cfg.collection!r} uses distance {actual}, expected Cosine"
                )

    def _create_payload_indexes(self) -> None:
        from qdrant_client import models

        indexes = {
            "bucket_name": models.PayloadSchemaType.KEYWORD,
            "employee_id": models.PayloadSchemaType.KEYWORD,
            "active": models.PayloadSchemaType.BOOL,
            "model_name": models.PayloadSchemaType.KEYWORD,
        }
        for field_name, field_schema in indexes.items():
            try:
                self.client.create_payload_index(
                    collection_name=self.cfg.collection,
                    field_name=field_name,
                    field_schema=field_schema,
                )
            except Exception:
                pass

    def search(self, embedding: np.ndarray, top_k: int = 5) -> SearchMatch | None:
        from qdrant_client import models

        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        result = self.client.query_points(
            collection_name=self.cfg.collection,
            query=vector.tolist(),
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="bucket_name",
                        match=models.MatchValue(value=self.cfg.bucket_name),
                    ),
                    models.FieldCondition(
                        key="active",
                        match=models.MatchValue(value=True),
                    ),
                ],
            ),
            limit=max(1, int(top_k)),
            with_payload=True,
            with_vectors=False,
        )
        points = result.points
        if not points:
            return None

        by_employee: dict[str, list[float]] = {}
        for point in points:
            employee_id = str((point.payload or {}).get("employee_id") or "")
            if not employee_id:
                continue
            by_employee.setdefault(employee_id, []).append(float(point.score))
        if not by_employee:
            return None

        employee_id, employee_scores = max(
            by_employee.items(),
            key=lambda item: (
                len(item[1]),
                max(item[1]),
                float(np.mean(item[1])),
            ),
        )
        score = float(np.mean(employee_scores))
        return SearchMatch(employee_id, score, score >= self.cfg.match_threshold)


def build_qdrant_identity_search(cfg: Any) -> QdrantIdentitySearchStage:
    return QdrantIdentitySearchStage(
        QdrantIdentityStoreConfig(
            url=cfg.qdrant.url,
            api_key=cfg.qdrant.api_key,
            collection=cfg.qdrant.collection,
            bucket_name=cfg.bucket_name,
            match_threshold=cfg.match_threshold,
        )
    )

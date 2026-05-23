from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SearchMatch:
    employee_id: str
    score: float
    accepted: bool


class IdentitySearchStage:
    def __init__(
        self,
        embeddings: np.ndarray,
        employee_ids: list[str],
        match_threshold: float = 0.78,
    ):
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.employee_ids = [str(item) for item in employee_ids]
        if len(self.embeddings) != len(self.employee_ids):
            raise ValueError("embeddings rows must match employee_ids length")
        self.match_threshold = match_threshold
        self._faiss = None
        self._index = None
        self._build()

    @classmethod
    def build_empty(
        cls,
        match_threshold: float = 0.78,
    ) -> "IdentitySearchStage":
        return cls(np.zeros((0, 512), dtype=np.float32), [], match_threshold)

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        denom = np.linalg.norm(values, axis=1, keepdims=True)
        denom[denom == 0] = 1
        return values / denom

    def _build(self) -> None:
        if len(self.embeddings) == 0:
            return
        try:
            import faiss

            self._faiss = faiss
            dim = self.embeddings.shape[1]
            vectors = self._normalize(self.embeddings.copy())
            self._index = faiss.IndexFlatIP(dim)
            self._index.add(vectors)
        except Exception as exc:
            print(f"FAISS unavailable, using NumPy vector search: {exc}")
            self._faiss = None
            self._index = None

    def search(self, embedding: np.ndarray, top_k: int = 8) -> SearchMatch | None:
        if len(self.embeddings) == 0:
            return None
        query = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        top_k = min(top_k, len(self.embeddings))

        if self._index is not None:
            similarities, indices = self._index.search(self._normalize(query.copy()), top_k)
            scores = [float(value) for value in similarities[0]]
            raw_indices = indices[0]
        else:
            vectors = self._normalize(self.embeddings.copy())
            q = self._normalize(query.copy())[0]
            similarities = vectors @ q
            raw_indices = np.argsort(-similarities)[:top_k]
            scores = [float(similarities[idx]) for idx in raw_indices]

        by_employee: dict[str, list[float]] = {}
        for idx, score in zip(raw_indices, scores):
            if int(idx) < 0:
                continue
            try:
                employee_id = self.employee_ids[int(idx)]
            except IndexError:
                continue
            by_employee.setdefault(employee_id, []).append(score)
        if not by_employee:
            return None

        employee_id, employee_scores = max(by_employee.items(), key=lambda item: float(np.mean(item[1])))
        score = float(np.mean(employee_scores))
        return SearchMatch(employee_id, score, score >= self.match_threshold)

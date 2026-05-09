from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from database.firebase import get_employee_ids_from_bucket, upsert_employee
from infer.infer_image import transform_image
from infer.utils import device, get_recogn_model, mtcnn


class EmbeddingManager:
    """Local identity store with one Firebase employee id per embedding row."""

    def __init__(self, bucket_name: str, recognition_model_name: str = "ms1mv3_arcface", local_root: str = "local_embeddings"):
        self.bucket_name = bucket_name
        self.recognition_model_name = recognition_model_name
        self.local_dir = Path(local_root) / bucket_name
        self.local_dir.mkdir(parents=True, exist_ok=True)

        self.embeddings_file = self.local_dir / f"{recognition_model_name}_embeddings.npy"
        self.employee_ids_file = self.local_dir / f"{recognition_model_name}_employee_ids.pkl"
        self.legacy_metadata_file = self.local_dir / f"{recognition_model_name}_metadata.pkl"

    def save(self, embeddings: np.ndarray, employee_ids: list[str]) -> None:
        if len(embeddings) != len(employee_ids):
            raise ValueError("embeddings rows must match employee_ids length")
        np.save(self.embeddings_file, embeddings.astype(np.float32))
        with open(self.employee_ids_file, "wb") as file:
            pickle.dump([str(item) for item in employee_ids], file)

    def load(self) -> tuple[np.ndarray | None, list[str] | None]:
        if self.embeddings_file.exists() and self.employee_ids_file.exists():
            embeddings = np.load(self.embeddings_file).astype(np.float32)
            with open(self.employee_ids_file, "rb") as file:
                employee_ids = [str(item) for item in pickle.load(file)]
            if len(embeddings) != len(employee_ids):
                raise ValueError(f"Invalid identity store: {self.embeddings_file} and {self.employee_ids_file} lengths differ")
            return embeddings, employee_ids

        return self._try_migrate_legacy_metadata()

    def _try_migrate_legacy_metadata(self) -> tuple[np.ndarray | None, list[str] | None]:
        if not self.embeddings_file.exists() or not self.legacy_metadata_file.exists():
            return None, None

        embeddings = np.load(self.embeddings_file).astype(np.float32)
        with open(self.legacy_metadata_file, "rb") as file:
            image2class, index2class = pickle.load(file)

        employee_ids = []
        for vector_idx in range(len(embeddings)):
            class_id = image2class.get(vector_idx)
            employee_id = index2class.get(class_id)
            if employee_id is None:
                raise ValueError(f"Legacy metadata has no employee_id for vector index {vector_idx}")
            employee_ids.append(str(employee_id))

        self.save(embeddings, employee_ids)
        return embeddings, employee_ids

    def load_person_ids(self) -> list[str]:
        return get_employee_ids_from_bucket(self.bucket_name)

    def add_employee_from_folder(
        self,
        employee_id: str,
        folder_path: str,
        profile: dict | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        upsert_employee(self.bucket_name, employee_id, profile)

        embeddings, employee_ids = self.load()
        if embeddings is None or employee_ids is None:
            embeddings = np.zeros((0, 512), dtype=np.float32)
            employee_ids = []

        images = []
        for image_path in sorted(Path(folder_path).glob("*")):
            if not image_path.is_file():
                continue
            try:
                images.append(Image.open(image_path).convert("RGB"))
            except Exception as exc:
                print(f"Skipping invalid image {image_path}: {exc}")

        if not images:
            raise ValueError(f"No valid images found in {folder_path}")

        model = get_recogn_model()
        aligned = []
        for image in images:
            face = mtcnn(image)
            aligned.append(transform_image(face if face is not None else image))

        batch = torch.cat(aligned, dim=0).to(device)
        with torch.no_grad():
            new_embeddings = model(batch).detach().cpu().numpy().astype(np.float32)

        embeddings = np.vstack([embeddings, new_embeddings])
        employee_ids.extend([str(employee_id)] * len(new_embeddings))
        self.save(embeddings, employee_ids)
        return embeddings, employee_ids

    def delete_employee(self, employee_id: str) -> tuple[np.ndarray, list[str]]:
        embeddings, employee_ids = self.load()
        if embeddings is None or employee_ids is None:
            raise ValueError("Identity store is empty")

        keep = [idx for idx, current_id in enumerate(employee_ids) if current_id != employee_id]
        embeddings = embeddings[keep]
        employee_ids = [employee_ids[idx] for idx in keep]
        self.save(embeddings, employee_ids)
        return embeddings, employee_ids

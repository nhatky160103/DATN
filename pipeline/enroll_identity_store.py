from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect_images(employee_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(employee_dir.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS
    ]


def enroll_dataset(
    bucket_name: str,
    dataset_root: str,
    triton_url: str = "localhost:8000",
    arcface_model: str = "arcface",
) -> None:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(root)

    import cv2
    import numpy as np

    from pipeline.stages.embedding import FaceEmbeddingStage
    from pipeline.triton_client import TritonInferenceClient

    embedder = FaceEmbeddingStage(TritonInferenceClient(triton_url), arcface_model)
    employee_dirs = [path for path in sorted(root.iterdir()) if path.is_dir()]
    if not employee_dirs:
        raise ValueError(f"No employee folders found in {root}")

    embeddings: list[np.ndarray] = []
    employee_ids: list[str] = []
    for employee_dir in employee_dirs:
        employee_id = employee_dir.name
        images = _collect_images(employee_dir)
        if not images:
            print(f"Skipped {employee_id}: no supported images in {employee_dir}")
            continue

        enrolled = 0
        for image_path in images:
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Skipped unreadable image: {image_path}")
                continue
            embeddings.append(embedder.predict(image))
            employee_ids.append(employee_id)
            enrolled += 1
        print(f"Enrolled {employee_id}: {enrolled} image(s)")

    if not embeddings:
        raise ValueError(f"No embeddings generated from {root}")

    output_dir = Path("local_embeddings") / bucket_name
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "ms1mv3_arcface_embeddings.npy", np.vstack(embeddings).astype(np.float32))
    with open(output_dir / "ms1mv3_arcface_employee_ids.pkl", "wb") as file:
        pickle.dump(employee_ids, file)
    print(f"Saved {len(employee_ids)} embedding(s) to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build local FAISS identity store. Folder names must be Firebase employee IDs."
    )
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--dataset-root", required=True, help="Root folder with subfolders named by Firebase employee_id")
    parser.add_argument("--triton-url", default=os.getenv("TRITON_URL", "localhost:8000"))
    parser.add_argument("--arcface-model", default=os.getenv("TRITON_ARCFACE_MODEL", "arcface"))
    args = parser.parse_args()
    enroll_dataset(args.bucket, args.dataset_root, args.triton_url, args.arcface_model)


if __name__ == "__main__":
    main()

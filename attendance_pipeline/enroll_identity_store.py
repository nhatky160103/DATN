from __future__ import annotations

import argparse
import json
from pathlib import Path


def enroll_dataset(bucket_name: str, dataset_root: str, profile_json: str | None = None) -> None:
    from infer.get_embedding import EmbeddingManager

    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(root)

    profiles = {}
    if profile_json:
        with open(profile_json, "r", encoding="utf-8") as file:
            profiles = json.load(file)

    manager = EmbeddingManager(bucket_name)
    employee_dirs = [path for path in sorted(root.iterdir()) if path.is_dir()]
    if not employee_dirs:
        raise ValueError(f"No employee folders found in {root}")

    for employee_dir in employee_dirs:
        employee_id = employee_dir.name
        profile = profiles.get(employee_id, {})
        manager.add_employee_from_folder(employee_id, str(employee_dir), profile)
        print(f"Enrolled {employee_id} from {employee_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build local FAISS identity store. Folder names must be Firebase employee IDs."
    )
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--dataset-root", required=True, help="Root folder with subfolders named by Firebase employee_id")
    parser.add_argument("--profiles-json", default=None, help="Optional JSON object keyed by employee_id")
    args = parser.parse_args()
    enroll_dataset(args.bucket, args.dataset_root, args.profiles_json)


if __name__ == "__main__":
    main()

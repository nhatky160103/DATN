from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import firebase_admin
from firebase_admin import credentials, db


SERVICE_ACCOUNT_PATH = os.getenv("FIREBASE_SERVICE_ACCOUNT", "database/ServiceAccountKey.json")
DATABASE_URL = os.getenv("FIREBASE_DATABASE_URL", "https://facerecognition-905ff-default-rtdb.firebaseio.com/")


def init_firebase() -> None:
    if firebase_admin._apps:
        return
    cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    firebase_admin.initialize_app(cred, {"databaseURL": DATABASE_URL})


init_firebase()


def get_data(path: str) -> Any:
    return db.reference(path).get()


def save_data(path: str, data: dict[str, Any]) -> None:
    db.reference(path).update(data)


def set_data(path: str, data: Any) -> None:
    db.reference(path).set(data)


def get_all_bucket_names() -> list[str]:
    data = db.reference("/").get()
    return list(data.keys()) if data else []


def create_bucket(bucket_name: str, config_data: dict[str, Any] | None = None) -> bool:
    if bucket_name in get_all_bucket_names():
        return False
    db.reference(f"{bucket_name}/Employees").set({})
    if config_data:
        db.reference(f"{bucket_name}/Config").set(config_data)
    return True


def create_default_config(bucket_name: str, config_data: dict[str, Any]) -> bool:
    ref = db.reference(f"{bucket_name}/Config")
    if ref.get() is not None:
        return False
    ref.set(config_data)
    return True


def add_config_to_bucket(bucket_name: str, data: dict[str, Any]) -> bool:
    db.reference(f"{bucket_name}/Config").set(data)
    return True


def load_config_from_bucket(bucket_name: str) -> dict[str, Any] | None:
    return db.reference(f"{bucket_name}/Config").get()


def get_employee_ids_from_bucket(bucket_name: str) -> list[str]:
    employees = db.reference(f"{bucket_name}/Employees").get() or {}
    return list(employees.keys())


def get_person_ids_from_bucket(bucket_name: str) -> list[str]:
    return get_employee_ids_from_bucket(bucket_name)


def employee_exists(bucket_name: str, employee_id: str) -> bool:
    return db.reference(f"{bucket_name}/Employees/{employee_id}").get() is not None


def upsert_employee(bucket_name: str, employee_id: str, profile: dict[str, Any] | None = None) -> str:
    """Create/update an employee using the exact Firebase ID used by FAISS."""
    if not employee_id:
        raise ValueError("employee_id is required")

    payload = profile.copy() if profile else {}
    payload.setdefault("created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    payload["employee_id"] = employee_id
    db.reference(f"{bucket_name}/Employees/{employee_id}").update(payload)
    return employee_id


def push_recognition_event(bucket_name: str, event: dict[str, Any]) -> str:
    """Append one recognition result event to Firebase."""
    payload = event.copy()
    payload.setdefault("created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    ref = db.reference(f"{bucket_name}/RecognitionEvents").push(payload)
    return ref.key


def delete_person(bucket_name: str, employee_id: str) -> bool:
    ref = db.reference(f"{bucket_name}/Employees/{employee_id}")
    if ref.get() is None:
        return False
    ref.delete()
    return True


def delete_bucket(bucket_name: str, local_embedding_root: str = "local_embeddings") -> bool:
    ref = db.reference(bucket_name)
    existed = ref.get() is not None
    if existed:
        ref.delete()

    local_embedding_dir = Path(local_embedding_root) / bucket_name
    if local_embedding_dir.exists():
        shutil.rmtree(local_embedding_dir)
    return existed


def get_employee_count(bucket_name: str) -> int:
    employees = db.reference(f"{bucket_name}/Employees").get() or {}
    return len(employees)

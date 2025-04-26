import os
import firebase_admin
from firebase_admin import credentials, db
import time

from .cloudinary import  upload_folder_to_cloudinary, delete_folder_from_cloudinary

cred = credentials.Certificate("database/ServiceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://facerecognition-905ff-default-rtdb.firebaseio.com/"
})


def generate_numeric_id(bucket_name):
    ref = db.reference(f'{bucket_name}/Employees')
    data = ref.get() or {}

    existing_ids = set(data.keys())
    i = 0
    while True:
        candidate_id = str(i).zfill(6)
        if candidate_id not in existing_ids:
            return candidate_id
        i += 1

# Lấy dữ liệu từ Firebase
def get_data(bucket_name):
    ref = db.reference(bucket_name)
    data = ref.get()
    return data

# Lưu dữ liệu vào Firebase
def save_data(bucket_name, data):
    ref = db.reference(bucket_name)
    ref.update(data)
    print("Data updated in Firebase")


def add_person(bucket_name, folder_path, name: str, age: int, gender: str = 'Male', salary: int = 0, email: str = None, year: int = 1):
    person_id = generate_numeric_id(bucket_name)

    images = upload_folder_to_cloudinary(bucket_name, person_id, folder_path)

    person_data = {
        person_id: {
            "name": name,
            "age": age,
            "gender": gender,
            "salary": salary,
            "email": email,
            "year": year,
            "images": images
        }
    }

    save_path = f"{bucket_name}/Employees"
    save_data(save_path, person_data)

    print(f"✅ Added person with ID {person_id}")
    return person_id

# Cập nhật thông tin người
def update_person(bucket_name, person_id, **kwargs):
    ref = db.reference(f"{bucket_name}/{person_id}")
    data = ref.get()

    if data is None:
        print(f"Person with ID {person_id} not found!")
        return False

    updated_data = {key: value for key, value in kwargs.items() if key in data}
    ref.update(updated_data)
    print(f"Updated person with ID {person_id}")
    return True

# Xóa người
def delete_person(bucket_name, person_id):
    ref = db.reference(f"{bucket_name}/Employees/{person_id}")
    if ref.get() is None:
        print(f"Person with ID {person_id} not found!")
        return False

    ref.delete()
    delete_folder_from_cloudinary(bucket_name, person_id)
    print(f"Deleted person with ID {person_id}")
    return True


def create_default_config(bucket_name, config_data):
    ref = db.reference(f"{bucket_name}/Config")
    if ref.get() is not None:
        print(f"⚠️ Config already exists in '{bucket_name}'")
        return False

    ref.set(config_data)
    print(f"✅ Config has been created in '{bucket_name}'")
    return True


def add_config_to_bucket(bucket_name, data):
    # Kiểm tra xem bucket có tồn tại chưa
    bucket_ref = db.reference(f"{bucket_name}/Config")
    existing_config = bucket_ref.get()

    if existing_config is not None:
        # Nếu config đã tồn tại, tiến hành cập nhật
        bucket_ref.update(data)
        print(f"✅ Config has been updated in '{bucket_name}'")
    else:
        # Nếu config chưa tồn tại, tạo mới
        bucket_ref.set(data)
        print(f"✅ Config has been created in '{bucket_name}'")

    return True


def load_config_from_bucket(bucket_name):
    ref = db.reference(f"{bucket_name}/Config")
    config = ref.get()

    if config is None:
        print(f"⚠️ No config found in '{bucket_name}'")
        return None

    print(f"✅ Config loaded from '{bucket_name}'")
    return config



def get_all_bucket_names():
    root_ref = db.reference('/')
    data = root_ref.get()
    if data is None:
        return []
    return list(data.keys())


def create_new_bucket(bucket_name: str, config_data: dict = None):
    # Kiểm tra bucket đã tồn tại chưa
    existing_buckets = get_all_bucket_names()
    if bucket_name in existing_buckets:
        print(f"⚠️ Bucket '{bucket_name}' already exists.")
        return False

    # Tạo cấu trúc bucket cơ bản
    db.reference(f"{bucket_name}/Employees").set({})
    print(f"✅ Created new bucket: '{bucket_name}' with empty Employees list.")

    # Tạo config nếu được cung cấp
    if config_data:
        db.reference(f"{bucket_name}/Config").set(config_data)
        print(f"✅ Config set for bucket '{bucket_name}'.")

    return True




    
    
if __name__ =="__main__":
    data = load_config_from_bucket('Hust')
    print(data)
    # delete_person('Hust', '000009')
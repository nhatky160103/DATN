import os
import firebase_admin
from firebase_admin import credentials, db
import time
from datetime import datetime
import shutil

from .timeKeeping import create_daily_timekeeping
from .cloudinary import  (upload_folder_to_cloudinary, 
                          delete_folder_from_cloudinary, 
                          cloudinary_new_bucket,
                          delete_bucket_from_cloudinary)

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


def update_today_timekeeping(bucket_name, person_id):
    today = datetime.now().strftime("%Y-%m-%d")
    ref = db.reference(f"{bucket_name}/Timekeeping/{today}")

    # Nếu bảng hôm nay tồn tại thì update thêm người mới
    if ref.get() is not None:
        employee_data = {
            "check_in": "",
            "check_out": "",
            "working_hours": 0.0,
            "attendance_time": 0.0,
            "attendance_in": "",
            "attendance_out": "",
            "comes_late": 0.0,
            "leaves_early": 0.0,
            "overtime": 0.0
        }
        ref.child(person_id).set(employee_data)
        print(f"✅ Added new employee {person_id} to today's timekeeping {today}")
    else:
        # Nếu hôm nay chưa có bảng → tạo lại cả bảng mới luôn
        create_daily_timekeeping(bucket_name, date=today)



def delete_from_today_timekeeping(bucket_name, person_id):
    today = datetime.now().strftime("%Y-%m-%d")
    ref = db.reference(f"{bucket_name}/Timekeeping/{today}")

    # Nếu bảng hôm nay tồn tại
    if ref.get() is not None:
        person_ref = ref.child(person_id)
        if person_ref.get() is not None:
            person_ref.delete()
            print(f"🗑️ Deleted {person_id} from today's timekeeping ({today})")
            return True
        else:
            print(f"⚠️ Person {person_id} not found in today's timekeeping ({today})")
            return False
    else:
        print(f"⚠️ Timekeeping table for {today} does not exist.")
        return False



def add_person(bucket_name, folder_path, name: str, age: int, gender: str = 'Male', salary: int = 0, email: str = None, year: int = 1):
    person_id = generate_numeric_id(bucket_name)

    images = upload_folder_to_cloudinary(bucket_name, person_id, folder_path)
    update_today_timekeeping(bucket_name, person_id)
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

# Xóa người
def delete_person(bucket_name, person_id):
    ref = db.reference(f"{bucket_name}/Employees/{person_id}")
    if ref.get() is None:
        print(f"Person with ID {person_id} not found!")
        return False

    ref.delete()
    delete_folder_from_cloudinary(bucket_name, person_id)
    delete_from_today_timekeeping(bucket_name, person_id)
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


def create_new_bucket(bucket_name: str, config_data: dict = None, logo_path: str = None):
    # Kiểm tra bucket đã tồn tại chưa
    existing_buckets = get_all_bucket_names()
    if bucket_name in existing_buckets:
        print(f"⚠️ Bucket '{bucket_name}' already exists.")
        return False

    logo_url = cloudinary_new_bucket(bucket_name, logo_path=logo_path)
    db.reference(f"{bucket_name}/Logo/url").set(logo_url)
    db.reference(f"{bucket_name}/Employees").set({})
    print(f"✅ Created new bucket: '{bucket_name}' with empty Employees list.")

    # Tạo config nếu được cung cấp
    if config_data:
        db.reference(f"{bucket_name}/Config").set(config_data)
        print(f"✅ Config set for bucket '{bucket_name}'.")

    return True

def get_logo_url(bucket_name):
    ref = db.reference(f"{bucket_name}/Logo/url")
    url = ref.get()
    return url


def get_person_ids_from_bucket(bucket_name):
    # Lấy dữ liệu nhân viên từ Firebase
    ref = db.reference(f"{bucket_name}/Employees")
    data = ref.get()

    # Nếu không có dữ liệu thì trả về danh sách rỗng
    if data is None:
        print(f"⚠️ No employees found in bucket '{bucket_name}'")
        return []

    # Lấy danh sách các person_id từ dữ liệu
    person_ids = list(data.keys())
    return person_ids


def delete_bucket(bucket_name):
    ref = db.reference(bucket_name)
    firebase_ok = ref.get() is not None
    if firebase_ok:
        ref.delete()
        print(f"✅ Deleted bucket '{bucket_name}' from Firebase.")
    else:
        print(f"⚠️ Bucket '{bucket_name}' does not exist in Firebase.")

    cloudinary_ok = delete_bucket_from_cloudinary(bucket_name)

    # Xóa local embedding folder
    local_embedding_dir = os.path.join("local_embeddings", bucket_name)
    if os.path.exists(local_embedding_dir):
        try:
            shutil.rmtree(local_embedding_dir)
            print(f"🗑️ Deleted local embeddings folder: {local_embedding_dir}")
            local_ok = True
        except Exception as e:
            print(f"⚠️ Failed to delete local embeddings folder: {e}")
            local_ok = False
    else:
        print(f"ℹ️ No local embeddings folder found for '{bucket_name}'")
        local_ok = True

    if firebase_ok and cloudinary_ok and local_ok:
        print(f"✅ Successfully deleted bucket '{bucket_name}' from Firebase, Cloudinary, and local embeddings.")
        return True
    return False
    

def get_employee_count(bucket_name):
    """
    Trả về số lượng nhân viên trong bucket (dựa vào node Employees).
    """
    ref = db.reference(f"{bucket_name}/Employees")
    data = ref.get()
    if not data:
        return 0
    return len(data)


if __name__ =="__main__":
    # data = load_config_from_bucket('Hust')
    # print(data)
    num_person = get_person_ids_from_bucket('Neu')
    print(num_person)
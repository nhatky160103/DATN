import os
import cloudinary
import cloudinary.uploader
import cloudinary.api
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get values from the .env file
cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME")
api_key = os.getenv("CLOUDINARY_API_KEY")
api_secret = os.getenv("CLOUDINARY_API_SECRET")

# Configure Cloudinary
cloudinary.config(
    cloud_name=cloud_name,
    api_key=api_key,
    api_secret=api_secret
)

# Upload folder to Cloudinary
def upload_folder_to_cloudinary(bucket_name, person_id, folder_path):
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} not found!")
        return False

    cloud_folder = f"{bucket_name}/Employees/{person_id}"

    try:
        if cloudinary.api.resources(type="upload", prefix=cloud_folder, max_results=1)["resources"]:
            print(f"Folder {cloud_folder} already exists on Cloudinary!")
            return False
    except Exception as e:
        print(f"Error checking Cloudinary folder: {e}")
        return False

    if not os.listdir(folder_path):
        print(f"No images found in the folder {folder_path}!")
        return False

    images = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            upload_result = cloudinary.uploader.upload(file_path, folder=cloud_folder)
            print(f"Uploaded {filename} to Cloudinary with URL: {upload_result['url']}")
            images.append(upload_result['url'])
    return images

# Xóa folder từ Cloudinary
def delete_folder_from_cloudinary(bucket_name, person_id):
    cloud_folder = f"{bucket_name}/Employees/{person_id}"
    try:
        result = cloudinary.api.delete_resources_by_prefix(cloud_folder)
        cloudinary.api.delete_folder(cloud_folder)
        print(f"Deleted folder {cloud_folder} from Cloudinary.")
        return True
    except Exception as e:
        print(f"Error deleting folder {cloud_folder}: {e}")
        return False

# Lấy URL ảnh từ Cloudinary
def get_images_from_cloudinary(bucket_name, person_id):
    cloud_folder = f"{bucket_name}/Employees/{person_id}"
    try:
        result = cloudinary.api.resources(type="upload", prefix=cloud_folder, max_results=500)
        image_urls = [image['url'] for image in result['resources']]
        print(f"Found {len(image_urls)} images for person {person_id}.")
        return image_urls
    except Exception as e:
        print(f"Error retrieving images for person {person_id}: {e}")
        return []



def upload_embedding_to_cloudinary(bucket_name, folder_path):
    if not os.path.exists(folder_path):
        print(f"❌ Folder {folder_path} not found!")
        return False

    cloud_folder = f"{bucket_name}/Embeddings"

    # Kiểm tra xem folder đã có gì trên Cloudinary chưa
    try:
        existing = cloudinary.api.resources(
            type="upload", prefix=cloud_folder, max_results=1
        )["resources"]
        if existing:
            print(f"⚠️ Folder {cloud_folder} already exists on Cloudinary!")
            return False
    except Exception as e:
        print(f"❌ Error checking Cloudinary folder: {e}")
        return False

    uploaded_files = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):
            try:
                # Upload file dưới dạng raw
                upload_result = cloudinary.uploader.upload(
                    file_path,
                    folder=cloud_folder,
                    public_id=os.path.splitext(filename)[0],  # bỏ đuôi để không bị trùng
                    resource_type="raw",
                    overwrite=True
                )
                print(f"✅ Uploaded {filename} → {upload_result['secure_url']}")
                uploaded_files.append(upload_result['secure_url'])

            except Exception as e:
                print(f"❌ Failed to upload {filename}: {e}")

    return uploaded_files


if __name__ == "__main__":
    # upload_folder_to_cloudinary('Hust', '000001', 'data/Testset/baejun')
    # image_list = get_images_from_cloudinary('Hust', '000000')
    # print(image_list)
    upload_embedding_to_cloudinary('Hust', 'data/data_source')

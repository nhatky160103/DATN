from flask import Flask, render_template, request, jsonify, send_file, session, Response
import os, cv2, yaml, shutil, queue
from werkzeug.utils import secure_filename
import time
from datetime import datetime, timedelta
import io
import numpy as np
from PIL import Image

from infer.get_embedding import EmbeddingManager
from infer.infer_camera import infer_camera, check_validation
from database.timeKeeping import (create_daily_timekeeping,
                                   export_to_excel, 
                                   process_check_in_out)
from database.firebase import (get_all_bucket_names, 
                               load_config_from_bucket, 
                               add_config_to_bucket, 
                               create_new_bucket, 
                               delete_bucket, 
                               get_logo_url,
                               get_employee_count)
from models.lightqnet.tf_face_quality_model import TfFaceQaulityModel

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.permanent_session_lifetime = timedelta(days=7)


@app.before_request
def make_session_permanent():
    session.permanent = True

print("🕒 Initializing timekeeping table for today...")
today = datetime.now().strftime("%Y-%m-%d")

# Khởi tạo biến toàn cục để lưu embedding và metadata cho từng bucket
app.config['embeddings'] = {}
app.config['image2class'] = {}
app.config['index2class'] = {}
app.config['config'] = {}

# Initialize face quality model
face_quality_model = TfFaceQaulityModel()

def get_default_config():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config

for bucket in get_all_bucket_names():
    create_daily_timekeeping(bucket, date=today)

    try:
        print(f"🔍 Loading local embeddings for bucket: {bucket}")
        manager = EmbeddingManager(bucket_name=bucket)
        embeddings, image2class, index2class = manager.load(load_local=True)

        app.config['embeddings'][bucket] = embeddings
        app.config['image2class'][bucket] = image2class
        app.config['index2class'][bucket] = index2class
        app.config['config'][bucket]  = load_config_from_bucket(bucket) or get_default_config()
        print(f"✅ Loaded to global config for bucket: {bucket}")

    except Exception as e:
        print(f"❌ Error loading for bucket {bucket}: {e}")

def get_or_set_default_bucket():
    bucket = session.get('selected_bucket')
    if not bucket:
        all_buckets = get_all_bucket_names()
        if all_buckets:
            bucket = all_buckets[0]
            session['selected_bucket'] = bucket
    return bucket

@app.route('/set_selected_bucket', methods=['POST'])
def set_selected_bucket():
    data = request.get_json()
    selected_bucket = data.get('bucket')
    if selected_bucket:
        session['selected_bucket'] = selected_bucket
    print(f"Selected bucket: {selected_bucket}")
    return jsonify({'status': 'success', 'selected_bucket': selected_bucket})


@app.route('/')
def index():
    buckets = get_all_bucket_names()
    selected_bucket = get_or_set_default_bucket()
    return render_template('index.html', buckets=buckets, selected_bucket=selected_bucket)


@app.route('/dashboard')
def dashboard():
    bucket_name = get_or_set_default_bucket()
    config_data = load_config_from_bucket(bucket_name) or get_default_config()
    logo_url = get_logo_url(bucket_name)
    return render_template('dashboard2.html', config=config_data, bucket_name=bucket_name, logo_url=logo_url)

@app.route('/add_member', methods=['POST'])
def add_member():
    name = request.form.get('name')
    age = request.form.get('age')
    gender = request.form.get('gender')
    salary = request.form.get('salary')
    email = request.form.get('email')
    year = request.form.get('experience_year')
    photos = request.files.getlist('photos')
    UPLOAD_FOLDER = 'static/uploads'
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    bucket_name = get_or_set_default_bucket()
    manager = EmbeddingManager(bucket_name)

    for idx, photo in enumerate(photos):
        if photo:
            filename = secure_filename(f"{name}_{idx}.jpg")
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            photo.save(filepath)
            print(f"📸 Photo {idx + 1} saved at: {filepath}")

    embeddings, image2class, index2class =  manager.add_employee(UPLOAD_FOLDER, name, age, gender, salary, email, year)
    if embeddings is not None:
        app.config['embeddings'][bucket_name] = embeddings
        app.config['image2class'][bucket_name] = image2class
        app.config['index2class'][bucket_name] = index2class
        print(f"🔄 Updated global embeddings for bucket: {bucket_name}")
    else:
        print(f"⚠️ Failed to reload embeddings after adding member to {bucket_name}")

    shutil.rmtree(UPLOAD_FOLDER)
    return jsonify({"status": "success", "saved_photos": UPLOAD_FOLDER})


@app.route("/export-timekeeping", methods=["POST"])
def handle_export_timekeeping():


    data = request.get_json()
    selected_date = data.get('date')
    selected_bucket = get_or_set_default_bucket()
    
    try:
        df = export_to_excel(selected_bucket, selected_date)
        if df is None or df.empty:
            return jsonify({"success": False, "message": "❗ No timekeeping data found."}), 404

        data_json = df.to_dict(orient='records')
        return jsonify({"success": True, "data": data_json})
    except Exception as e:
        return jsonify({"success": False, "message": f"❌ Error: {str(e)}"}), 500


@app.route("/download-excel", methods=["POST"])
def download_excel():
    data = request.get_json()
    selected_date = data.get('date')
    selected_bucket = get_or_set_default_bucket()

    df = export_to_excel(selected_bucket, selected_date)
    if df is None or df.empty:
        return jsonify({"success": False, "message": "No data for selected date."}), 404

    output = io.BytesIO()
    df.to_excel(output, index=False)
    output.seek(0)

    return send_file(output, as_attachment=True, download_name="timekeeping.xlsx", mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


@app.route('/save_config', methods=['POST'])
def save_config():
    try:
        config_data = request.get_json() or get_default_config()
        bucket = get_or_set_default_bucket()
        app.config['config'][bucket] = config_data
        add_config_to_bucket(bucket, config_data)

        return jsonify({"message": "Configuration saved successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

camera_data = queue.Queue()

@app.route('/video_feed')
def video_feed():
    print('all keys:',  app.config['config'].keys())
    try:
        bucket_name = get_or_set_default_bucket()
        # Không giữ reference tới stream cũ, chỉ trả về Response mới mỗi lần
        return Response(
            infer_camera(config=app.config['config'][bucket_name], result_queue=camera_data),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
    except Exception as e:
        print(f"Error streaming video: {e}")
        return "Error streaming video"


@app.route('/get_results', methods=['GET'])
def get_results():
    bucket_name = get_or_set_default_bucket()

    embeddings = app.config['embeddings'].get(bucket_name)
    image2class = app.config['image2class'].get(bucket_name)
    index2class = app.config['index2class'].get(bucket_name)

    if not camera_data.empty():
        input_data = camera_data.get()
        employee_id = check_validation(input_data, embeddings, image2class, index2class, app.config['config'][bucket_name])
        process_check_in_out(bucket_name, employee_id)
        
        # Trả về kết quả với thông tin chi tiết hơn
        return jsonify({
            'employee_id': employee_id,
            'time': datetime.now().timestamp(),
            'recognition_id': input_data.get('recognition_id', 0),
            'status': 'success'
        })
    else:
        return jsonify({"status": "no_results", "message": "No results available yet"})



@app.route('/get_person_ids', methods=['GET'])
def get_person_ids():
    try:
        bucket_name = get_or_set_default_bucket()
        manager = EmbeddingManager(bucket_name)
        person_ids = manager.load_person_ids()
        return jsonify(person_ids)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API để xóa nhân viên theo person_id
@app.route('/delete_person/<person_id>', methods=['DELETE'])
def delete_person(person_id):
    try:
        bucket_name = get_or_set_default_bucket()
        manager = EmbeddingManager(bucket_name)
        embeddings, image2class, index2class = manager.delete_employee(person_id)

        if embeddings is not None:
            app.config['embeddings'][bucket_name] = embeddings
            app.config['image2class'][bucket_name] = image2class
            app.config['index2class'][bucket_name] = index2class
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "error": "Failed to delete employee."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/create_bucket", methods=["POST"])
def create_bucket():
    # Nếu client gửi form-data (có file), dùng request.form và request.files
    if request.content_type and request.content_type.startswith('multipart/form-data'):
        bucket_name = request.form.get('bucket_name')
        logo_file = request.files.get('logo')
        config_data = get_default_config()
        logo_path = None
        if logo_file:
            temp_dir = os.path.join('static', 'uploads')
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, secure_filename(logo_file.filename))
            logo_file.save(temp_path)
            logo_path = temp_path
    else:
        # Nếu client gửi JSON như cũ
        data = request.get_json()
        bucket_name = data.get('bucket_name')
        config_data = get_default_config()
        logo_path = None

    if not bucket_name:
        return jsonify({"success": False, "message": "Bucket name is required."})

    try:
        created = create_new_bucket(bucket_name, config_data, logo_path=logo_path)
        if logo_path:
            os.remove(logo_path)
    except Exception as e:
        if logo_path:
            try: os.remove(logo_path)
            except: pass
        return jsonify({"success": False, "message": str(e)})

    if created:
        print('created:',bucket_name)
        app.config['embeddings'][bucket_name] = None
        app.config['image2class'][bucket_name] = None
        app.config['index2class'][bucket_name] = None
        app.config['config'][bucket_name] = load_config_from_bucket(bucket_name) or get_default_config()
        return jsonify({"success": True})
    else:
        return jsonify({"success": False, "message": f"Bucket '{bucket_name}' already exists or creation failed."})


@app.route('/delete_bucket', methods=['POST'])
def handle_delete_bucket():
    data = request.get_json()
    bucket_name = data.get('bucket_name')
    if not bucket_name:
        return jsonify({"success": False, "message": "Missing bucket_name"}), 400

    result = delete_bucket(bucket_name)
    if result:
        for key in ['embeddings', 'image2class', 'index2class', 'config']:
            app.config[key].pop(bucket_name, None)
        # Nếu bucket bị xóa là bucket đang chọn thì reset session
        if session.get('selected_bucket') == bucket_name:
            session.pop('selected_bucket', None)
        return jsonify({"success": True, "message": f"Bucket '{bucket_name}' deleted successfully."})
    else:
        return jsonify({"success": False, "message": f"Failed to delete bucket '{bucket_name}'."}), 500



@app.route('/get_employee_count', methods=['GET'])
def get_employee_count_api():
    bucket_name = request.args.get('bucket_name') or get_or_set_default_bucket()
    if not bucket_name:
        return {'error': 'Missing bucket_name'}, 400
    count = get_employee_count(bucket_name)
    return {'count': count}

@app.route('/check_face_quality', methods=['POST'])
def check_face_quality():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Read image from request
        image_file = request.files['image']
        image_bytes = image_file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        height, width = img.shape[:2]  # Get image dimensions
        
        # Get face quality score
        # Crop center of the image to align face belong to elip shape
        start_h = int(0.1 * height)
        end_h = int(0.9 * height)
        start_w = int(0.2 * width)
        end_w = int(0.8 * width)
        score = face_quality_model.inference(img[start_h:end_h, start_w:end_w])
        
        return jsonify({
            'score': float(score),
            'success': True
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/camera_status')
def camera_status():
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        cap.release()
        return jsonify({"status": "ready"})
    else:
        return jsonify({"status": "busy"})

if __name__ == '__main__':
    import logging
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    app.run(debug=True)
from flask import Flask, render_template, request, jsonify, send_file, session
import os, cv2, yaml, shutil
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
import io
import numpy as np
from models.detection.detect import detect_face_and_nose
from models.Anti_spoof.FasNet_onnx import FasnetOnnx


from infer.get_embedding import EmbeddingManager
from infer.infer_camera import check_validation
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

from models.lightqnet.tf_face_quality_model_onnx import OnnxFaceQualityModel

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
face_quality_model = OnnxFaceQualityModel()
antispoof_model = FasnetOnnx()


def get_default_config():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config

for bucket in get_all_bucket_names():
    create_daily_timekeeping(bucket, date=today)

    try:
        print(f"🔍 Loading local embeddings for bucket: {bucket}")
        manager = EmbeddingManager(bucket_name=bucket)
        embeddings, image2class, index2class = manager.load(load_local=False)

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


valid_images_buffer = []
is_reals_buffer = []


@app.route('/infer_camera_upload', methods=['POST'])
def infer_camera_upload():
    global valid_images_buffer, is_reals_buffer
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'message': 'No image uploaded'}), 400

        # đọc ảnh
        image_file = request.files['image']
        image_bytes = image_file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'success': False, 'message': 'Invalid image'}), 400

        # lấy config
        bucket_name = get_or_set_default_bucket()
        embeddings = app.config['embeddings'].get(bucket_name)
        image2class = app.config['image2class'].get(bucket_name)
        index2class = app.config['index2class'].get(bucket_name)
        config = app.config['config'][bucket_name]

        guide_base = '/static/audio/'
        min_face_area = config['infer_video']['min_face_area']
        qscore_threshold = config['infer_video']['qscore_threshold']
        required_images = config['infer_video'].get('required_images', 3)

        # detect face
        face, center_point, prob = detect_face_and_nose(img)
        h, w = img.shape[:2]

        x1, y1, x2, y2 = map(int, face)
        area = (x2 - x1) * (y2 - y1)
        center_x, center_y = map(int, center_point)

        # check diện tích
        if area < min_face_area * h * w:
            return jsonify({
                'employee_id': None,
                'time': datetime.now().timestamp(),
                'audio_guide': guide_base + 'closer.mp3',
                'valid': False,
                'enough_images': False
            })

        # check vị trí
        if not (w * 0.2 < center_x < w * 0.8 and h * 0.2 < center_y < h * 0.8):
            return jsonify({
                'employee_id': None,
                'time': datetime.now().timestamp(),
                'audio_guide': guide_base + 'guide_centerface.mp3',
                'valid': False,
                'enough_images': False
            })

        # crop + check quality
        crop_face = img[y1:y2, x1:x2]
        quality_score = face_quality_model.inference(crop_face)
        if quality_score < qscore_threshold:
            return jsonify({
                'employee_id': None,
                'time': datetime.now().timestamp(),
                'audio_guide': guide_base + 'guide_centerface.mp3',
                'valid': False,
                'enough_images': False
            })


        is_real, score = antispoof_model.analyze(img, [x1, y1, x2, y2])

        if len(valid_images_buffer) >= required_images:
            valid_images_buffer = []
            is_reals_buffer = []
            
        valid_images_buffer.append(crop_face)
        is_reals_buffer.append((is_real, score))

        # chưa đủ ảnh
        if len(valid_images_buffer) < required_images:
            return jsonify({
                'employee_id': None,
                'time': datetime.now().timestamp(),
                'audio_guide': guide_base + 'guide_keepface.mp3',
                'valid': True,
                'enough_images': False
            })

        # đủ ảnh -> gọi check_validation
        input_data = {
            'valid_images': valid_images_buffer,
            'is_reals': is_reals_buffer
        }
        employee_id, audio_guide = check_validation(
            input_data, embeddings, image2class, index2class, config
        )
        process_check_in_out(bucket_name, employee_id)
        # reset buffer
        valid_images_buffer = []
        is_reals_buffer = []
        
        return jsonify({
            'employee_id': employee_id,
            'time': datetime.now().timestamp(),
            'audio_guide': audio_guide,
            'valid': True,
            'enough_images': True
        })

    except Exception as e:
        import traceback
        traceback.print_exc()  # in ra console server
        return jsonify({'success': False, 'message': str(e)}), 500


if __name__ == '__main__':
    import logging
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    app.run(debug=True)
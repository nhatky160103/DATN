from flask import Flask, render_template, request, jsonify, send_file, session, redirect, Response
import os, cv2, yaml, shutil, queue
from werkzeug.utils import secure_filename
import time
from datetime import datetime, timedelta

from infer.get_embedding import EmbeddingManager
from infer.infer_camera import infer_camera, check_validation
from database.timeKeeping import create_daily_timekeeping, export_to_excel, process_check_in_out
from database.firebase import get_all_bucket_names, load_config_from_bucket, add_config_to_bucket

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

        if embeddings is not None:
            app.config['embeddings'][bucket] = embeddings
            app.config['image2class'][bucket] = image2class
            app.config['index2class'][bucket] = index2class
            app.config['config'][bucket]  = load_config_from_bucket(bucket) or get_default_config()
            print(f"✅ Loaded to global config for bucket: {bucket}")
        else:
            print(f"⚠️ No data found for bucket: {bucket}")
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

@app.route('/')
def index():
    buckets = get_all_bucket_names()
    selected_bucket = get_or_set_default_bucket()
    return render_template('index.html', buckets=buckets, selected_bucket=selected_bucket)
    return render_template('index.html')

@app.route('/set_selected_bucket', methods=['POST'])
def set_selected_bucket():
    data = request.get_json()
    selected_bucket = data.get('bucket')
    if selected_bucket:
        session['selected_bucket'] = selected_bucket
    print(f"Selected bucket: {selected_bucket}")
    return jsonify({'status': 'success', 'selected_bucket': selected_bucket})

@app.route('/dashboard')
def dashboard():
    bucket_name = get_or_set_default_bucket()
    config_data = load_config_from_bucket(bucket_name) or get_default_config()
    return render_template('dashboard.html', config=config_data, bucket_name=bucket_name)

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

    person_id = add_person(bucket_name, UPLOAD_FOLDER, name, age, gender, salary, email, year)
    embeddings, image2class, index2class =  manager.add_employee(person_id)
    if embeddings is not None:
        app.config['embeddings'][bucket_name] = embeddings
        app.config['image2class'][bucket_name] = image2class
        app.config['index2class'][bucket_name] = index2class
        print(f"🔄 Updated global embeddings for bucket: {bucket_name}")
    else:
        print(f"⚠️ Failed to reload embeddings after adding member to {bucket_name}")

    shutil.rmtree(UPLOAD_FOLDER)
    return jsonify({"status": "success", "saved_photos": UPLOAD_FOLDER})

@app.route('/create-embedding', methods=['POST'])
def handle_create_embedding():
    try:
        selected_bucket = get_or_set_default_bucket()
        embeddings, image2class, index2class = create_data_embeddings(selected_bucket)
        if embeddings is None or embeddings.size == 0:
            return jsonify({"success": False, "message": "No selected bucket in session."}), 400
        return jsonify({"success": True})
    except Exception as e:
        print("Error:", e)
        return jsonify({"success": False, "message": "Error during embedding creation."}), 500

@app.route("/export-timekeeping", methods=["POST"])
def handle_export_timekeeping():
    data = request.get_json()
    selected_date = data.get('date')  
    selected_bucket = get_or_set_default_bucket()
    try:
        export_to_excel(selected_bucket, selected_date)
        return jsonify({"message": f"✅ Data exported for bucket '{selected_bucket}' on {selected_date}."})
    except Exception as e:
        return jsonify({"message": f"❌ Error: {str(e)}"}), 500

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
    try:
        bucket_name = get_or_set_default_bucket()
        return Response(infer_camera(config = app.config['config'][bucket_name], 
        result_queue=camera_data), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Error streaming video: {e}")
        return "Error streaming video"


@app.route('/get_results', methods=['GET'])
def get_results():

    bucket_name = get_or_set_default_bucket()

    embeddings = app.config['embeddings'].get(bucket_name)
    image2class = app.config['image2class'].get(bucket_name)
    index2class = app.config['index2class'].get(bucket_name)

    if embeddings is None:
        return jsonify({"status": "error", "message": f"⚠️ No embeddings loaded for bucket: {bucket_name}"}), 500

    if not camera_data.empty():
        input_data = camera_data.get()
        employee_id = check_validation(input_data, embeddings, image2class, index2class, app.config['config'][bucket_name])
        process_check_in_out(bucket_name, employee_id)
        return jsonify({
            'employee_id': employee_id,
            'time': datetime.now().timestamp(),
        })
    else:
        return jsonify({"status": "no_results", "message": "No results available yet"})

if __name__ == '__main__':
    app.run(debug=True)
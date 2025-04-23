from flask import Flask, render_template, request, jsonify, send_file, session, redirect, Response
from database.firebase import *
import os, cv2
from werkzeug.utils import secure_filename
import yaml, shutil
from infer.get_embedding import create_data_embeddings
from infer.infer_camera import infer_camera
from datetime import datetime
from database.timeKeeping import create_daily_timekeeping, export_to_excel
from threading import Thread

app = Flask(__name__)
app.secret_key = os.urandom(24)

print("🕒 Initializing timekeeping table for today...")
today = datetime.now().strftime("%Y-%m-%d")
for bucket in get_all_bucket_names():
    create_daily_timekeeping(bucket, date=today)

def get_default_config():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config


@app.route('/')
def index():
    buckets = get_all_bucket_names()
    selected_bucket = session.get('selected_bucket')  # Get bucket from session
    if not selected_bucket and buckets:
        selected_bucket = buckets[0]  # Choose default bucket if no value in session
        session['selected_bucket'] = selected_bucket  # Save to session
    
    return render_template('index.html', buckets=buckets, selected_bucket=selected_bucket)


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
    # Check if 'selected_bucket' exists in session
    bucket_name = session.get('selected_bucket')
    if bucket_name is None:
        return redirect('/')  # Redirect to index page if no bucket selected
    config_data = load_config_from_bucket(bucket_name)
    if config_data is None:
        config_data = get_default_config()
    return render_template('dashboard.html', config=config_data, bucket_name=bucket_name)


@app.route('/add_member', methods=['POST'])
def add_member():
    name = request.form.get('name')
    age = request.form.get('age')
    gender = request.form.get('gender')
    salary = request.form.get('salary')
    email = request.form.get('email')
    year = request.form.get('experience_year')

    photos = request.files.getlist('photos')  # Get list of photos
    UPLOAD_FOLDER = 'static/uploads'
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    for idx, photo in enumerate(photos):
        if photo:
            filename = secure_filename(f"{name}_{idx}.jpg")
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            photo.save(filepath)
            print(f"📸 Photo {idx + 1} saved at: {filepath}")

    add_person(session.get('selected_bucket'), UPLOAD_FOLDER, name, age, gender, salary, email, year)
    shutil.rmtree(UPLOAD_FOLDER)

    return jsonify({"status": "success", "saved_photos": UPLOAD_FOLDER})


@app.route('/create-embedding', methods=['POST'])
def handle_create_embedding():
    try:
        selected_bucket = session.get('selected_bucket')
        
        if not selected_bucket:
            return jsonify({"success": False, "message": "No bucket selected."}), 400

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
    selected_bucket = session.get("selected_bucket")
    if not selected_bucket:
        return jsonify({"message": "❌ Bucket not found in session!"}), 400

    try:
        export_to_excel(selected_bucket, selected_date)
        return jsonify({"message": f"✅ Data exported for bucket '{selected_bucket}' on {selected_date}."})
    except Exception as e:
        return jsonify({"message": f"❌ Error: {str(e)}"}), 500


@app.route('/save_config', methods=['POST'])
def save_config():
    try:
        # Receive data from client (JSON)
        config_data = request.get_json()

        print(config_data)
        print(session.get('selected_bucket'))

        add_config_to_bucket(session.get('selected_bucket'), config_data)
        return jsonify({"message": "Configuration saved successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400



@app.route('/open_camera', methods=['POST'])
def open_camera():
    return jsonify({"status": "success", "message": "Camera is starting..."})

@app.route('/video_feed')
def video_feed():
    try:
        return Response(infer_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Error streaming video: {e}")
        return "Error streaming video"


# def generate_frames():
#     cap = cv2.VideoCapture(0)
#     while True:
#         success, frame = cap.read()
#         if not success:
#             break
#         _, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#     # Đóng camera và cửa sổ OpenCV khi kết thúc
#     cap.release()
#     cv2.destroyAllWindows()

if __name__ == '__main__':
    app.run(debug=True)

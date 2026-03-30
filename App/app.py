import os
import sys
import io
import base64
import math
import logging
import subprocess
import cv2
import numpy as np
import imageio_ffmpeg
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions
from flask import Flask, request, render_template, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import uuid
import threading
import tensorflow as tf
from tensorflow.keras.models import load_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB limit
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model (suppress lz4 I/O warnings)
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'tmp_checkpoint', 'best_model.h5')
logger.info('Loading model from %s', MODEL_PATH)
_stderr = sys.stderr
sys.stderr = io.StringIO()
model = load_model(MODEL_PATH)
sys.stderr = _stderr
logger.info('Model loaded successfully')
INPUT_SIZE = 128

# Initialize MediaPipe face detector
logger.info('Initializing MediaPipe face detector')
FACE_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'blaze_face_short_range.tflite')
face_detector_options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
    min_detection_confidence=0.5
)
logger.info('MediaPipe face detector ready')

# In-memory job store: job_id -> {status, result, ...}
jobs = {}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def face_to_base64(face_rgb):
    face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', face_bgr)
    return base64.b64encode(buffer).decode('utf-8')


def reencode_to_h264(input_path, output_path=None):
    """Re-encode a video to H.264 for browser compatibility. Overwrites in-place if no output_path."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    if output_path is None:
        output_path = input_path
    tmp = input_path + '.reencode.mp4'
    cmd = [
        ffmpeg_exe, '-y', '-i', input_path,
        '-c:v', 'libx264', '-preset', 'fast',
        '-movflags', '+faststart', '-pix_fmt', 'yuv420p',
        tmp
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error('ffmpeg reencode failed: %s', result.stderr)
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False
    try:
        os.replace(tmp, output_path)
    except OSError:
        os.remove(input_path)
        os.rename(tmp, output_path)
    return True


def extract_faces_from_video(video_path):
    logger.info('Extracting faces from video: %s', video_path)
    faces = []
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    if frame_rate == 0:
        logger.warning('Could not read frame rate from video')
        cap.release()
        return faces

    with FaceDetector.create_from_options(face_detector_options) as face_det:
        while cap.isOpened():
            frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES)
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id % math.floor(frame_rate) == 0:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                results = face_det.detect(mp_image)
                for detection in results.detections:
                    score = detection.categories[0].score
                    if score > 0.5:
                        bbox = detection.bounding_box
                        bx, by, bw, bh = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
                        h, w = image_rgb.shape[:2]
                        margin_x = int(bw * 0.3)
                        margin_y = int(bh * 0.3)
                        x1 = max(0, bx - margin_x)
                        x2 = min(w, bx + bw + margin_x)
                        y1 = max(0, by - margin_y)
                        y2 = min(h, by + bh + margin_y)
                        crop = image_rgb[y1:y2, x1:x2]
                        if crop.size > 0:
                            crop_resized = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE))
                            faces.append(crop_resized)

    cap.release()
    logger.info('Face extraction complete — %d faces found', len(faces))
    return faces


def create_processed_video(video_path, output_path, face_scores=None):
    """Re-encode video with face bounding boxes and per-face REAL/FAKE label."""
    logger.info('Creating processed video with bounding boxes: %s', output_path)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Write to a temp file with mp4v codec first
    temp_path = output_path + '.tmp.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_path, fourcc, fps, (w, h))

    if not out.isOpened():
        logger.error('VideoWriter failed to open: %s', temp_path)
        cap.release()
        return

    frame_count = 0
    with FaceDetector.create_from_options(face_detector_options) as face_det:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            results = face_det.detect(mp_image)
            for detection in results.detections:
                det_score = detection.categories[0].score
                if det_score > 0.5:
                    bbox = detection.bounding_box
                    bx, by, bw, bh = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
                    x, y = max(0, bx), max(0, by)

                    # Crop and predict this face individually
                    margin_x = int(bw * 0.3)
                    margin_y = int(bh * 0.3)
                    x1 = max(0, bx - margin_x)
                    x2 = min(w, bx + bw + margin_x)
                    y1 = max(0, by - margin_y)
                    y2 = min(h, by + bh + margin_y)
                    crop = image_rgb[y1:y2, x1:x2]
                    if crop.size > 0:
                        crop_resized = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE))
                        face_input = np.array([crop_resized], dtype='float32') / 255.0
                        score = float(model.predict(face_input, verbose=0)[0][0])
                    else:
                        score = 0.0

                    is_real = score > 0.5
                    label = 'REAL' if is_real else 'FAKE'
                    color = (0, 255, 0) if is_real else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
                    text = f'{label} {score:.2f}'
                    cv2.putText(frame, text, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            out.write(frame)
            frame_count += 1

    cap.release()
    out.release()
    logger.info('Wrote %d frames to temp file, re-encoding to H.264', frame_count)

    # Re-encode to H.264 for browser compatibility
    if reencode_to_h264(temp_path, output_path):
        logger.info('Processed video saved (H.264): %s', output_path)
    else:
        logger.error('Failed to re-encode processed video')

    # Clean up temp file
    try:
        os.remove(temp_path)
    except OSError:
        pass


def predict_deepfake(faces):
    if not faces:
        logger.warning('No faces to predict on')
        return None, 0, []

    logger.info('Running prediction on %d face(s)', len(faces))

    face_array = np.array(faces, dtype='float32') / 255.0
    predictions = model.predict(face_array, verbose=0)
    avg_prediction = float(np.mean(predictions))

    # Build per-face details (up to 3 evenly spaced faces)
    total = len(faces)
    if total <= 3:
        indices = list(range(total))
    else:
        indices = [0, total // 2, total - 1]

    faces_detail = []
    for i in indices:
        faces_detail.append({
            'thumbnail': face_to_base64(faces[i]),
            'score': float(predictions[i][0])
        })

    logger.info('Prediction complete — avg score: %.4f, faces: %d', avg_prediction, total)
    return avg_prediction, total, faces_detail


def cleanup_old_uploads(exclude=None):
    """Delete all files in the upload folder except those in exclude."""
    exclude = set(exclude or [])
    folder = app.config['UPLOAD_FOLDER']
    for f in os.listdir(folder):
        fpath = os.path.join(folder, f)
        if os.path.isfile(fpath) and fpath not in exclude:
            try:
                os.remove(fpath)
            except PermissionError:
                pass


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_video(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, mimetype='video/mp4')


def process_video_job(job_id, filepath, unique_name):
    """Background worker: extract faces, predict, create processed video."""
    try:
        logger.info('[Job %s] Starting face detection', job_id)
        jobs[job_id]['status'] = 'detecting'

        faces = extract_faces_from_video(filepath)
        avg_score, num_faces, faces_detail = predict_deepfake(faces)

        if avg_score is None:
            logger.warning('[Job %s] No faces detected', job_id)
            jobs[job_id].update({
                'status': 'done',
                'error': 'No faces detected in the video.',
                'video_url': f'/uploads/{unique_name}',
            })
            return

        is_real = avg_score > 0.5
        label = 'REAL' if is_real else 'FAKE'
        confidence = avg_score if is_real else (1 - avg_score)

        # Publish detection results immediately
        logger.info('[Job %s] Detection done — result: %s, confidence: %.2f%%, faces: %d',
                    job_id, label, confidence * 100, num_faces)
        jobs[job_id].update({
            'status': 'processing_video',
            'result': label,
            'confidence': round(confidence * 100, 2),
            'score': round(avg_score, 4),
            'num_faces': num_faces,
            'faces_detail': faces_detail,
            'video_url': f'/uploads/{unique_name}',
        })

        # Now generate processed video (results already visible to client)
        logger.info('[Job %s] Starting video processing', job_id)
        processed_name = f"processed_{unique_name}"
        processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_name)
        create_processed_video(filepath, processed_path)

        logger.info('[Job %s] Video processing done', job_id)
        jobs[job_id].update({
            'status': 'done',
            'processed_url': f'/uploads/{processed_name}',
        })
    except Exception as e:
        logger.error('[Job %s] Error: %s', job_id, e)
        jobs[job_id].update({'status': 'done', 'error': str(e)})


@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded.'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: mp4, avi, mov, mkv, wmv'}), 400

    cleanup_old_uploads()

    ext = secure_filename(file.filename).rsplit('.', 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(filepath)
    logger.info('Video uploaded: %s (%s)', file.filename, unique_name)

    # Re-encode upload to H.264 so browser can play it
    logger.info('Re-encoding uploaded video to H.264')
    reencode_to_h264(filepath)

    job_id = uuid.uuid4().hex
    logger.info('Created job %s for %s', job_id, unique_name)
    jobs[job_id] = {'status': 'uploading', 'video_url': f'/uploads/{unique_name}'}

    thread = threading.Thread(target=process_video_job, args=(job_id, filepath, unique_name))
    thread.start()

    return jsonify({'job_id': job_id, 'video_url': f'/uploads/{unique_name}'})


@app.route('/status/<job_id>')
def job_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(job)


if __name__ == '__main__':
    logger.info('Starting Flask server on http://0.0.0.0:5000')
    app.run(debug=True, host='0.0.0.0', port=5000)

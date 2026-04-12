import os
import base64
import math
import logging
import subprocess
import cv2
import numpy as np
import imageio_ffmpeg
from mtcnn import MTCNN
from ultralytics import YOLO
from flask import Flask
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import keras.src.layers.normalization.batch_normalization as _bn_module

import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stderr
)

# Monkey-patch BatchNormalization to accept legacy renorm kwargs
_OrigBN = _bn_module.BatchNormalization
_orig_bn_init = _OrigBN.__init__


def _patched_bn_init(self, *args, **kwargs):
    kwargs.pop('renorm', None)
    kwargs.pop('renorm_clipping', None)
    kwargs.pop('renorm_momentum', None)
    _orig_bn_init(self, *args, **kwargs)


_OrigBN.__init__ = _patched_bn_init
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB limit
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'tmp_checkpoint', 'best_model.keras')
logger.info('Loading model from %s', MODEL_PATH)
model = load_model(MODEL_PATH)
logger.info('Model loaded successfully')
INPUT_SIZE = 224
MIN_FACE_SIZE = 90  # same as 02-prepare_fake_real_dataset.py

# Initialize MTCNN face detector (same as training pipeline 01-crop_faces_with_mtcnn.py)
logger.info('Initializing MTCNN face detector')
mtcnn_detector = MTCNN()
logger.info('MTCNN face detector ready')

# Initialize YOLO face detector (for processed video overlay only)
logger.info('Initializing YOLO face detector')
FACE_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'yolov8n-face.pt')
face_detector = YOLO(FACE_MODEL_PATH)
logger.info('YOLO face detector ready')

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


def scale_frame(frame):
    """Scale frame exactly like 00-convert_video_to_image.py"""
    h, w = frame.shape[:2]
    if w < 300:
        scale_ratio = 2
    elif w > 1900:
        scale_ratio = 0.33
    elif w > 1000:
        scale_ratio = 0.5
    else:
        scale_ratio = 1
    if scale_ratio != 1:
        new_w = int(w * scale_ratio)
        new_h = int(h * scale_ratio)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return frame


def extract_faces_from_video(video_path):
    """Extract faces using MTCNN — matching training pipeline (01-crop_faces_with_mtcnn.py)."""
    logger.info('Extracting faces from video: %s', video_path)
    faces = []
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    if frame_rate == 0:
        logger.warning('Could not read frame rate from video')
        cap.release()
        return faces

    while cap.isOpened():
        frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % math.floor(frame_rate) == 0:
            # Step 1: Scale frame (same as 00-convert_video_to_image.py)
            frame = scale_frame(frame)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = image_rgb.shape[:2]

            # Step 2: MTCNN face detection (same as 01-crop_faces_with_mtcnn.py)
            results = mtcnn_detector.detect_faces(image_rgb)
            num_faces = len(results)

            for result in results:
                bounding_box = result['box']
                confidence = result['confidence']
                # Same logic as training: if single face keep it, if multiple only keep > 0.95
                if num_faces < 2 or confidence > 0.95:
                    bx, by, bw, bh = bounding_box
                    margin_x = bw * 0.3
                    margin_y = bh * 0.3
                    x1 = int(max(0, bx - margin_x))
                    x2 = int(min(w, bx + bw + margin_x))
                    y1 = int(max(0, by - margin_y))
                    y2 = int(min(h, by + bh + margin_y))
                    crop = image_rgb[y1:y2, x1:x2]
                    # Step 3: Filter small faces (same as 02-prepare_fake_real_dataset.py MIN_IMAGE_SIZE=90)
                    if crop.shape[0] < MIN_FACE_SIZE or crop.shape[1] < MIN_FACE_SIZE:
                        continue
                    if crop.size > 0:
                        crop_resized = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE))
                        faces.append(crop_resized)

    cap.release()
    logger.info('Face extraction complete — %d faces found', len(faces))
    return faces


def create_processed_video(video_path, output_path, face_scores=None):
    """Re-encode video with face bounding boxes (detection only, no labels)."""
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

    # Only run detection every N frames; reuse cached overlays in between
    detect_interval = max(1, int(fps // 3))  # ~3 detections per second
    frame_count = 0
    cached_boxes = []  # list of (x1, y1, x2, y2)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % detect_interval == 0:
            results = face_detector(frame, verbose=False)[0]
            cached_boxes = []

            for box in results.boxes:
                if box.conf[0] > 0.5:
                    bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                    cached_boxes.append((max(0, bx1), max(0, by1), bx2, by2))

        # Draw face boxes on every frame (green color, no labels)
        for (x1, y1, x2, y2) in cached_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

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

    face_array = preprocess_input(np.array(faces, dtype='float32'))
    predictions = model.predict(face_array, verbose=0)
    flat_preds = predictions.flatten()
    # Use top-K mean: average the top 30% of predictions (at least 3)
    # Rationale: real videos have many high-confidence real frames; fake videos have NONE
    sorted_desc = np.sort(flat_preds)[::-1]  # highest first
    k = max(3, int(len(sorted_desc) * 0.3))
    top_k = sorted_desc[:k]
    avg_prediction = float(np.mean(top_k))
    # Write diagnostics to file
    diag_path = os.path.join(os.path.dirname(__file__), 'diag_log.txt')
    with open(diag_path, 'a') as f:
        f.write(f'Raw predictions: min={float(np.min(predictions)):.4f}, max={float(np.max(predictions)):.4f}, top{k}_mean={avg_prediction:.4f}, mean={float(np.mean(predictions)):.4f}\n')
        f.write(f'All scores (sorted desc): {sorted_desc.tolist()}\n')
        f.write(f'Top-{k} used: {top_k.tolist()}\n')
        f.write(f'Num faces: {len(faces)}\n\n')
    logger.info('Raw predictions: min=%.4f, max=%.4f, top%d_mean=%.4f, mean=%.4f, n=%d',
                float(np.min(predictions)), float(np.max(predictions)),
                k, avg_prediction, float(np.mean(flat_preds)), len(flat_preds))

    # Build per-face details (up to 5 faces sorted by relevance)
    is_real = avg_prediction > 0.5
    # Sort face indices by score: highest first for REAL, lowest first for FAKE
    sorted_indices = np.argsort(flat_preds)[::-1] if is_real else np.argsort(flat_preds)
    indices = sorted_indices[:5].tolist()

    faces_detail = []
    for i in indices:
        faces_detail.append({
            'thumbnail': face_to_base64(faces[i]),
            'score': float(predictions[i][0])
        })

    logger.info('Prediction complete — avg score: %.4f, faces: %d', avg_prediction, len(faces))
    return avg_prediction, len(faces), faces_detail


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


from route import routes
app.register_blueprint(routes)


if __name__ == '__main__':
    logger.info('Starting Flask server on http://0.0.0.0:5000')
    app.run(debug=False, host='0.0.0.0', port=5000)

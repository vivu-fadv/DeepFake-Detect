import os
import logging
import uuid
import threading
from flask import Blueprint, request, render_template, send_from_directory, jsonify
from werkzeug.utils import secure_filename

logger = logging.getLogger(__name__)

routes = Blueprint('routes', __name__)


def _get_app_deps():
    """Import app-level objects to avoid circular imports."""
    from app import (
        app, jobs, allowed_file, cleanup_old_uploads,
        extract_faces_from_video, predict_deepfake,
        create_processed_video, reencode_to_h264
    )
    return app, jobs, allowed_file, cleanup_old_uploads, \
        extract_faces_from_video, predict_deepfake, \
        create_processed_video, reencode_to_h264


@routes.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@routes.route('/uploads/<filename>')
def uploaded_video(filename):
    from app import app
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, mimetype='video/mp4')


def process_video_job(job_id, filepath, unique_name):
    """Background worker: extract faces, predict, create processed video."""
    app, jobs, _, _, extract_faces_from_video, predict_deepfake, \
        create_processed_video, _ = _get_app_deps()
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


@routes.route('/predict', methods=['POST'])
def predict():
    app, jobs, allowed_file, cleanup_old_uploads, _, _, _, reencode_to_h264 = _get_app_deps()

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

    logger.info('Re-encoding uploaded video to H.264')
    reencode_to_h264(filepath)

    job_id = uuid.uuid4().hex
    logger.info('Created job %s for %s', job_id, unique_name)
    jobs[job_id] = {'status': 'uploading', 'video_url': f'/uploads/{unique_name}'}

    thread = threading.Thread(target=process_video_job, args=(job_id, filepath, unique_name))
    thread.start()

    return jsonify({'job_id': job_id, 'video_url': f'/uploads/{unique_name}'})


@routes.route('/status/<job_id>')
def job_status(job_id):
    from app import jobs
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(job)

FROM python:3.10-slim

# System dependencies for OpenCV, ffmpeg, and video processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (required by HF Spaces)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY App/ .

# Copy model files
COPY tmp_checkpoint/best_model.keras models/best_model.keras

# Create uploads directory with proper permissions
RUN mkdir -p uploads && chown -R user:user /app

USER user

EXPOSE 7860

CMD ["python", "app.py"]

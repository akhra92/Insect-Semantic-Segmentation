# Use Python 3.9 slim base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libopenblas-dev \
    gfortran \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2 \
    segmentation-models-pytorch==0.3.3 \
    albumentations==1.3.1 \
    opencv-python==4.8.1.78 \
    matplotlib==3.7.2 \
    numpy==1.24.3 \
    Pillow==10.0.0 \
    tqdm==4.66.1 \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    python-multipart==0.0.6

# Copy all project files
COPY . .

# Make both entrypoint scripts executable
RUN chmod +x docker-entrypoint.sh

# Create directories for outputs
RUN mkdir -p saved_models inference_results

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port for API mode
EXPOSE 8000

# Use new flexible entrypoint script
ENTRYPOINT ["./docker-entrypoint.sh"]

# Default to training mode
CMD ["train"]
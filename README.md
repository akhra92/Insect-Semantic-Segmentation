# 🐛 Insect Semantic Segmentation Project

A deep learning project for semantic segmentation of insects using PyTorch and U-Net architecture with ResNet34 encoder. This project includes training, inference, API server, and interactive Streamlit demo capabilities.
NOTE: The Dockerfile should be modified to be able to run this project in GPU!

## 📋 Table of Contents
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
  - [Docker Usage](#-docker-usage)
  - [Local Development](#-local-development)
- [API Usage](#-api-usage)
- [Streamlit Demo](#-streamlit-demo)
- [Training](#-training)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)


## 📁 Project Structure

```
Segmentation_Project/
├── 📄 README.md                     # This file
├── 🐳 Dockerfile                    # Docker configuration
├── ⚙️  docker-entrypoint.sh         # Flexible entrypoint script
├── 🧠 main.py                       # Main training script
├── 🏗️  train.py                     # Training logic
├── 📊 test.py                       # Inference logic
├── 📁 dataset.py                    # Dataset handling
├── 🛠️  utils.py                     # Utility functions and metrics
├── 🌐 api.py                        # FastAPI server
├── 🎨 streamlit_demo.py             # Streamlit demo app
├── 📋 requirements-streamlit.txt    # Streamlit dependencies
├── 💾 saved_models/                 # Trained models directory
├── 📈 inference_results/            # Inference outputs
└── 📂 datasets/                     # Dataset directory
    └── insect_semantic_segmentation/
        └── arthropodia/
            ├── images/              # Input images
            └── labels/              # Ground truth masks
```

## 🛠 Installation

### Prerequisites
- Python 3.9+
- Docker (optional)
- 4GB+ RAM
- CUDA GPU (optional, for faster training)

### Docker Installation (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd Segmentation_Project

# Build Docker image
docker build -t segmentation_project .
```

### Local Installation

```bash
# Clone the repository
git clone <repository-url>
cd Segmentation_Project

# Install dependencies
pip install torch==2.0.1 torchvision==0.15.2
pip install segmentation-models-pytorch==0.3.3
pip install albumentations==1.3.1 opencv-python==4.8.1.78
pip install matplotlib==3.7.2 numpy==1.24.3 Pillow==10.0.0
pip install tqdm==4.66.1 fastapi==0.104.1 uvicorn==0.24.0
```

## 🚀 Usage

### 🐳 Docker Usage

#### Training Mode
```bash
# Basic training
docker run --rm -v $(pwd):/app segmentation_project

# Custom parameters
docker run --rm -v $(pwd):/app \
  -e EPOCHS=20 \
  -e BATCH_SIZE=16 \
  -e LEARNING_RATE=0.001 \
  segmentation_project train
```

#### API Mode
```bash
# Start API server
docker run --rm -p 8000:8000 -v $(pwd):/app segmentation_project api

# Run inference only
docker run --rm -v $(pwd):/app segmentation_project inference
```

#### Help
```bash
# Show all available options
docker run --rm segmentation_project help
```

### 💻 Local Development

#### Training
```bash
python main.py -bs 16 -lr 0.001 -ep 10 -d cpu -nw 2
```

#### API Server
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

#### Streamlit Demo
```bash
pip install -r requirements-streamlit.txt
streamlit run streamlit_demo.py
```

### API Endpoints

- **GET** `/` - API information
- **GET** `/health` - Health check
- **GET** `/docs` - Interactive API documentation
- **GET** `/model/info` - Model information
- **POST** `/predict` - Single image segmentation
- **POST** `/predict/batch` - Batch image segmentation


### Cloud Deployment Options

#### 🎈 Streamlit Cloud (Free & Easy)
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository and deploy


# Self-hosted
streamlit run streamlit_demo.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true


## 🐛 Troubleshooting

### Common Issues

#### Docker Issues
```bash
# Permission denied
chmod +x docker-entrypoint.sh

# Port already in use
docker run -p 8001:8000 segmentation_project api

# Out of memory
docker run -e BATCH_SIZE=4 -e NUM_WORKERS=0 segmentation_project train
```

#### Model Issues
```bash
# Model not found
# Ensure saved_models/insect_best_model.pt exists
ls -la saved_models/

# CUDA errors (M1 Mac)
# Use CPU mode
docker run -e DEVICE=cpu segmentation_project train
```

#### API Issues
```bash
# Check API health
curl http://localhost:8000/health

# View API docs
open http://localhost:8000/docs

# Check logs
docker logs <container_id>
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Submit Pull Request
# 🐛 Insect Semantic Segmentation Project

A deep learning project for semantic segmentation of insects using PyTorch and U-Net architecture with ResNet34 encoder. This project includes training, inference, API server, and interactive Streamlit demo capabilities.
NOTE: The Dockerfile should be modified to be able to run this project in GPU!
[Try Demo Here](share.streamlit.io)

## 📋 Table of Contents
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
  - [Docker Usage](#-docker-usage)
  - [Local Development](#-local-development)
- [API Usage](#-api-usage)
- [Streamlit Demo](#-streamlit-demo)
- [Training](#-training)
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
├── 📋 requirements.txt              # Streamlit Cloud dependencies (CPU torch)
├── 📦 packages.txt                  # apt packages for Streamlit Cloud
├── ⚙️  .streamlit/config.toml       # Streamlit app config
├── 💾 saved_models/                 # Trained models directory
├── 📈 inference_results/            # Inference outputs
└── 📂 datasets/                     # Dataset directory
    └── insect_semantic_segmentation/
        └── arthropodia/
            ├── images/              # Input images
            └── labels/              # Ground truth masks
```
**NOTE:** Download the dataset using **downloader.py** file
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
cd <repository-folder>

# Build Docker image
docker build -t segmentation_project .
```

### Local Installation

```bash
# Clone the repository
git clone <repository-url>
cd <repository-folder>

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
pip install -r requirements.txt
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

#### 🎈 Streamlit Community Cloud

The repo is pre-configured for Streamlit Cloud:

| File | Purpose |
|------|---------|
| `requirements.txt` | Python deps — pinned to **CPU-only** PyTorch wheels |
| `packages.txt` | apt packages needed by OpenCV (see note below) |
| `.streamlit/config.toml` | Upload limit + theme |

**Steps**

1. Train a model (`python main.py ...`) — it is saved to `saved_models/insect_best_model.pt`.
2. Host the `.pt` file somewhere with a direct download link (Hugging Face Hub or a
   GitHub Release). Model weights are ~97 MB and are excluded by `.gitignore`, so they
   are **not** committed to the repo.
3. Push the repo to GitHub.
4. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → pick this repo.
5. In **Advanced settings**:
   - Main file path: `streamlit_demo.py`
   - Python version: **3.12**
   - Secrets:
     ```toml
     MODEL_URL = "https://huggingface.co/<user>/<repo>/resolve/main/insect_best_model.pt"
     ```
6. Deploy. The app downloads the weights on first boot and caches them.

Without `MODEL_URL`, the app still runs — users just have to upload a `.pt` file
via the sidebar.

> **Note on `packages.txt`.** Streamlit Cloud runs Debian **trixie** (13), where GLib was
> renamed `libglib2.0-0` → `libglib2.0-0t64` as part of the 64-bit `time_t` transition.
> Requesting the old name makes apt fall back to the image's stale `bullseye-security`
> source, whose Debian 11 build depends on `libffi7`/`libpcre3` — neither exists in trixie,
> so the build dies with *"held broken packages"*. Keep the `t64` suffix. Also note that
> `packages.txt` is parsed as one package name per line and does **not** support `#`
> comments — a comment line will be treated as a package and fail the install.
>
> GLib is needed because `albumentations` imports OpenCV, and the wheel links
> `libgthread-2.0` / `libglib-2.0` without bundling them.

#### Self-hosted

```bash
streamlit run streamlit_demo.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Submit Pull Request
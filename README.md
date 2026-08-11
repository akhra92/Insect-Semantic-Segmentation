# 🐛 Insect Semantic Segmentation Project

A deep learning project for semantic segmentation of insects using PyTorch and U-Net architecture with ResNet34 encoder. This project includes training, inference, API server, and interactive Streamlit demo capabilities.
NOTE: The Dockerfile should be modified to be able to run this project in GPU!
[Try Demo Here->](https://insect-semantic-segmentation.streamlit.app/)

## 📋 Table of Contents
- [Project Structure](#-project-structure)
- [Results](#-results)
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
├── 🖼️  assets/                      # Bundled sample image for the demo
├── 💾 saved_models/                 # Trained models directory
├── 📉 plots/                        # Training curves (loss, PA, mIoU)
├── 📈 inference_results/            # Inference outputs
└── 📂 datasets/                     # Dataset directory
    └── insect_semantic_segmentation/
        └── arthropodia/
            ├── images/              # Input images
            └── labels/              # Ground truth masks
```
**NOTE:** Download the dataset using **downloader.py** file

## 📊 Results

U-Net with a ResNet34 encoder, trained for 9 epochs on the arthropodia dataset
(4,949 image/mask pairs, split 80/10/10 → 3,959 train / 494 val / 496 test) at
256×256, batch size 8, Adam @ 1e-3, cross-entropy loss. The checkpoint is selected
by best validation loss.

| Metric | Train (final) | Val (final) | Val @ saved checkpoint |
|:---|---:|---:|---:|
| Loss | 0.063 | 0.088 | 0.090 |
| Pixel Accuracy | 0.974 | 0.965 | 0.964 |
| mIoU | 0.850 | 0.799 | 0.802 |

<sub>Values read off the curves below, so they are approximate. The shipped checkpoint
is from **epoch 4**, not the last epoch: `Trainer` only re-saves when validation loss
improves by more than `thresh=0.005`, and every later epoch gained only ~0.002. The
five stagnant epochs that follow are what trip the `early_stop_thresh=5` guard and end
the run at 9 of the requested 10 epochs.</sub>

### 📉 Training Curves

| Loss | Pixel Accuracy | mIoU |
|:---:|:---:|:---:|
| ![Loss curve](plots/loss_curve.png) | ![Pixel accuracy curve](plots/pa_curve.png) | ![mIoU curve](plots/iou_curve.png) |

Both losses fall steeply for the first three epochs. From epoch 4 onward the
validation curves flatten (loss ≈0.088, mIoU ≈0.80) while the training curves keep
improving — the ~0.05 mIoU train/val gap at the end is mild overfitting. Validation
mIoU is the noisiest of the three, dipping to 0.726 at epoch 3 before peaking at
≈0.817 at epoch 6.

### 🎯 Inference on Test Images

Five held-out test samples — input, ground-truth mask, and predicted mask:

![Inference results](inference_results/inference_visualization.png)

The model localises the insect reliably across varied backgrounds (leaf, bark,
perforated metal, moss). Predicted masks are noticeably smoother than the labels:
fine appendages such as legs and antennae get absorbed into the body blob, which is
expected when predicting at 256×256 and the main source of the residual mIoU gap.

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


Without `MODEL_URL`, the app still runs — users just have to upload a `.pt` file
via the sidebar.

#### Self-hosted

```bash
streamlit run streamlit_demo.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true
```
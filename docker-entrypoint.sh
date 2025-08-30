#!/bin/bash

echo "=== Segmentation Project Container Started ==="

# Create necessary directories
mkdir -p saved_models inference_results

# Set default parameters if not provided
BATCH_SIZE=${BATCH_SIZE:-8}
LEARNING_RATE=${LEARNING_RATE:-0.001}
EPOCHS=${EPOCHS:-10}
NUM_WORKERS=${NUM_WORKERS:-2}
DEVICE=${DEVICE:-cpu}
API_PORT=${API_PORT:-8000}

# Check the mode (first argument)
MODE=${1:-train}

case $MODE in
    "train")
        echo "=== TRAINING MODE ==="
        echo "Training Parameters:"
        echo "  Batch Size: $BATCH_SIZE"
        echo "  Learning Rate: $LEARNING_RATE"
        echo "  Epochs: $EPOCHS"
        echo "  Workers: $NUM_WORKERS"
        echo "  Device: $DEVICE"
        
        # Check if dataset exists
        if [ ! -d "datasets/insect_semantic_segmentation/arthropodia/images" ]; then
            echo "WARNING: Dataset not found at datasets/insect_semantic_segmentation/arthropodia/images"
            echo "Please mount your dataset or check the path."
            echo "Expected structure:"
            echo "  datasets/insect_semantic_segmentation/arthropodia/"
            echo "    ├── images/"
            echo "    └── labels/"
        fi
        
        echo "=== Starting Training ==="
        python main.py \
            -bs $BATCH_SIZE \
            -lr $LEARNING_RATE \
            -ep $EPOCHS \
            -d $DEVICE \
            -nw $NUM_WORKERS \
            "${@:2}"  # Pass any additional arguments
        
        echo "=== Training Complete ==="
        ;;
        
    "api")
        echo "=== API MODE ==="
        echo "API Parameters:"
        echo "  Port: $API_PORT"
        echo "  Device: $DEVICE"
        echo "  Workers: auto"
        
        # Check if model exists
        if [ ! -f "saved_models/insect_best_model.pt" ]; then
            echo "WARNING: Trained model not found at saved_models/insect_best_model.pt"
            echo "API will start but predictions will fail until model is available."
            echo "Please train the model first or mount a directory with the trained model."
        else
            echo "Found trained model at saved_models/insect_best_model.pt"
        fi
        
        echo "=== Starting API Server ==="
        echo "API will be available at: http://0.0.0.0:$API_PORT"
        echo "API Documentation at: http://0.0.0.0:$API_PORT/docs"
        
        # Start the API server
        uvicorn api:app --host 0.0.0.0 --port $API_PORT "${@:2}"
        ;;
        
    "inference")
        echo "=== INFERENCE MODE ==="
        echo "Inference Parameters:"
        echo "  Device: $DEVICE"
        
        # Check if model exists
        if [ ! -f "saved_models/insect_best_model.pt" ]; then
            echo "ERROR: Trained model not found at saved_models/insect_best_model.pt"
            echo "Please train the model first or mount a directory with the trained model."
            exit 1
        fi
        
        echo "=== Running Inference ==="
        python -c "
from test import Inference
from dataset import get_dataloaders
from utils import get_transforms

transform = get_transforms(img_size=256)
_, _, test_loader, _ = get_dataloaders(
    root='./datasets/insect_semantic_segmentation/arthropodia',
    transform=transform,
    batch_size=1,
    num_workers=0
)

inference_runner = Inference(
    model_path='saved_models/insect_best_model.pt',
    device='$DEVICE'
)
inference_runner.run(dl=test_loader)
"
        echo "=== Inference Complete ==="
        ;;
        
    "bash"|"shell")
        echo "=== INTERACTIVE SHELL MODE ==="
        echo "Available commands:"
        echo "  python main.py -d cpu -bs 8  # Run training"
        echo "  uvicorn api:app --host 0.0.0.0 --port 8000  # Start API"
        echo "  ls saved_models/  # Check saved models"
        exec bash "${@:2}"
        ;;
        
    "help"|"--help"|"-h")
        echo "=== USAGE HELP ==="
        echo "docker run [options] segmentation_project [MODE] [ARGS...]"
        echo ""
        echo "Available modes:"
        echo "  train       - Run training (default)"
        echo "  api         - Start API server"
        echo "  inference   - Run inference on test set"
        echo "  bash        - Interactive shell"
        echo "  help        - Show this help"
        echo ""
        echo "Environment variables for training:"
        echo "  BATCH_SIZE     - Batch size (default: 8)"
        echo "  LEARNING_RATE  - Learning rate (default: 0.001)"
        echo "  EPOCHS         - Number of epochs (default: 10)"
        echo "  NUM_WORKERS    - DataLoader workers (default: 2)"
        echo "  DEVICE         - Device to use (default: cpu)"
        echo ""
        echo "Environment variables for API:"
        echo "  API_PORT       - API port (default: 8000)"
        echo "  DEVICE         - Device to use (default: cpu)"
        echo ""
        echo "Examples:"
        echo "  # Training"
        echo "  docker run -v \$(pwd):/app segmentation_project train"
        echo "  docker run -e EPOCHS=20 -e BATCH_SIZE=16 segmentation_project train"
        echo ""
        echo "  # API"
        echo "  docker run -p 8000:8000 -v \$(pwd):/app segmentation_project api"
        echo "  docker run -p 5000:5000 -e API_PORT=5000 segmentation_project api"
        echo ""
        echo "  # Interactive"
        echo "  docker run -it -v \$(pwd):/app segmentation_project bash"
        ;;
        
    *)
        echo "ERROR: Unknown mode '$MODE'"
        echo "Available modes: train, api, inference, bash, help"
        echo "Use 'docker run segmentation_project help' for more information"
        exit 1
        ;;
esac
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
from utils import get_transforms
import io
import os
from typing import List

app = FastAPI(
    title="Insect Segmentation API",
    description="API for insect semantic segmentation using deep learning",
    version="1.0.0"
)

# Global variables for model
model = None
transform = None
device = "cpu"

def load_model():
    """Load the trained segmentation model"""
    global model, transform
    
    model_path = "saved_models/insect_best_model.pt"
    
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=503, 
            detail=f"Model not found at {model_path}. Please train the model first."
        )
    
    # Initialize model
    model = smp.Unet(encoder_name="resnet34", encoder_depth=5, classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Initialize transforms
    transform = get_transforms(img_size=256)
    
    print(f"Model loaded successfully from {model_path}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        load_model()
    except Exception as e:
        print(f"Warning: Could not load model on startup: {e}")
        print("Model will be loaded on first prediction request")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Insect Segmentation API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Upload image for segmentation",
            "/health": "GET - Health check",
            "/model/info": "GET - Model information"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_loaded = model is not None
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "device": device
    }

@app.get("/model/info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "architecture": "U-Net with ResNet34 encoder",
        "input_size": "256x256",
        "classes": 2,
        "device": device
    }

@app.post("/predict")
async def predict_segmentation(file: UploadFile = File(...)):
    """
    Predict segmentation mask for uploaded image
    
    Args:
        file: Image file (jpg, png, etc.)
        
    Returns:
        JSON with prediction results
    """
    global model, transform
    
    # Load model if not already loaded
    if model is None or transform is None:
        try:
            load_model()
        except Exception as e:
            raise HTTPException(status_code=503, detail=str(e))
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="File must be an image"
        )
    
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        original_size = image.size
        
        # Convert to numpy array
        image_np = np.array(image)
        
        # Apply transforms
        transformed = transform(image=image_np)
        input_tensor = transformed["image"].unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            prediction = model(input_tensor)
            probabilities = torch.softmax(prediction, dim=1)
            mask = torch.argmax(prediction, dim=1).squeeze().cpu().numpy()
        
        # Convert mask to binary (0 or 1)
        binary_mask = (mask > 0).astype(int)
        
        # Calculate statistics
        total_pixels = binary_mask.size
        foreground_pixels = np.sum(binary_mask)
        background_pixels = total_pixels - foreground_pixels
        
        return {
            "success": True,
            "prediction": {
                "mask": binary_mask.tolist(),
                "shape": list(binary_mask.shape)
            },
            "statistics": {
                "original_image_size": list(original_size),
                "processed_image_size": list(binary_mask.shape),
                "total_pixels": int(total_pixels),
                "foreground_pixels": int(foreground_pixels),
                "background_pixels": int(background_pixels),
                "foreground_percentage": float(foreground_pixels / total_pixels * 100)
            },
            "confidence": {
                "max_probability": float(torch.max(probabilities).item()),
                "min_probability": float(torch.min(probabilities).item())
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing image: {str(e)}"
        )

@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict segmentation masks for multiple images
    
    Args:
        files: List of image files
        
    Returns:
        JSON with batch prediction results
    """
    if len(files) > 10:  # Limit batch size
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 files allowed per batch"
        )
    
    results = []
    
    for i, file in enumerate(files):
        try:
            # Process each file individually
            result = await predict_segmentation(file)
            results.append({
                "file_index": i,
                "filename": file.filename,
                "result": result
            })
        except Exception as e:
            results.append({
                "file_index": i,
                "filename": file.filename,
                "error": str(e)
            })
    
    return {
        "batch_results": results,
        "total_files": len(files),
        "successful_predictions": len([r for r in results if "error" not in r])
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
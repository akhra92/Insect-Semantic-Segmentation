import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageOps
import segmentation_models_pytorch as smp
from utils import get_transforms
import matplotlib.pyplot as plt
import cv2
import io
import os
from typing import Optional

# Page configuration
st.set_page_config(
    page_title="Insect Segmentation Demo",
    page_icon="🐛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-container {
        border: 2px solid #e6e9ef;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(model_path: str = "saved_models/insect_best_model.pt") -> Optional[torch.nn.Module]:
    """Load the trained segmentation model"""
    try:
        if not os.path.exists(model_path):
            st.error(f"❌ Model not found at {model_path}")
            st.info("💡 Please ensure you have trained the model or upload a trained model file.")
            return None
        
        # Load model
        model = smp.Unet(encoder_name="resnet34", encoder_depth=5, classes=2)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        
        st.success("✅ Model loaded successfully!")
        return model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

@st.cache_data
def get_image_transforms():
    """Get image transforms for preprocessing"""
    return get_transforms(img_size=256)

def predict_segmentation(model: torch.nn.Module, image: Image.Image, transform) -> tuple:
    """Run segmentation prediction on an image"""
    try:
        # Convert PIL image to numpy array
        image_np = np.array(image.convert("RGB"))
        original_size = image.size
        
        # Apply transforms
        transformed = transform(image=image_np)
        input_tensor = transformed["image"].unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            prediction = model(input_tensor)
            probabilities = torch.softmax(prediction, dim=1)
            mask = torch.argmax(prediction, dim=1).squeeze().cpu().numpy()
        
        # Convert to binary mask (0 or 1)
        binary_mask = (mask > 0).astype(np.uint8)
        
        # Calculate statistics
        total_pixels = binary_mask.size
        foreground_pixels = np.sum(binary_mask)
        background_pixels = total_pixels - foreground_pixels
        
        return {
            "mask": binary_mask,
            "probabilities": probabilities.cpu().numpy(),
            "original_size": original_size,
            "processed_size": binary_mask.shape,
            "total_pixels": total_pixels,
            "foreground_pixels": foreground_pixels,
            "background_pixels": background_pixels,
            "foreground_percentage": (foreground_pixels / total_pixels) * 100
        }
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def create_overlay_visualization(original_image: Image.Image, mask: np.ndarray, alpha: float = 0.5):
    """Create an overlay visualization of the original image and predicted mask"""
    # Resize mask to match original image size
    original_np = np.array(original_image)
    mask_resized = cv2.resize(mask.astype(np.uint8), original_image.size, interpolation=cv2.INTER_NEAREST)
    
    # Create colored mask (red for foreground)
    colored_mask = np.zeros_like(original_np)
    colored_mask[mask_resized == 1] = [255, 0, 0]  # Red for insects
    
    # Create overlay
    overlay = cv2.addWeighted(original_np, 1-alpha, colored_mask, alpha, 0)
    
    return Image.fromarray(overlay)

def main():
    # Header
    st.markdown('<h1 class="main-header">🐛 Insect Segmentation Demo</h1>', unsafe_allow_html=True)
    st.markdown("Upload an image of insects to get semantic segmentation results using deep learning!")
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h2 class="sub-header">⚙️ Configuration</h2>', unsafe_allow_html=True)
        
        # Model loading
        model_option = st.radio(
            "Model Source:",
            ["Load from saved_models/", "Upload model file"]
        )
        
        model = None
        if model_option == "Load from saved_models/":
            model = load_model()
        else:
            uploaded_model = st.file_uploader(
                "Upload trained model (.pt file)",
                type=['pt'],
                help="Upload your trained PyTorch model file"
            )
            if uploaded_model:
                # Save uploaded file temporarily
                with open("temp_model.pt", "wb") as f:
                    f.write(uploaded_model.read())
                model = load_model("temp_model.pt")
        
        # Visualization options
        st.markdown('<h3 class="sub-header">🎨 Visualization</h3>', unsafe_allow_html=True)
        overlay_alpha = st.slider("Overlay Transparency", 0.0, 1.0, 0.5, 0.1)
        show_probabilities = st.checkbox("Show Probability Heatmap", False)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📤 Upload Image</h2>', unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image containing insects for segmentation"
        )
        
        # Example images
        st.markdown("**Or try example images:**")
        example_images = {
            "🐛 Example 1": "https://via.placeholder.com/400x300/4CAF50/FFFFFF?text=Upload+Your+Image",
            "🦋 Example 2": "https://via.placeholder.com/400x300/FF9800/FFFFFF?text=Upload+Your+Image"
        }
        
        selected_example = st.selectbox("Select example:", ["None"] + list(example_images.keys()))
        
        # Display uploaded or example image
        image = None
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        elif selected_example != "None":
            st.info("💡 Example images are placeholders. Upload your own insect images for real segmentation!")
    
    with col2:
        st.markdown('<h2 class="sub-header">🎯 Prediction Results</h2>', unsafe_allow_html=True)
        
        if image is not None and model is not None:
            # Get transforms
            transform = get_image_transforms()
            
            # Run prediction
            with st.spinner("🔄 Running segmentation..."):
                result = predict_segmentation(model, image, transform)
            
            if result:
                # Display results
                mask = result["mask"]
                
                # Create visualizations
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                # Original image
                axes[0, 0].imshow(image)
                axes[0, 0].set_title("Original Image")
                axes[0, 0].axis('off')
                
                # Predicted mask
                axes[0, 1].imshow(mask, cmap='gray')
                axes[0, 1].set_title("Predicted Mask")
                axes[0, 1].axis('off')
                
                # Overlay
                overlay_img = create_overlay_visualization(image, mask, overlay_alpha)
                axes[1, 0].imshow(overlay_img)
                axes[1, 0].set_title("Overlay (Red = Insect)")
                axes[1, 0].axis('off')
                
                # Probability heatmap (if requested)
                if show_probabilities and len(result["probabilities"].shape) > 2:
                    prob_map = result["probabilities"][0, 1]  # Foreground probability
                    axes[1, 1].imshow(prob_map, cmap='hot', interpolation='nearest')
                    axes[1, 1].set_title("Foreground Probability")
                    axes[1, 1].axis('off')
                else:
                    axes[1, 1].text(0.5, 0.5, 'Probability\nHeatmap\n(Enable in sidebar)', 
                                   ha='center', va='center', transform=axes[1, 1].transAxes)
                    axes[1, 1].axis('off')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Statistics
                st.markdown('<h3 class="sub-header">📊 Statistics</h3>', unsafe_allow_html=True)
                
                col_stats1, col_stats2, col_stats3 = st.columns(3)
                
                with col_stats1:
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<h4>🎯 Foreground %</h4>'
                        f'<h2>{result["foreground_percentage"]:.1f}%</h2>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                with col_stats2:
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<h4>🔍 Total Pixels</h4>'
                        f'<h2>{result["total_pixels"]:,}</h2>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                with col_stats3:
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<h4>📐 Image Size</h4>'
                        f'<h2>{result["original_size"][0]}×{result["original_size"][1]}</h2>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                # Detailed metrics
                with st.expander("📈 Detailed Metrics"):
                    metrics_data = {
                        "Metric": ["Total Pixels", "Foreground Pixels", "Background Pixels", "Original Size", "Processed Size"],
                        "Value": [
                            f"{result['total_pixels']:,}",
                            f"{result['foreground_pixels']:,}",
                            f"{result['background_pixels']:,}",
                            f"{result['original_size'][0]} × {result['original_size'][1]}",
                            f"{result['processed_size'][0]} × {result['processed_size'][1]}"
                        ]
                    }
                    st.table(metrics_data)
                
                # Download results
                st.markdown('<h3 class="sub-header">💾 Download Results</h3>', unsafe_allow_html=True)
                
                col_dl1, col_dl2 = st.columns(2)
                
                with col_dl1:
                    # Download mask
                    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                    mask_buffer = io.BytesIO()
                    mask_img.save(mask_buffer, format='PNG')
                    
                    st.download_button(
                        label="📥 Download Mask",
                        data=mask_buffer.getvalue(),
                        file_name="segmentation_mask.png",
                        mime="image/png"
                    )
                
                with col_dl2:
                    # Download overlay
                    overlay_buffer = io.BytesIO()
                    overlay_img.save(overlay_buffer, format='PNG')
                    
                    st.download_button(
                        label="📥 Download Overlay",
                        data=overlay_buffer.getvalue(),
                        file_name="segmentation_overlay.png",
                        mime="image/png"
                    )
        
        elif image is not None and model is None:
            st.warning("⚠️ Please load a model first!")
        
        elif image is None:
            st.info("👆 Please upload an image to see predictions")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; margin-top: 2rem;'>
        <p>🚀 Built with Streamlit | 🧠 Powered by PyTorch & Segmentation Models</p>
        <p>📧 For questions or issues, please contact the development team</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
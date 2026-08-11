import streamlit as st
import torch
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
from utils import get_transforms
import matplotlib.pyplot as plt
import cv2
import io
import os
import tempfile
import urllib.request
from typing import Optional

DEFAULT_MODEL_PATH = "saved_models/insect_best_model.pt"
SAMPLE_IMAGE_PATH = "assets/sample_insect.jpg"

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


def get_config(key: str) -> Optional[str]:
    """Read a setting from Streamlit secrets, falling back to env vars."""
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        # No secrets.toml present (e.g. plain local run)
        pass
    return os.environ.get(key)


@st.cache_resource(show_spinner="⬇️ Downloading model weights...")
def download_model(url: str) -> Optional[str]:
    """Download model weights once per app instance and cache them on disk"""
    dest = os.path.join(tempfile.gettempdir(), "insect_model_remote.pt")
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "insect-segmentation-demo"})
        with urllib.request.urlopen(request) as response, open(dest, "wb") as f:
            f.write(response.read())
        return dest
    except Exception as e:
        st.error(f"❌ Could not download weights from MODEL_URL: {e}")
        return None


def build_model() -> torch.nn.Module:
    """Build the architecture without fetching ImageNet weights (they get overwritten anyway)"""
    return smp.Unet(encoder_name="resnet34", encoder_weights=None, encoder_depth=5, classes=2)


@st.cache_resource(show_spinner="🧠 Loading model...")
def load_model(model_path: str, _cache_key: float = 0.0) -> Optional[torch.nn.Module]:
    """Load the trained segmentation model"""
    try:
        if not os.path.exists(model_path):
            return None

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        # Support both bare state_dicts and {"state_dict": ...} style checkpoints
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        model = build_model()
        model.load_state_dict(checkpoint)
        model.eval()
        return model

    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None


def resolve_model() -> Optional[torch.nn.Module]:
    """Find weights from the repo, then from MODEL_URL, in that order"""
    if os.path.exists(DEFAULT_MODEL_PATH):
        return load_model(DEFAULT_MODEL_PATH, os.path.getmtime(DEFAULT_MODEL_PATH))

    model_url = get_config("MODEL_URL")
    if model_url:
        downloaded = download_model(model_url)
        if downloaded:
            return load_model(downloaded)
        return None

    st.warning(
        f"⚠️ No weights found at `{DEFAULT_MODEL_PATH}` and no `MODEL_URL` configured. "
        "Switch to **Upload model file** in the sidebar, or set `MODEL_URL` in the app secrets."
    )
    return None


@st.cache_data(show_spinner=False)
def load_sample_image() -> Optional[Image.Image]:
    """Load the bundled demo image used when the user has not uploaded one"""
    if not os.path.exists(SAMPLE_IMAGE_PATH):
        return None
    with Image.open(SAMPLE_IMAGE_PATH) as img:
        # convert() forces the lazy read, so nothing holds the file handle open
        return img.convert("RGB")


@st.cache_resource
def get_image_transforms():
    """Get image transforms for preprocessing"""
    return get_transforms(img_size=256)


def predict_segmentation(model: torch.nn.Module, image: Image.Image, transform) -> Optional[dict]:
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
        foreground_pixels = int(np.sum(binary_mask))
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
    # Force 3-channel RGB so grayscale/RGBA uploads do not break the blend
    original_rgb = original_image.convert("RGB")
    original_np = np.array(original_rgb)

    # Resize mask to match original image size (PIL size is (width, height), same as cv2 dsize)
    mask_resized = cv2.resize(mask.astype(np.uint8), original_rgb.size, interpolation=cv2.INTER_NEAREST)

    # Create colored mask (red for foreground)
    colored_mask = np.zeros_like(original_np)
    colored_mask[mask_resized == 1] = [255, 0, 0]  # Red for insects

    # Create overlay
    overlay = cv2.addWeighted(original_np, 1 - alpha, colored_mask, alpha, 0)

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
            ["Use bundled/remote weights", "Upload model file"]
        )

        model = None
        if model_option == "Use bundled/remote weights":
            model = resolve_model()
        else:
            uploaded_model = st.file_uploader(
                "Upload trained model (.pt file)",
                type=['pt'],
                help="Upload your trained PyTorch model file"
            )
            if uploaded_model:
                # Write to a temp file keyed on the upload, so a new file busts the cache
                temp_path = os.path.join(tempfile.gettempdir(), f"uploaded_{uploaded_model.name}")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_model.getbuffer())
                model = load_model(temp_path, os.path.getmtime(temp_path))

        if model is not None:
            st.success("✅ Model loaded")

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

        # Display the uploaded image, or fall back to the bundled sample
        image, is_sample = None, False
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
        else:
            image = load_sample_image()
            if image is not None:
                is_sample = True
                st.image(
                    image,
                    caption="Sample image — upload your own above to replace it",
                    use_container_width=True
                )

    with col2:
        st.markdown('<h2 class="sub-header">🎯 Prediction Results</h2>', unsafe_allow_html=True)

        if image is not None and model is not None:
            if is_sample:
                st.caption("Showing results for the bundled sample image.")

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
                plt.close(fig)

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
            # Only reachable when the bundled sample is missing from the deployment
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

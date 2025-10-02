import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import os
import requests
import hashlib

# Configuration  
CHECKPOINT_PATH = "https://drive.usercontent.google.com/download?id=1hr8cDHAJImLc6QNNa4fvQQHjdHoDyIj5&export=download&authuser=0&confirm=t"

@st.cache_resource
def load_trained_model():
    """Load trained BLIP model (cached for performance) - supports local file or Google Drive URL"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        st.info(f"🔧 Device: {device}")
        
        # Check if we should try to load real model
        try_real_model = st.checkbox("🚀 Try to load real AI model (from Google Drive)", value=False, key="load_model")
        
        if not try_real_model:
            st.warning("⚠️ Running in DEMO mode for stability")
            dummy_config = {
                'image_size': 256,
                'vit': 'large',
                'prompt': 'a 3d rendered car ',
                'max_length': 25,
                'min_length': 5
            }
            return None, dummy_config, device
        
        # User wants to try real model
        st.info("🔄 Attempting to load real AI model...")
        
        # Use hash of URL for cache filename
        url_hash = hashlib.md5(CHECKPOINT_PATH.encode()).hexdigest()
        cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'blip_ckpt')
        os.makedirs(cache_dir, exist_ok=True)
        local_ckpt = os.path.join(cache_dir, f'ckpt_{url_hash}.pth')
        
        if not os.path.exists(local_ckpt):
            st.info("📥 Downloading checkpoint from Google Drive...")
            
            # Simple download with progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            response = requests.get(CHECKPOINT_PATH, stream=True)
            response.raise_for_status()
            
            downloaded = 0
            with open(local_ckpt, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress_bar.progress(min(downloaded / (50 * 1024 * 1024), 1.0))  # Assume ~50MB
                        status_text.text(f"Downloaded: {downloaded // (1024*1024):.1f}MB")
            
            progress_bar.progress(1.0)
            status_text.text("✅ Download completed!")
            st.success(f"Checkpoint saved to cache!")
        else:
            st.info("✅ Checkpoint found in cache!")
        
        # Check if file is valid (not HTML)
        with open(local_ckpt, 'rb') as f:
            first_bytes = f.read(100)
            if b'<html>' in first_bytes.lower() or b'<!doctype' in first_bytes.lower():
                st.error("❌ Downloaded file is HTML (Google Drive redirect issue)")
                os.remove(local_ckpt)
                raise Exception("Invalid checkpoint file - HTML detected")
        
        # Try to import BLIP model
        try:
            import sys
            models_path = os.path.join(os.getcwd(), 'models')
            if models_path not in sys.path:
                sys.path.insert(0, models_path)
                
            from models.blip import blip_decoder
            st.success("✅ BLIP model imported successfully!")
        except ImportError as e:
            st.error(f"❌ Cannot import BLIP model: {e}")
            raise Exception(f"BLIP import failed: {e}")
        
        # Load checkpoint
        st.info("🔄 Loading checkpoint...")
        try:
            checkpoint = torch.load(local_ckpt, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(local_ckpt, map_location=device)
        
        config = checkpoint['config']
        st.success(f"✅ Config loaded: {config}")
        
        # Create model
        model = blip_decoder(
            pretrained='',
            image_size=config['image_size'],
            vit=config['vit'],
            prompt=config.get('prompt', 'a 3d rendered car ')
        )
        
        model.load_state_dict(checkpoint['model'])
        model.eval()
        model = model.to(device)
        
        st.success(f"🚀 Model loaded successfully on {device}!")
        return model, config, device
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("Falling back to DEMO mode...")
        dummy_config = {
            'image_size': 256,
            'vit': 'large',
            'prompt': 'a 3d rendered car ',
            'max_length': 25,
            'min_length': 5
        }
        return None, dummy_config, device if 'device' in locals() else torch.device('cpu')

def preprocess_image(image, config, device):
    """Preprocess image for BLIP model"""
    image_size = config.get('image_size', 384)
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor

def generate_caption_dummy(image, config):
    """Generate dummy caption for demo purposes"""
    import random
    
    # Car door status captions
    captions = [
        "a 3d rendered car with doors closed",
        "a 3d rendered car with front door open",
        "a 3d rendered car with rear door open", 
        "a 3d rendered car with multiple doors open",
        "a 3d rendered car with driver door open",
        "a 3d rendered car with passenger door open",
        "a 3d rendered silver car with doors closed",
        "a 3d rendered blue car with front door open",
        "a 3d rendered red car with rear door open"
    ]
    
    return random.choice(captions)

def generate_caption(model, image, config, device):
    """Generate caption using the loaded model"""
    if model is None:
        return generate_caption_dummy(image, config)
    
    try:
        with torch.no_grad():
            image_tensor = preprocess_image(image, config, device)
            
            # Generate caption
            caption = model.generate(
                image_tensor,
                sample=False,
                num_beams=3,
                max_length=config.get('max_length', 20),
                min_length=config.get('min_length', 5)
            )
            
            return caption[0]
            
    except Exception as e:
        st.error(f"❌ Error during caption generation: {str(e)}")
        st.info("Using dummy caption...")
        return generate_caption_dummy(image, config)

def main():
    st.set_page_config(
        page_title="3D Car Caption Generator",
        page_icon="🚗",
        layout="wide"
    )
    
    st.title("🚗 3D Car Caption Generator")
    st.markdown("Upload an image of a 3D car and get an AI-generated caption about its door status!")
    
    # Load model
    with st.spinner("Loading AI model..."):
        model, config, device = load_trained_model()
    
    # Create columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📸 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['png', 'jpg', 'jpeg'],
            help="Upload a 3D car image to generate a caption"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        st.header("🤖 AI Caption")
        
        if uploaded_file is not None:
            # Generate caption
            with st.spinner("Generating caption..."):
                caption = generate_caption(model, image, config, device)
            
            # Display results
            st.success("Caption generated!")
            st.write("**Generated Caption:**")
            st.info(f"📝 {caption}")
            
            # Show model info
            if model is None:
                st.warning("⚠️ Demo mode active - using dummy captions")
            else:
                st.success("🚀 Real AI model active")
            
            # Model configuration
            with st.expander("🔧 Model Configuration"):
                st.json(config)
        else:
            st.info("👆 Please upload an image to generate a caption")

if __name__ == "__main__":
    main()

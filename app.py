import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import os

# Configuration
CHECKPOINT_PATH = "https://drive.google.com/uc?id=1hr8cDHAJImLc6QNNa4fvQQHjdHoDyIj5&export=download"

@st.cache_resource
def load_trained_model():
    """Load trained BLIP model (cached for performance) - supports local file or Google Drive URL"""
    import torch
    import hashlib
    import urllib
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def is_url(path):
        return path.startswith('http://') or path.startswith('https://')

    # If checkpoint is a URL, download to cache
    ckpt_path = CHECKPOINT_PATH
    if is_url(CHECKPOINT_PATH):
        # Use hash of URL for cache filename
        url_hash = hashlib.md5(CHECKPOINT_PATH.encode()).hexdigest()
        cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'blip_ckpt')
        os.makedirs(cache_dir, exist_ok=True)
        local_ckpt = os.path.join(cache_dir, f'ckpt_{url_hash}.pth')
        if not os.path.exists(local_ckpt):
            try:
                import requests
                st.info(f"🔄 Downloading checkpoint from Google Drive...")
                
                # Get file size for progress bar
                response = requests.head(CHECKPOINT_PATH)
                total_size = int(response.headers.get('content-length', 0))
                
                # Download with progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with requests.get(CHECKPOINT_PATH, stream=True) as r:
                    r.raise_for_status()
                    downloaded = 0
                    with open(local_ckpt, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                if total_size > 0:
                                    progress = downloaded / total_size
                                    progress_bar.progress(progress)
                                    status_text.text(f"Downloaded: {downloaded // (1024*1024):.1f}MB / {total_size // (1024*1024):.1f}MB ({progress*100:.1f}%)")
                
                progress_bar.progress(1.0)
                status_text.text("✅ Download completed!")
                st.success(f"Checkpoint downloaded to {local_ckpt}")
                st.info(f"📁 Cache location: `{local_ckpt}`")
            except Exception as e:
                st.warning(f"⚠️ Could not download checkpoint: {str(e)}")
                dummy_config = {
                    'image_size': 256,
                    'vit': 'large',
                    'prompt': 'a 3d rendered car ',
                    'max_length': 25,
                    'min_length': 5
                }
                return None, dummy_config, device
        ckpt_path = local_ckpt

    if not os.path.exists(ckpt_path):
        st.warning("⚠️ Model checkpoint not found. Running in DEMO mode.")
        dummy_config = {
            'image_size': 256,
            'vit': 'large',
            'prompt': 'a 3d rendered car ',
            'max_length': 25,
            'min_length': 5
        }
        return None, dummy_config, device

    try:
        try:
            # Try import dengan handling fairscale missing
            import sys
            import importlib.util
            
            # Add models to Python path jika belum ada
            models_path = os.path.join(os.getcwd(), 'models')
            if models_path not in sys.path:
                sys.path.insert(0, models_path)
            
            # Check for fairscale and create mock if needed
            try:
                import fairscale
                st.info("✅ Fairscale available")
            except ImportError:
                st.warning("⚠️ Fairscale not available, creating mock...")
                # Create minimal fairscale mock
                import types
                fairscale_mock = types.ModuleType('fairscale')
                fairscale_nn = types.ModuleType('fairscale.nn')
                fairscale_nn.FusedLayerNorm = torch.nn.LayerNorm  # Fallback to standard LayerNorm
                fairscale_mock.nn = fairscale_nn
                sys.modules['fairscale'] = fairscale_mock
                sys.modules['fairscale.nn'] = fairscale_nn
                st.info("✅ Fairscale mock created")
            
            from models.blip import blip_decoder
            st.success("✅ BLIP model imported successfully!")
            
        except ImportError as import_err:
            st.error(f"⚠️ BLIP model import failed: {str(import_err)}")
            st.info("🔍 Debugging info:")
            
            # Debug info
            current_dir = os.getcwd()
            models_dir = os.path.join(current_dir, 'models')
            blip_file = os.path.join(models_dir, 'blip.py')
            
            st.write(f"**Current directory:** `{current_dir}`")
            st.write(f"**Models directory exists:** {os.path.exists(models_dir)}")
            st.write(f"**BLIP file exists:** {os.path.exists(blip_file)}")
            
            if os.path.exists(models_dir):
                files = os.listdir(models_dir)
                st.write(f"**Files in models/:** {files}")
            
            st.warning("⚠️ Running in DEMO mode due to import failure.")
            dummy_config = {
                'image_size': 256,
                'vit': 'large',
                'prompt': 'a 3d rendered car ',
                'max_length': 25,
                'min_length': 5
            }
            return None, dummy_config, device

        st.info(f"🔄 Loading checkpoint from: `{ckpt_path}`")
        # PyTorch 2.6+ requires weights_only=False for model checkpoints
        try:
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            # Fallback for older PyTorch versions
            checkpoint = torch.load(ckpt_path, map_location=device)
        st.success("✅ Checkpoint loaded successfully!")
        
        config = checkpoint['config']
        st.info(f"📋 Config loaded: {config}")
        
        model = blip_decoder(
            pretrained='',
            image_size=config['image_size'],
            vit=config['vit'],
            prompt=config.get('prompt', 'a 3d rendered car ')
        )
        st.info("🏗️ Model architecture created!")
        
        model.load_state_dict(checkpoint['model'])
        st.info("💾 Model weights loaded!")
        
        model.eval()
        model = model.to(device)
        st.success(f"🚀 Model ready on {device}!")
        
        return model, config, device
    except Exception as e:
        st.warning(f"⚠️ Could not load model: {str(e)}")
        st.info("Running in DEMO mode for UI preview")
        dummy_config = {
            'image_size': 256,
            'vit': 'large',
            'prompt': 'a 3d rendered car ',
            'max_length': 25,
            'min_length': 5
        }
        return None, dummy_config, device

def preprocess_image(image, image_size, device):
    """Preprocess uploaded image for model - DUMMY SAFE VERSION"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Define transform (same as training)
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        
        # Apply transform
        processed_image = transform(image).unsqueeze(0).to(device)
        return processed_image
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

def generate_caption_dummy(image_name="uploaded_image.jpg"):
    """Generate dummy caption for demo purposes"""
    import random
    
    dummy_captions = [
        "a 3d rendered car with closed doors",
        "a 3d rendered car with front left door open", 
        "a 3d rendered car with hood open",
        "a 3d rendered car with front doors open",
        "a 3d rendered car with rear doors open",
        "a 3d rendered car with all doors open"
    ]
    
    return random.choice(dummy_captions)

def generate_caption(model, image, config, num_beams=3):
    """Generate caption for image - WITH DUMMY FALLBACK"""
    if model is None:
        # DEMO MODE - Return dummy caption
        return generate_caption_dummy()
    
    try:
        with torch.no_grad():
            caption = model.generate(
                image, 
                sample=False, 
                num_beams=num_beams,
                max_length=config.get('max_length', 25),
                min_length=config.get('min_length', 5)
            )
            return caption[0]
    except Exception as e:
        st.error(f"Error generating caption: {e}")
        return generate_caption_dummy()

def main():
    # Page config
    st.set_page_config(
        page_title="3D Car Caption Generator",
        page_icon="🚗",
        layout="centered"
    )
    
    # Title and description
    st.title("🚗 3D Car Caption Generator")
    st.markdown("Upload an image of a 3D rendered car and get AI-generated caption describing door status!")
    
    # Load model (with dummy fallback)
    with st.spinner("Loading AI model..."):
        model, config, device = load_trained_model()
    
    # Show model status
    if model is None:
        st.info("🎭 **DEMO MODE ACTIVE** - UI Preview (Model not loaded)")
    else:
        st.success("✅ **AI MODEL LOADED** - Ready for real predictions!")
    
    # Display model info
    with st.expander("ℹ️ Model Information"):
        if model is not None:
            st.write(f"**Status:** ✅ Model Loaded Successfully")
        else:
            st.write(f"**Status:** 🎭 Demo Mode (Model Not Available)")
        
        st.write(f"**Checkpoint:** checkpoint_04.pth")
        st.write(f"**Image Size:** {config['image_size']}px")
        st.write(f"**Model Type:** {config['vit']}")
        st.write(f"**Prompt:** {config['prompt']}")
        st.write(f"**Device:** {device}")
        
        if model is None:
            st.warning("⚠️ Running in demo mode - predictions will be random examples")
    
    # File upload
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a car image...", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload a 3D rendered car image to get caption prediction"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 Uploaded Image")
            st.image(image, caption=f"Original size: {image.size}", use_column_width=True)
        
        with col2:
            st.subheader("🤖 AI Prediction")
            
            # Show prediction status
            if model is None:
                st.info("🎭 Demo mode - will generate example caption")
            else:
                st.info("✅ Real AI model - will generate actual prediction")
            
            # Generation settings
            with st.expander("⚙️ Generation Settings"):
                num_beams = st.slider("Number of beams", min_value=1, max_value=10, value=3)
                if model is None:
                    st.warning("Demo mode: Settings visible but not functional")
                else:
                    st.info("Higher beams = better quality but slower")
            
            # Generate caption button
            if st.button("🎯 Generate Caption", type="primary"):
                with st.spinner("Generating caption..."):
                    try:
                        if model is not None:
                            # Real model prediction
                            processed_image = preprocess_image(image, config['image_size'], device)
                            if processed_image is not None:
                                caption = generate_caption(model, processed_image, config, num_beams)
                            else:
                                caption = generate_caption_dummy()
                        else:
                            # Demo mode
                            import time
                            time.sleep(1)  # Simulate processing time
                            caption = generate_caption_dummy()
                        
                        # Display result
                        if model is not None:
                            st.success("Caption generated successfully!")
                        else:
                            st.success("Demo caption generated! (This is a sample result)")
                        
                        st.markdown(f"### 💬 Generated Caption:")
                        st.markdown(f"**{caption}**")
                        
                        # Additional info
                        if model is not None:
                            st.info(f"Generated using {num_beams} beams")
                        else:
                            st.warning(f"🎭 Demo mode - Random example caption (beams setting: {num_beams})")
                        
                    except Exception as e:
                        st.error(f"Error generating caption: {str(e)}")
                        st.info("Falling back to demo caption...")
                        caption = generate_caption_dummy()
                        st.markdown(f"### 💬 Fallback Caption:")
                        st.markdown(f"**{caption}**")
    
    # Examples section
    st.subheader("📋 Example Usage")
    st.markdown("""
    **Expected captions:**
    - "a 3d rendered car with closed doors"
    - "a 3d rendered car with front left door open"
    - "a 3d rendered car with hood open"
    - "a 3d rendered car with front doors open"
    - "a 3d rendered car with rear doors open"
    - "a 3d rendered car with all doors open"
    """)
    
    # Status footer
    st.markdown("---")
    if model is not None:
        st.markdown("**✅ Status: AI Model Loaded • Built with BLIP • Trained on 3D car dataset**")
    else:
        st.markdown("**🎭 Status: Demo Mode • Built with BLIP • Trained on 3D car dataset**")
        st.caption("Note: Upload any image to see the UI in action (results will be demo examples)")

if __name__ == '__main__':
    main()

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import os

from models.blip import blip_decoder

# Configuration
CHECKPOINT_PATH = "checkpoint_04.pth"

@st.cache_resource
def load_trained_model():
    """Load trained BLIP model (cached for performance)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Load checkpoint
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        config = checkpoint['config']
        
        # Create model with same config as training
        model = blip_decoder(
            pretrained='',
            image_size=config['image_size'], 
            vit=config['vit'],
            prompt=config.get('prompt', 'a 3d rendered car ')
        )
        
        # Load trained weights
        model.load_state_dict(checkpoint['model'])
        model.eval()
        model = model.to(device)
        
        return model, config, device
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error(f"Make sure checkpoint exists at: {CHECKPOINT_PATH}")
        return None, None, None

def preprocess_image(image, image_size, device):
    """Preprocess uploaded image for model"""
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

def generate_caption(model, image, config, num_beams=3):
    """Generate caption for image"""
    with torch.no_grad():
        caption = model.generate(
            image, 
            sample=False, 
            num_beams=num_beams,
            max_length=config.get('max_length', 25),
            min_length=config.get('min_length', 5)
        )
        return caption[0]

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
    
    # Load model
    with st.spinner("Loading AI model..."):
        model, config, device = load_trained_model()
    
    if model is None:
        st.error("Failed to load model. Please check the checkpoint path.")
        st.stop()
    
    # Display model info
    with st.expander("ℹ️ Model Information"):
        st.write(f"**Checkpoint:** checkpoint_04.pth")
        st.write(f"**Image Size:** {config['image_size']}px")
        st.write(f"**Model Type:** {config['vit']}")
        st.write(f"**Prompt:** {config['prompt']}")
        st.write(f"**Device:** {device}")
    
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
            
            # Generation settings
            with st.expander("⚙️ Generation Settings"):
                num_beams = st.slider("Number of beams", min_value=1, max_value=10, value=3)
                st.info("Higher beams = better quality but slower")
            
            # Generate caption button
            if st.button("🎯 Generate Caption", type="primary"):
                with st.spinner("Generating caption..."):
                    try:
                        # Preprocess image
                        processed_image = preprocess_image(image, config['image_size'], device)
                        
                        # Generate caption
                        caption = generate_caption(model, processed_image, config, num_beams)
                        
                        # Display result
                        st.success("Caption generated successfully!")
                        st.markdown(f"### 💬 Generated Caption:")
                        st.markdown(f"**{caption}**")
                        
                        # Additional info
                        st.info(f"Generated using {num_beams} beams")
                        
                    except Exception as e:
                        st.error(f"Error generating caption: {str(e)}")
    
    # Examples section
    st.subheader("📋 Example Usage")
    st.markdown("""
    **Expected captions:**
    - "a 3d rendered car with closed doors"
    - "a 3d rendered car with front left door open"
    - "a 3d rendered car with hood open"
    - "a 3d rendered car with front doors open"
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("**Built with BLIP model • Trained on 3D car dataset**")

if __name__ == '__main__':

    main()

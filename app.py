import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import os

# Configuration  
CHECKPOINT_PATH = "https://drive.usercontent.google.com/download?id=1hr8cDHAJImLc6QNNa4fvQQHjdHoDyIj5&export=download&authuser=0&confirm=t"

@st.cache_resource
def load_trained_model():
    """Load trained BLIP model (cached for performance) - SIMPLIFIED VERSION"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        st.info(f"� Device: {device}")
        
        # Always return demo config for now to avoid crashes
        st.warning("⚠️ Running in DEMO mode for stability")
        dummy_config = {
            'image_size': 256,
            'vit': 'large',
            'prompt': 'a 3d rendered car ',
            'max_length': 25,
            'min_length': 5
        }
        return None, dummy_config, device
        
    except Exception as e:
        st.error(f"❌ Error in load_trained_model: {str(e)}")
        dummy_config = {
            'image_size': 256,
            'vit': 'large',
            'prompt': 'a 3d rendered car ',
            'max_length': 25,
            'min_length': 5
        }
        return None, dummy_config, torch.device('cpu')

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
    
    try:
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
            try:
                # Display uploaded image
                image = Image.open(uploaded_file)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("📸 Uploaded Image")
                    st.image(image, caption=f"Original size: {image.size}", use_column_width=True)
                
                with col2:
                    st.subheader("🤖 AI Prediction")
                    
                    # Generate caption button
                    if st.button("🎯 Generate Caption", type="primary"):
                        with st.spinner("Generating caption..."):
                            try:
                                # Always demo mode for now
                                import time
                                time.sleep(1)  # Simulate processing time
                                caption = generate_caption_dummy()
                                
                                # Display result
                                st.success("Demo caption generated!")
                                st.markdown(f"### 💬 Generated Caption:")
                                st.markdown(f"**{caption}**")
                                st.warning("🎭 Demo mode - Random example caption")
                                
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                                caption = "a 3d rendered car with closed doors"
                                st.markdown(f"### 💬 Fallback Caption:")
                                st.markdown(f"**{caption}**")
            except Exception as e:
                st.error(f"Error processing uploaded file: {str(e)}")
        
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
        st.markdown("**🎭 Status: Demo Mode • Built with BLIP • Trained on 3D car dataset**")
        st.caption("Note: Upload any image to see the UI in action (results will be demo examples)")
        
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.info("Please refresh the page or contact support.")

if __name__ == '__main__':
    main()

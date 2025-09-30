import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import argparse
import yaml

from models.blip import blip_decoder

'''
# Test dengan checkpoint epoch 4
python inference_car3d.py --image datasets/car3d/images/img_0001_p0_y0.jpg --checkpoint output/Car3D_Test/checkpoint_04.pth

# Test dengan different beam search
python inference_car3d.py --image datasets/car3d/images/img_0001_p30_y45.jpg --checkpoint output/Car3D_Test/checkpoint_04.pth --num_beams 5
'''

def load_image(image_path, image_size, device):
    """Load and preprocess image"""
    raw_image = Image.open(image_path).convert('RGB')
    
    print(f"Original image size: {raw_image.size}")
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    
    image = transform(raw_image).unsqueeze(0).to(device)
    return image, raw_image

def load_trained_model(checkpoint_path, device):
    """Load your trained BLIP model from checkpoint"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    print(f"Checkpoint info:")
    print(f"  - Epoch: {checkpoint['epoch']}")
    print(f"  - Image size: {config['image_size']}")
    print(f"  - VIT: {config['vit']}")
    print(f"  - Prompt: '{config['prompt']}'")
    
    # Create model with same config as training
    model = blip_decoder(
        pretrained='',  # No pretrained, we'll load our weights
        image_size=config['image_size'], 
        vit=config['vit'],
        prompt=config['prompt']
    )
    
    # Load trained weights
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model = model.to(device)
    
    return model, config

def generate_caption(model, image, config, num_beams=3):
    """Generate caption for image"""
    with torch.no_grad():
        # Use same settings as training config
        caption = model.generate(
            image, 
            sample=False, 
            num_beams=num_beams,
            max_length=config.get('max_length', 25),
            min_length=config.get('min_length', 5)
        )
        return caption[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--num_beams', type=int, default=3, help='Number of beams for generation')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load trained model
    model, config = load_trained_model(args.checkpoint, device)
    
    # Load and preprocess image
    image, raw_image = load_image(args.image, config['image_size'], device)
    
    # Generate caption
    print("\nGenerating caption...")
    caption = generate_caption(model, image, config, args.num_beams)
    
    # Display results
    print(f"\n{'='*50}")
    print(f"IMAGE: {args.image}")
    print(f"GENERATED CAPTION: {caption}")
    print(f"{'='*50}")
    
    # Optional: Show image (if running in Jupyter)
    try:
        from IPython.display import display
        display(raw_image.resize((300, 300)))
    except:
        print("(Install IPython to display image)")

if __name__ == '__main__':
    main()
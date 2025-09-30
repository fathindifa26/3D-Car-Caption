import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import json

from models.blip import blip_decoder

def get_ground_truth_from_csv(filename, csv_path):
    """Extract ground truth door status from CSV"""
    try:
        df = pd.read_csv(csv_path)
        row = df[df['filename'] == filename]
        
        if row.empty:
            return "Unknown (not found in CSV)"
        
        row = row.iloc[0]  # Get first match
        
        # Extract door status (same logic as convert_label_to_json.py)
        doors = []
        if row['front_left'] == 1:
            doors.append('front left door open')
        if row['front_right'] == 1:
            doors.append('front right door open')
        if row['rear_left'] == 1:
            doors.append('rear left door open')
        if row['rear_right'] == 1:
            doors.append('rear right door open')
        if row['hood'] == 1:
            doors.append('hood open')
        
        # Extract pose information
        pitch = int(row['pitch'])
        yaw = int(row['yaw'])
        
        # Generate ground truth description
        if doors:
            if len(doors) == 1:
                gt_description = f"{doors[0]} (pitch={pitch}°, yaw={yaw}°)"
            else:
                doors_text = ", ".join(doors[:-1]) + f" and {doors[-1]}"
                gt_description = f"{doors_text} (pitch={pitch}°, yaw={yaw}°)"
        else:
            gt_description = f"all doors closed (pitch={pitch}°, yaw={yaw}°)"
        
        return gt_description
        
    except Exception as e:
        return f"Error reading CSV: {str(e)}"

def analyze_prediction_accuracy(predicted_caption, ground_truth):
    """Simple accuracy analysis"""
    gt_lower = ground_truth.lower()
    pred_lower = predicted_caption.lower()
    
    accuracy_notes = []
    
    # Check door mentions
    doors_mentioned = []
    if 'front left' in pred_lower or 'front-left' in pred_lower:
        doors_mentioned.append('front_left')
    if 'front right' in pred_lower or 'front-right' in pred_lower:
        doors_mentioned.append('front_right')
    if 'rear left' in pred_lower or 'rear-left' in pred_lower or 'back left' in pred_lower:
        doors_mentioned.append('rear_left')
    if 'rear right' in pred_lower or 'rear-right' in pred_lower or 'back right' in pred_lower:
        doors_mentioned.append('rear_right')
    if 'hood' in pred_lower or 'bonnet' in pred_lower:
        doors_mentioned.append('hood')
    
    # Check for "closed" vs "open"
    if 'closed' in pred_lower and 'all doors closed' in gt_lower:
        accuracy_notes.append("✅ Correctly identified closed doors")
    elif any(door in gt_lower for door in ['front left', 'front right', 'rear left', 'rear right', 'hood']) and 'open' in pred_lower:
        accuracy_notes.append("✅ Correctly identified open doors")
    elif 'closed' in pred_lower and any(door in gt_lower for door in ['front left', 'front right', 'rear left', 'rear right', 'hood']):
        accuracy_notes.append("❌ Predicted closed but should be open")
    elif 'open' in pred_lower and 'all doors closed' in gt_lower:
        accuracy_notes.append("❌ Predicted open but should be closed")
    
    # Overall assessment
    if not accuracy_notes:
        accuracy_notes.append("⚠️  Unclear prediction accuracy")
    
    return " | ".join(accuracy_notes)

def test_multiple_images(checkpoint_path, image_dir, csv_path, num_samples=10):
    """Test trained model on multiple images with ground truth comparison"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading model from: {checkpoint_path}")
    print(f"Reading ground truth from: {csv_path}")
    print(f"Testing images from: {image_dir}")
    print("="*80)
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    print(f"Model info: Epoch {checkpoint['epoch']}, Image size: {config['image_size']}")
    print("="*80)
    
    model = blip_decoder(
        pretrained='', 
        image_size=config['image_size'], 
        vit=config['vit'],
        prompt=config.get('prompt', 'a 3d rendered car ')
    )
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model = model.to(device)
    
    # Prepare transform
    transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size']), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    
    # Get test images
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    if num_samples > 0:
        image_files = image_files[:num_samples]
    
    results = []
    print(f"Testing {len(image_files)} images...\n")
    
    for i, img_file in enumerate(image_files, 1):
        img_path = os.path.join(image_dir, img_file)
        
        # Load image
        try:
            raw_image = Image.open(img_path).convert('RGB')
            image = transform(raw_image).unsqueeze(0).to(device)
        except Exception as e:
            print(f"❌ Error loading {img_file}: {e}")
            continue
        
        # Get ground truth
        ground_truth = get_ground_truth_from_csv(img_file, csv_path)
        
        # Generate caption
        with torch.no_grad():
            caption = model.generate(
                image, 
                sample=False, 
                num_beams=3, 
                max_length=config.get('max_length', 25),
                min_length=config.get('min_length', 5)
            )
        
        predicted_caption = caption[0]
        
        # Analyze accuracy
        accuracy = analyze_prediction_accuracy(predicted_caption, ground_truth)
        
        result = {
            'image': img_file,
            'predicted_caption': predicted_caption,
            'ground_truth': ground_truth,
            'accuracy_analysis': accuracy
        }
        results.append(result)
        
        # Display results
        print(f"🖼️  [{i:2d}/{len(image_files)}] {img_file}")
        print(f"🎯 GROUND TRUTH: {ground_truth}")
        print(f"🤖 PREDICTED   : {predicted_caption}")
        print(f"📊 ACCURACY    : {accuracy}")
        print("-" * 80)
    
    # Save detailed results
    output_file = f"test_results_detailed_epoch{checkpoint['epoch']}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate summary statistics
    total_tests = len(results)
    correct_predictions = len([r for r in results if "✅" in r['accuracy_analysis']])
    incorrect_predictions = len([r for r in results if "❌" in r['accuracy_analysis']])
    unclear_predictions = total_tests - correct_predictions - incorrect_predictions
    
    print(f"\n📈 SUMMARY STATISTICS")
    print(f"="*50)
    print(f"Total images tested: {total_tests}")
    print(f"Correct predictions: {correct_predictions} ({correct_predictions/total_tests*100:.1f}%)")
    print(f"Incorrect predictions: {incorrect_predictions} ({incorrect_predictions/total_tests*100:.1f}%)")
    print(f"Unclear predictions: {unclear_predictions} ({unclear_predictions/total_tests*100:.1f}%)")
    print(f"\nDetailed results saved to: {output_file}")
    
    return results

if __name__ == '__main__':
    # Configuration
    checkpoint_path = 'output/Car3D_Test/checkpoint_04.pth'  # Update path as needed
    image_dir = 'datasets/car3d/images'
    csv_path = 'datasets/car3d/images/labels_3d.csv'
    
    # Test with ground truth comparison
    results = test_multiple_images(
        checkpoint_path=checkpoint_path,
        image_dir=image_dir, 
        csv_path=csv_path,
        num_samples=20  # Set to 0 for all images
    )
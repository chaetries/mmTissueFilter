import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from pathlib import Path
from model_utils import UNet

# Configuration
MODEL_PATH = "../models/best_model.pth"
INPUT_IMAGE_PATH = "sample_plot.png"
OUTPUT_PATH = "prediction_result.png"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_and_preprocess_image(image_path):
    print(f"Loading image from {image_path}...")
    img = Image.open(image_path).convert('RGB') # Convert to RGB
    
    # SPECIAL HANDLING for the composite plot sample
    # If using the provided sample_plot.png (which is a 3-panel plot), crop the first panel
    # The original image size was (2187, 761)
    if img.size[0] > 1500 and "sample_plot" in str(image_path):
        print("Detected composite plot. Cropping the first panel (M11 image)...")
        # Approximate crop for the first panel. Adjust coordinates if needed.
        # Assuming 3 equal panels roughly.
        width = img.size[0] // 3
        # The plot has white borders/titles. We'll try to center crop or just take the chunk.
        # Simple crop: left third
        img = img.crop((0, 0, width, img.size[1]))
        # Resize to remove title/axis if possible? 
        # For now, we'll just resize the whole thing to 512x512, ensuring we are robust.
    
    # Resize to 512x512
    img = img.resize((512, 512), Image.Resampling.BILINEAR)
    
    # Convert to numpy and normalize
    # ImageNet normalization expecting [0, 1] input range
    img_np = np.array(img).astype(np.float32) / 255.0
    
    # Normalize (ImageNet)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = (img_np - mean) / std
    
    # To Tensor: [C, H, W]
    img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0) # Add batch dim -> [1, 3, 512, 512]
    
    return img_tensor, np.array(img)

def run_inference():
    # 1. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    print("Loading model...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    
    # Determine number of classes from weights if possible, else default to 4
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        if 'final.weight' in state_dict:
            num_classes = state_dict['final.weight'].shape[0]
        else:
            num_classes = 4 
    else:
        num_classes = 4

    model = UNet(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    # 2. Process Image
    input_tensor, original_img_display = load_and_preprocess_image(INPUT_IMAGE_PATH)
    input_tensor = input_tensor.to(DEVICE)
    
    # 3. Inference
    print("Running inference...")
    with torch.no_grad():
        logits = model(input_tensor)
        pred_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
        
    # 4. Visualize
    print("Saving result...")
    # Define colors: 0=Black, 1=Blue, 2=Green, 3=Red
    # Map class 0->[0,0,0], 1->[0,0,255], 2->[0,255,0], 3->[255,0,0]
    colors = np.array([
        [0, 0, 0],
        [0, 0, 255],
        [0, 255, 0],
        [255, 0, 0]
    ], dtype=np.uint8)
    
    if num_classes > 4:
        # Fallback for more classes
        colors = np.random.randint(0, 255, (num_classes, 3), dtype=np.uint8)
        colors[0] = [0,0,0]

    h, w = pred_mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for c in range(num_classes):
        colored_mask[pred_mask == c] = colors[c] if c < len(colors) else [255, 255, 255]
        
    # Plot side by side
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(original_img_display)
    ax[0].set_title("Input (Resized/Cropped)")
    ax[0].axis('off')
    
    ax[1].imshow(colored_mask)
    ax[1].set_title("Prediction (Blue=Tissue, Green=OS, Red=Vaginal)")
    ax[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH)
    print(f"Done! Result saved to {os.path.abspath(OUTPUT_PATH)}")

if __name__ == "__main__":
    run_inference()

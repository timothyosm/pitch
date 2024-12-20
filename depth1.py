import cv2
import torch
import time
import numpy as np
import torchvision.transforms as transforms

def create_custom_colormap(colors, num_steps=256):
    """
    Create a custom colormap from a list of colors.
    
    Args:
        colors: List of RGB tuples [(R,G,B), ...] with values from 0-255
        num_steps: Number of interpolation steps (default: 256)
    
    Returns:
        numpy array of shape (256, 3) for OpenCV applyColorMap
    """
    if len(colors) < 2:
        raise ValueError("Need at least 2 colors to create a colormap")
    
    # Convert colors to numpy array
    colors = np.array(colors, dtype=np.float32)
    
    # Calculate steps between each color pair
    n_colors = len(colors)
    steps_per_segment = num_steps // (n_colors - 1)
    
    # Initialize the colormap
    colormap = np.zeros((256, 3), dtype=np.uint8)
    
    # Interpolate between each pair of colors
    for i in range(n_colors - 1):
        start_idx = i * steps_per_segment
        end_idx = (i + 1) * steps_per_segment if i < n_colors - 2 else num_steps
        
        for j in range(3):  # RGB channels
            colormap[start_idx:end_idx, j] = np.linspace(
                colors[i][j],
                colors[i + 1][j],
                end_idx - start_idx
            ).astype(np.uint8)
    
    return colormap

# Create custom colormap
colors = [
      (0, 0, 0),        # Pure black
    (20, 0, 0),
    (40, 0, 0),
    (60, 0, 0),
    (80, 0, 0),
    (100, 0, 0),
    (120, 0, 0),
    (140, 0, 0),
    (160, 0, 0),
    (180, 0, 0),
    (200, 20, 0),
    (220, 40, 0),
    (240, 60, 0),
    (255, 80, 0),
    (255, 100, 0),
    (255, 120, 0),
    (255, 140, 0),
    (255, 160, 0),
    (255, 180, 0),
    (255, 200, 0)     # Bright orange
]
custom_colormap = create_custom_colormap(colors)

# Select which MiDaS model to use
model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access webcam.")
    exit()

def enhance_depth_map(depth_map, detail_factor=1.5):
    """Enhance local contrast to bring out fine details"""
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    normalized = (depth_map - depth_min) / (depth_max - depth_min)
    enhanced = np.power(normalized, 1/detail_factor)
    return enhanced

def apply_custom_colormap(image, colormap):
    """Apply custom colormap to single-channel image"""
    # Ensure image is 8-bit
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
        
    # Create output image
    output = np.zeros((*image.shape, 3), dtype=np.uint8)
    
    # Apply colormap
    for i in range(3):
        output[..., i] = np.take(colormap[:, i], image)
    
    return output

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Frame not received from webcam.")
        break

    # Convert to RGB and prepare for MiDaS
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    
    # Enhance and normalize depth map
    enhanced_depth = enhance_depth_map(depth_map)
    depth_visual = (enhanced_depth * 255).astype(np.uint8)
    
    # Apply custom colormap
    colored = apply_custom_colormap(depth_visual, custom_colormap)
    
    # Show the depth map
    cv2.imshow("Depth Map", colored)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
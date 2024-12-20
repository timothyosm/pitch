import cv2
import torch
import time
import numpy as np
import torchvision.transforms as transforms

# Select a smaller/faster MiDaS model for better performance
model_type = "MiDaS_small"

# Load the MiDaS model from torch hub
midas = torch.hub.load("intel-isl/MiDaS", model_type)

# MiDaS transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type in ["DPT_Large", "DPT_Hybrid"]:
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# Set device: Prefer MPS (Apple Silicon), then fallback to CUDA or CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

midas.to(device)
midas.eval()

# (Optional) If you want to try half precision on MPS or CUDA:
# midas.half()

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Could not access webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Frame not received from webcam.")
        break

    # Convert to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Optionally downsample further to speed up processing:
    # img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2), interpolation=cv2.INTER_AREA)

    # Prepare input
    input_batch = transform(img).to(device)
    # If using half precision (uncomment if desired):
    # input_batch = input_batch.half()

    with torch.no_grad():
        prediction = midas(input_batch)
        # Use bilinear interpolation instead of bicubic for MPS compatibility
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bilinear",  # changed from "bicubic"
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    # Normalize depth for visualization
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_visual = (depth_map - depth_min) / (depth_max - depth_min + 1e-6)
    depth_visual = (depth_visual * 255).astype(np.uint8)

    # Apply a colormap
    depth_colored = cv2.applyColorMap(depth_visual, cv2.COLORMAP_MAGMA)

    cv2.imshow("Depth Map", depth_colored)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import torch
import time
import numpy as np
import torchvision.transforms as transforms

# Use the best MiDaS model for high-quality depth (DPT_Large)
model_type = "DPT_Large"

# Load the MiDaS model from torch hub
midas = torch.hub.load("intel-isl/MiDaS", model_type)

# MiDaS transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type in ["DPT_Large", "DPT_Hybrid"]:
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# Set device: Prefer MPS (Apple Silicon), then CUDA, then CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

midas.to(device)
midas.eval()

# Initialize webcam at a reasonably high resolution if possible
cap = cv2.VideoCapture(0)
# Try higher resolutions if supported by your webcam and machine:
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: Could not access webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Frame not received from webcam.")
        break

    # Convert to RGB for MiDaS
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Prepare input for the model
    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        # Use bilinear interpolation to resize the depth map back to the original frame size
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bilinear",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    # Normalize the depth map for visualization
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_visual = (depth_map - depth_min) / (depth_max - depth_min + 1e-6)
    depth_visual = (depth_visual * 255).astype(np.uint8)

    # ---- Depth-Fog Effect ----
    # Normalize depth for fog calculation
    depth_normalized = (depth_map - depth_min) / (depth_max - depth_min + 1e-6)

    # Fog parameters
    fog_color = np.array([200, 200, 200], dtype=np.float32)  # Light gray in BGR
    fog_intensity = 0.7  # Adjust this to increase/decrease fog strength

    # Fog factor is proportional to depth. Near = little fog, Far = more fog
    fog_factor = depth_normalized * fog_intensity
    fog_factor_3d = fog_factor[..., None]  # to broadcast over BGR channels

    # Blend the original frame with the fog color
    frame_float = frame.astype(np.float32)
    frame_fog = (frame_float * (1 - fog_factor_3d) + fog_color * fog_factor_3d).astype(np.uint8)
    # ---------------------------

    # Apply a color map for depth visualization (optional, you can remove if you only want fog)
    depth_colored = cv2.applyColorMap(depth_visual, cv2.COLORMAP_MAGMA)

    # Display both the depth map and the fog effect
    cv2.imshow("Depth Map", depth_colored)
    cv2.imshow("Fog Effect", frame_fog)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

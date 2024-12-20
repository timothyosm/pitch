import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.cuda.amp import autocast

class DepthEstimator:
    def __init__(self, model_type="DPT_Large"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
        self.midas.to(self.device).eval()
        
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.dpt_transform if model_type in ["DPT_Large", "DPT_Hybrid"] else midas_transforms.small_transform
        
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def create_tricolor_map(self, depth_map):
        # Normalize depth map to 0-1 range
        depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        
        # Create empty RGB image
        height, width = depth_map.shape
        colored_map = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Define color boundaries (adjust these values to change the color distribution)
        boundaries = [0.33, 0.66]  # Divides the depth range into thirds
        
        # Define colors: blue, yellow, red
        colors = {
            'near': np.array([255, 0, 0]),      # Red
            'mid': np.array([255, 255, 0]),     # Yellow
            'far': np.array([0, 0, 255])        # Blue
        }
        
        # Apply colors based on depth
        colored_map[depth_norm <= boundaries[0]] = colors['near']
        colored_map[(depth_norm > boundaries[0]) & (depth_norm <= boundaries[1])] = colors['mid']
        colored_map[depth_norm > boundaries[1]] = colors['far']
        
        return colored_map

    @torch.no_grad()
    def process_frame(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        input_batch = self.transform(img).to(self.device)
        
        with autocast(enabled=self.device.type == "cuda"):
            prediction = self.midas(input_batch)
            
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        colored_depth = self.create_tricolor_map(depth_map)
        
        return colored_depth

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not access webcam")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    depth_estimator = DepthEstimator()
    
    cv2.namedWindow("Depth Map (Tricolor)", cv2.WINDOW_NORMAL)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            colored_depth = depth_estimator.process_frame(frame)
            
            # Convert from RGB to BGR for OpenCV display
            colored_depth_bgr = cv2.cvtColor(colored_depth, cv2.COLOR_RGB2BGR)
            cv2.imshow("Depth Map (Tricolor)", colored_depth_bgr)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
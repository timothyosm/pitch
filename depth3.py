import cv2
import torch
import time
import numpy as np
import torchvision.transforms as transforms
from typing import Tuple, Optional, Dict
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class CameraConfig:
    width: int = 640  # Reduced from 1280 for better performance
    height: int = 480  # Reduced from 720 for better performance
    device_id: int = 0
    fps: Optional[int] = 30

class DepthEstimator:
    def __init__(self, model_type: str = "MiDaS_small", camera_config: Optional[CameraConfig] = None):
        self.model_type = model_type
        self.camera_config = camera_config or CameraConfig()
        self.device = self._setup_device()
        self.model, self.transform = self._setup_model()
        self.cap = self._setup_camera()  # This will now work with the added method
        self.running = False
        self.fps_history = []
        
        self.visualization_params = {
            'min_depth_percentile': 5,
            'max_depth_percentile': 95,
            'contrast': 1.3,
            'brightness': 1.0,
            'blur_size': 2,
            'color_map': cv2.COLORMAP_TURBO
        }
        
        self.frame_skip = 1
        self.frame_count = 0

    def _setup_camera(self) -> cv2.VideoCapture:
        """Initialize and configure the webcam with optimal settings for M1."""
        cap = cv2.VideoCapture(self.camera_config.device_id)
        
        if not cap.isOpened():
            raise RuntimeError("Could not access webcam. Please check permissions and connection.")

        # Set properties with error checking
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_config.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_config.height)
        
        if self.camera_config.fps:
            cap.set(cv2.CAP_PROP_FPS, self.camera_config.fps)

        # Verify settings
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Set buffer size to minimum for lower latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        logging.info(f"Camera initialized at {actual_width}x{actual_height} @ {actual_fps}fps")
        return cap

    def _setup_device(self) -> torch.device:
        """Optimized device setup for M1 Mac."""
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            torch.mps.set_per_process_memory_fraction(0.7)
            logging.info("Using MPS (Apple Silicon) device")
        else:
            device = torch.device("cpu")
            logging.info("Using CPU device")
        return device

    def _setup_model(self) -> Tuple[torch.nn.Module, transforms.Compose]:
        """Initialize and configure the MiDaS model with optimizations."""
        try:
            model = torch.hub.load("intel-isl/MiDaS", self.model_type)
            model.to(self.device)
            model.eval()
            
            torch.set_grad_enabled(False)
            if hasattr(torch, 'inference_mode'):
                torch.inference_mode(True)
            
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            transform = midas_transforms.small_transform
            
            return model, transform
        except Exception as e:
            logging.error(f"Failed to load MiDaS model: {e}")
            raise

    def process_frame(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """Optimized frame processing."""
        frame = cv2.resize(frame, (384, 288))
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        input_batch = self.transform(img).to(self.device)

        with torch.inference_mode():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bilinear",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        
        refined_depth = self._refine_depth_map(depth_map)
        depth_visual = self._create_depth_visualization(refined_depth)
        
        return {
            'raw_depth': depth_map,
            'refined_depth': refined_depth,
            'visualization': cv2.resize(depth_visual, (self.camera_config.width, self.camera_config.height))
        }

    def _refine_depth_map(self, depth_map: np.ndarray) -> np.ndarray:
        """Simplified depth map refinement."""
        depth_filtered = cv2.GaussianBlur(
            depth_map.astype(np.float32),
            (self.visualization_params['blur_size'] * 2 + 1, self.visualization_params['blur_size'] * 2 + 1),
            0
        )
        
        min_val = np.percentile(depth_filtered, self.visualization_params['min_depth_percentile'])
        max_val = np.percentile(depth_filtered, self.visualization_params['max_depth_percentile'])
        
        depth_normalized = np.clip((depth_filtered - min_val) / (max_val - min_val + 1e-6), 0, 1)
        depth_contrast = np.power(depth_normalized, 1.0 / self.visualization_params['contrast'])
        
        return (depth_contrast * 255).astype(np.uint8)

    def _create_depth_visualization(self, refined_depth: np.ndarray) -> np.ndarray:
        """Simplified visualization creation."""
        return cv2.applyColorMap(refined_depth, self.visualization_params['color_map'])

    def run(self):
        """Optimized main processing loop."""
        self.running = True
        last_fps_update = time.time()
        fps_display = 0
        
        try:
            while self.running:
                self.frame_count += 1
                ret, frame = self.cap.read()
                
                if not ret:
                    logging.error("Failed to receive frame from webcam")
                    break

                if self.frame_count % self.frame_skip != 0:
                    cv2.imshow("Original", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                start_time = time.time()
                
                results = self.process_frame(frame)
                depth_colored = results['visualization']
                
                if time.time() - last_fps_update > 0.5:
                    fps_display = 1.0 / (time.time() - start_time)
                    last_fps_update = time.time()

                cv2.putText(depth_colored, f"FPS: {fps_display:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                cv2.imshow("Original", frame)
                cv2.imshow("Depth Map", depth_colored)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            logging.info("Processing interrupted by user")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        self.running = False
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()
        logging.info("Cleanup completed")

if __name__ == "__main__":
    try:
        config = CameraConfig(
            width=640,
            height=480,
            fps=30
        )
        
        depth_estimator = DepthEstimator(
            model_type="MiDaS_small",
            camera_config=config
        )
        depth_estimator.run()
        
    except Exception as e:
        logging.error(f"Application error: {e}")
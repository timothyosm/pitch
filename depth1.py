import cv2
import torch
import time
import numpy as np
import torchvision.transforms as transforms
from concurrent.futures import ThreadPoolExecutor
import coremltools as ct

class DepthMapper:
    def __init__(self):
        # Use MPS (Metal Performance Shaders) backend for Apple Silicon
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Initialize model
        self.model_type = "DPT_Large"
        self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self.midas.to(self.device)
        self.midas.eval()
        
        # Convert model to CoreML for better M3 performance
        self.convert_to_coreml()
        
        # Initialize transforms with hardware acceleration
        self.transform = self._create_optimized_transform()
        
        # Create thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize colormap
        self.custom_colormap = self.create_custom_colormap([
            (0, 0, 0),      # Pure black
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
            (255, 200, 0)   # Bright orange
        ])
        
        # Initialize video capture with optimized settings
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
    def convert_to_coreml(self):
        """Convert PyTorch model to CoreML format for M3 optimization"""
        try:
            # Create a trace of the model
            example_input = torch.rand(1, 3, 384, 384)
            traced_model = torch.jit.trace(self.midas, example_input)
            
            # Convert to CoreML
            self.coreml_model = ct.convert(
                traced_model,
                inputs=[ct.TensorType(shape=example_input.shape)],
                compute_units=ct.ComputeUnit.ALL  # Use all available compute units
            )
            
            # Save the model
            self.coreml_model.save("midas_depth_m3.mlmodel")
            print("Successfully converted to CoreML model")
            
        except Exception as e:
            print(f"CoreML conversion failed: {e}")
            self.coreml_model = None

    def _create_optimized_transform(self):
        """Create optimized transform pipeline for M3"""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((384, 384), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def create_custom_colormap(colors, num_steps=256):
        """Create custom colormap using numpy vectorization"""
        colors = np.array(colors, dtype=np.float32)
        n_colors = len(colors)
        steps_per_segment = num_steps // (n_colors - 1)
        
        # Pre-allocate array
        colormap = np.zeros((256, 3), dtype=np.uint8)
        
        # Vectorized interpolation
        for i in range(n_colors - 1):
            start_idx = i * steps_per_segment
            end_idx = (i + 1) * steps_per_segment if i < n_colors - 2 else num_steps
            
            # Create interpolation weights
            weights = np.linspace(0, 1, end_idx - start_idx)
            
            # Vectorized RGB interpolation
            colormap[start_idx:end_idx] = (
                colors[i] * (1 - weights[:, np.newaxis]) +
                colors[i + 1] * weights[:, np.newaxis]
            ).astype(np.uint8)
            
        return colormap

    def enhance_depth_map(self, depth_map, detail_factor=1.5):
        """Enhance depth map using vectorized operations"""
        depth_min = np.min(depth_map)
        depth_max = np.max(depth_map)
        normalized = np.clip((depth_map - depth_min) / (depth_max - depth_min), 0, 1)
        return np.power(normalized, 1/detail_factor, dtype=np.float32)

    def apply_custom_colormap(self, image):
        """Apply colormap using vectorized operations"""
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Vectorized colormap application
        return self.custom_colormap[image]

    def process_frame(self):
        """Process a single frame with error handling"""
        ret, frame = self.cap.read()
        if not ret:
            return None

        # Convert to RGB and prepare input
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).unsqueeze(0)
        
        if self.device.type == "mps":
            input_batch = input_batch.to(self.device)
            
        try:
            with torch.no_grad():
                prediction = self.midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            depth_map = prediction.cpu().numpy()
            
            # Process depth map in parallel
            enhanced_depth = self.executor.submit(
                self.enhance_depth_map, depth_map
            ).result()
            
            depth_visual = (enhanced_depth * 255).astype(np.uint8)
            colored = self.apply_custom_colormap(depth_visual)
            
            return colored
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return None

    def run(self):
        """Main processing loop with performance monitoring"""
        try:
            frame_times = []
            while True:
                start_time = time.time()
                
                colored = self.process_frame()
                if colored is None:
                    break
                
                # Calculate and display FPS
                frame_time = time.time() - start_time
                frame_times.append(frame_time)
                if len(frame_times) > 30:
                    frame_times.pop(0)
                fps = 1.0 / (sum(frame_times) / len(frame_times))
                
                # Add FPS display
                cv2.putText(colored, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                cv2.imshow("Depth Map", colored)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        self.executor.shutdown()

if __name__ == "__main__":
    mapper = DepthMapper()
    mapper.run()
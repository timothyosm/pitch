import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

def calculate_focus_measure(image, window_size=15):
    # Measure local variance as a focus indicator
    mean = cv2.blur(image.astype(float), (window_size, window_size))
    mean_sq = cv2.blur(image.astype(float) * image, (window_size, window_size))
    variance = mean_sq - mean * mean
    return variance

def estimate_relative_size(image):
    # Detect objects and estimate their relative sizes
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 100
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(image)
    
    size_map = np.zeros_like(image, dtype=float)
    for kp in keypoints:
        cv2.circle(size_map, (int(kp.pt[0]), int(kp.pt[1])), 
                  int(kp.size), kp.size/100.0, -1)
    return size_map

def texture_gradient(image):
    # Calculate texture density gradient
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    texture = gaussian_filter(magnitude, sigma=5)
    return texture

def estimate_depth(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate different depth cues
    focus_map = calculate_focus_measure(gray)
    size_map = estimate_relative_size(gray)
    texture_map = texture_gradient(gray)
    
    # Combine depth cues
    depth_map = (0.4 * focus_map + 
                 0.3 * size_map + 
                 0.3 * texture_map)
    
    # Enhance contrast
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map = cv2.equalizeHist(depth_map.astype(np.uint8))
    
    # Apply bilateral filter for edge preservation
    depth_map = cv2.bilateralFilter(depth_map, 9, 75, 75)
    
    return depth_map

def main():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        depth_map = estimate_depth(frame)
        
        # Create side-by-side display
        display = np.hstack((frame, cv2.cvtColor(depth_map, cv2.COLOR_GRAY2BGR)))
        cv2.imshow('Depth Mapping (Original | Depth)', display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
import cv2
import mediapipe as mp
import pygame
import numpy as np
from scipy.ndimage import distance_transform_edt
import matplotlib.cm as cm

# Initialize components
pygame.init()
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption('Thermal Camera Simulation')

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    enable_segmentation=True
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

previous_mask = None
previous_heat_map = None

def apply_thermal_noise(heat_map, noise_scale=0.015):
    """Adds realistic thermal sensor noise."""
    noise = np.random.normal(0, noise_scale, heat_map.shape)
    return np.clip(heat_map + noise, 0, 1)

def simulate_thermal_gradients(mask, prev_heat=None):
    """Simulates realistic thermal gradients with temporal consistency."""
    # Generate base heat distribution
    distance = distance_transform_edt(mask)
    if distance.max() > 0:
        heat_map = distance / distance.max()
    else:
        heat_map = np.zeros_like(mask)
    
    # Add anatomical heat variations
    heat_map = np.power(heat_map, 0.7)  # Emphasize core body heat
    
    # Add temporal consistency
    if prev_heat is not None:
        heat_map = cv2.addWeighted(heat_map, 0.7, prev_heat, 0.3, 0)
    
    # Add thermal conductivity simulation
    heat_map = cv2.GaussianBlur(heat_map, (15, 15), 3)
    
    return heat_map

def process_segmentation(mask, prev_mask=None):
    """Processes segmentation mask with realistic body shape considerations."""
    # Spatial smoothing
    mask = cv2.GaussianBlur(mask, (11, 11), 2)
    
    # Threshold for body detection
    _, mask = cv2.threshold(mask, 0.2, 1, cv2.THRESH_BINARY)
    
    # Enhance body shape
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Temporal smoothing
    if prev_mask is not None:
        mask = cv2.addWeighted(mask, 0.8, prev_mask, 0.2, 0)
    
    return mask

def create_thermal_visualization(frame):
    """Creates realistic thermal camera visualization."""
    global previous_mask, previous_heat_map
    
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    
    if results.segmentation_mask is None:
        return pygame.Surface((frame.shape[1], frame.shape[0]))
    
    # Process mask
    current_mask = process_segmentation(
        results.segmentation_mask.astype(np.float32),
        previous_mask
    )
    previous_mask = current_mask.copy()
    
    # Generate heat map
    heat_map = simulate_thermal_gradients(current_mask, previous_heat_map)
    previous_heat_map = heat_map.copy()
    
    # Add sensor noise
    heat_map = apply_thermal_noise(heat_map)
    
    # Apply colormap
    thermal = cm.inferno(heat_map)[:, :, :3]
    thermal = (thermal * 255).astype(np.uint8)
    
    # Create final image
    mask_3d = np.stack([current_mask] * 3, axis=-1)
    background = np.zeros_like(frame_rgb)
    result = thermal * mask_3d + background * (1 - mask_3d)
    
    # Convert to pygame surface
    return pygame.surfarray.make_surface(result.astype(np.uint8).swapaxes(0, 1))

def main():
    running = True
    clock = pygame.time.Clock()
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_q
            ):
                running = False
            elif event.type == pygame.VIDEORESIZE:
                global SCREEN_WIDTH, SCREEN_HEIGHT, screen
                SCREEN_WIDTH, SCREEN_HEIGHT = event.size
                screen = pygame.display.set_mode(
                    (SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE
                )
        
        ret, frame = cap.read()
        if not ret:
            break
            
        try:
            thermal = create_thermal_visualization(frame)
            thermal = pygame.transform.scale(thermal, (SCREEN_WIDTH, SCREEN_HEIGHT))
            screen.blit(thermal, (0, 0))
            pygame.display.flip()
        except Exception as e:
            print(f"Error: {e}")
            continue
            
        clock.tick(30)
    
    cap.release()
    pygame.quit()
    pose.close()

if __name__ == '__main__':
    main()
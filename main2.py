import cv2
import mediapipe as mp
import pygame
import numpy as np
from scipy.ndimage import distance_transform_edt
import matplotlib.cm as cm

# Initialize pygame and set up display window.
pygame.init()

# Define the initial screen size.
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption('Core-to-Edge Thermal Camera Effect')

# Initialize MediaPipe with segmentation enabled.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    enable_segmentation=True
)

# Initialize webcam.
cap = cv2.VideoCapture(0)
# Set camera resolution to reduce computational load.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Initialize previous mask for temporal smoothing.
previous_mask = None

def smooth_mask(current_mask, prev_mask, temporal_factor=0.2):
    """Applies temporal and spatial smoothing to the mask."""
    # Apply spatial smoothing.
    kernel_size = 15
    current_mask = cv2.GaussianBlur(current_mask, (kernel_size, kernel_size), 0)

    # Lower the threshold to include more of the arms.
    _, current_mask = cv2.threshold(current_mask, 0.1, 1, cv2.THRESH_BINARY)

    # Morphological operations.
    kernel = np.ones((5, 5), np.uint8)
    current_mask = cv2.morphologyEx(current_mask, cv2.MORPH_CLOSE, kernel)
    current_mask = cv2.morphologyEx(current_mask, cv2.MORPH_OPEN, kernel)

    # Apply dilation to widen the arms.
    dilation_kernel = np.ones((15, 15), np.uint8)
    current_mask = cv2.dilate(current_mask, dilation_kernel, iterations=1)

    # Temporal smoothing.
    if prev_mask is not None:
        current_mask = cv2.addWeighted(
            current_mask, 1 - temporal_factor, prev_mask, temporal_factor, 0)
    return current_mask

def create_core_heat_mask(segmentation_mask):
    """Creates a heat mask that's hotter in the core and cooler at the edges."""
    binary_mask = (segmentation_mask > 0.1).astype(np.uint8)
    distance_map = distance_transform_edt(binary_mask)

    if distance_map.max() > 0:
        distance_map = distance_map / distance_map.max()

    # Reduce gamma to make edges more pronounced.
    gamma = 1.0
    heat_mask = np.power(distance_map, gamma)
    heat_mask = heat_mask * segmentation_mask
    heat_mask = cv2.GaussianBlur(heat_mask, (11, 11), 0)

    return heat_mask

def apply_thermal_effect(frame, segmentation_mask):
    """Applies thermal camera effect to the frame with core-to-edge gradient."""
    heat_mask = create_core_heat_mask(segmentation_mask)
    noise = np.random.normal(0, 0.02, heat_mask.shape).astype(np.float32)
    heat_mask = np.clip(heat_mask + noise * segmentation_mask, 0, 1)
    heat_mask = cv2.GaussianBlur(heat_mask, (11, 11), 0)
    heat_mask = np.clip(heat_mask, 0, 1)

    # Apply the 'inferno' colormap.
    thermal_image = cm.get_cmap('inferno')(heat_mask)[:, :, :3]
    thermal_image = (thermal_image * 255).astype(np.uint8)

    # Create background.
    background = np.zeros_like(frame)

    # Blend thermal image with background.
    mask_3d = np.stack([segmentation_mask] * 3, axis=-1)
    result = thermal_image * mask_3d + background * (1 - mask_3d)

    return result.astype(np.uint8)

def draw_thermal_view(frame):
    """Creates a visualization with thermal camera effect."""
    global previous_mask

    # Flip the frame horizontally.
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.segmentation_mask is not None:
        current_mask = results.segmentation_mask.astype(np.float32)

        # Apply smoothing.
        smoothed_mask = smooth_mask(current_mask, previous_mask)
        previous_mask = smoothed_mask.copy()

        # Apply thermal effect.
        thermal_frame = apply_thermal_effect(frame_rgb, smoothed_mask)

        # Convert to pygame surface.
        thermal_surface = pygame.surfarray.make_surface(thermal_frame.swapaxes(0, 1))
        return thermal_surface
    else:
        # Return a blank surface if no segmentation mask is available.
        return pygame.Surface((frame.shape[1], frame.shape[0]))

def main():
    """Main loop for thermal camera effect display."""
    running = True
    clock = pygame.time.Clock()
    global SCREEN_WIDTH, SCREEN_HEIGHT, screen

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                SCREEN_WIDTH, SCREEN_HEIGHT = event.size
                screen = pygame.display.set_mode(
                    (SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE
                )

        ret, frame = cap.read()
        if not ret:
            break

        try:
            thermal_surface = draw_thermal_view(frame)
            thermal_surface = pygame.transform.scale(
                thermal_surface, (SCREEN_WIDTH, SCREEN_HEIGHT))
            screen.blit(thermal_surface, (0, 0))
            pygame.display.flip()
        except Exception as e:
            print(f"Error processing frame: {e}")
            continue

        clock.tick(30)

    cap.release()
    pygame.quit()
    pose.close()

if __name__ == '__main__':
    main()

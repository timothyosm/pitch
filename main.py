import cv2
import mediapipe as mp
import pygame
import numpy as np
from scipy.ndimage import distance_transform_edt

# Initialize pygame and set up display window
pygame.init()

# Define the screen size
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Core-to-Edge Thermal Camera Effect')

# Initialize MediaPipe with segmentation enabled
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    enable_segmentation=True
)

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)

# Initialize previous mask for temporal smoothing
previous_mask = None


def create_thermal_colormap():
    """Create a thermal colormap from cool to hot colors"""
    colors = [
        (0, 0, 0),        # Black for background
        (200, 0, 0),      # Deep red (coolest body part)
        (255, 60, 0),     # Red-orange
        (255, 150, 0),    # Orange
        (255, 255, 0),    # Yellow
        (255, 255, 255)   # White (hottest)
    ]

    colormap = np.zeros((256, 3), dtype=np.uint8)
    intervals = len(colors) - 1
    interval_size = 256 // intervals

    for i in range(intervals):
        start_color = np.array(colors[i])
        end_color = np.array(colors[i + 1])
        for j in range(interval_size):
            t = j / interval_size
            color = start_color * (1 - t) + end_color * t
            idx = i * interval_size + j
            if idx < 256:
                colormap[idx] = color

    return colormap


THERMAL_COLORMAP = create_thermal_colormap()


def smooth_mask(current_mask, prev_mask, temporal_factor=0.8):
    """Apply temporal and spatial smoothing to the mask"""
    # Apply strong spatial smoothing first
    kernel_size = 15  # Increased kernel size for stronger smoothing
    current_mask = cv2.GaussianBlur(
        current_mask, (kernel_size, kernel_size), 0)

    # Threshold the mask to make it more definitive
    current_mask = cv2.threshold(current_mask, 0.3, 1, cv2.THRESH_BINARY)[1]

    # Apply morphological operations to fill holes and smooth edges
    kernel = np.ones((5, 5), np.uint8)
    current_mask = cv2.morphologyEx(current_mask, cv2.MORPH_CLOSE, kernel)
    current_mask = cv2.morphologyEx(current_mask, cv2.MORPH_OPEN, kernel)

    # Apply temporal smoothing if we have a previous mask
    if prev_mask is not None:
        return cv2.addWeighted(current_mask, 1 - temporal_factor, prev_mask, temporal_factor, 0)
    return current_mask


def create_core_heat_mask(segmentation_mask):
    """Create a heat mask that's hotter in the core and cooler at the edges"""
    # Create binary mask with stronger threshold
    binary_mask = (segmentation_mask > 0.4).astype(np.uint8)

    # Calculate distance from edge for every point inside the mask
    distance_map = distance_transform_edt(binary_mask)

    # Normalize the distance map
    if distance_map.max() > 0:
        distance_map = distance_map / distance_map.max()

    # Apply gamma correction to create more pronounced core
    gamma = 1.5
    heat_mask = np.power(distance_map, gamma)

    # Scale back to original segmentation mask range and smooth
    heat_mask = heat_mask * segmentation_mask
    heat_mask = cv2.GaussianBlur(heat_mask, (11, 11), 0)

    return heat_mask


def apply_thermal_effect(frame, segmentation_mask):
    """Apply thermal camera effect to the frame with core-to-edge gradient"""
    # Create heat mask with core-to-edge gradient
    heat_mask = create_core_heat_mask(segmentation_mask)

    # Add very subtle noise for texture (reduced noise amount)
    noise = np.random.normal(0, 0.02, heat_mask.shape).astype(np.float32)
    heat_mask = np.clip(heat_mask + noise * segmentation_mask, 0, 1)

    # Apply stronger blur for smoother transition
    heat_mask = cv2.GaussianBlur(heat_mask, (11, 11), 0)

    # Convert to color range
    normalized_mask = (heat_mask * 255).astype(np.uint8)

    # Create the thermal image
    thermal_image = np.zeros_like(frame)
    for i in range(3):
        thermal_image[:, :, i] = THERMAL_COLORMAP[normalized_mask][:, :, i]

    # Create background
    background = np.zeros_like(frame)  # Pure black background

    # Blend thermal image with background
    mask_3d = np.stack([segmentation_mask] * 3, axis=-1)
    result = thermal_image * mask_3d + background * (1 - mask_3d)

    return result.astype(np.uint8)


def draw_thermal_view(frame):
    """Creates a visualization with thermal camera effect."""
    global previous_mask

    # Flip the frame horizontally to create mirror effect
    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # Create surface for drawing
    final_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    final_surface.fill((0, 0, 0))  # Black background

    if results.segmentation_mask is not None:
        # Process segmentation mask
        current_mask = cv2.resize(results.segmentation_mask,
                                  (SCREEN_WIDTH, SCREEN_HEIGHT))

        # Apply temporal and spatial smoothing
        smoothed_mask = smooth_mask(current_mask, previous_mask)
        previous_mask = smoothed_mask.copy()

        # Apply thermal effect
        thermal_frame = apply_thermal_effect(frame_rgb, smoothed_mask)

        # Convert to pygame surface and rotate to correct orientation
        thermal_frame = np.rot90(thermal_frame, k=-1)
        thermal_surface = pygame.surfarray.make_surface(thermal_frame)
        final_surface.blit(thermal_surface, (0, 0))

    return np.rot90(pygame.surfarray.array3d(final_surface))


def main():
    """Main loop for thermal camera effect display."""
    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False

        ret, frame = cap.read()
        if not ret:
            break

        try:
            output_array = draw_thermal_view(frame)
            surface = pygame.surfarray.make_surface(
                output_array.transpose(1, 0, 2))
            screen.blit(surface, (0, 0))
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
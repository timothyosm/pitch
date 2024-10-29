import cv2
import mediapipe as mp
import pygame
import numpy as np
import random

# Initialize pygame and set up display window
pygame.init()

# Define the screen size
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Realistic Full-Body Flame Effect')

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

# Flame buffer for flame effect
flame_buffer = None

def create_flame_palette(hue_shift=0):
    """Create a palette of colors representing flames, with optional hue shift."""
    colors = [
        (0, 0, 0),          # Black
        (7, 7, 7),          # Very dark grey
        (31, 7, 7),         # Dark red
        (47, 15, 7),        # Darker red
        (71, 15, 7),        # Dark red-orange
        (87, 23, 7),        # Red-orange
        (103, 31, 7),       # Orange
        (119, 31, 7),       # Orange-yellow
        (143, 39, 7),       # Yellow-orange
        (159, 47, 7),       # Yellow
        (175, 63, 7),       # Light yellow
        (191, 71, 7),       # Lighter yellow
        (199, 71, 7),       # Even lighter yellow
        (223, 79, 7),       # Pale yellow
        (223, 87, 7),       # Paler yellow
        (223, 87, 7),       # More pale yellow
        (215, 95, 7),       # Very pale yellow
        (215, 95, 7),       # Very pale yellow
        (215, 103, 15),     # Very pale yellow with a hint of orange
        (207, 111, 15),     # Pale yellow-orange
        (207, 119, 15),     # Pale orange
        (207, 127, 15),     # Light orange
        (199, 135, 23),     # Light orange
        (199, 135, 23),     # Light orange
        (191, 143, 23),     # Light orange-yellow
        (191, 151, 31),     # Light yellow
        (191, 159, 31),     # Lighter yellow
        (191, 159, 31),     # Lighter yellow
        (191, 167, 39),     # Lighter yellow
        (191, 167, 39),     # Lighter yellow
        (191, 175, 47),     # Very light yellow
        (183, 175, 47),     # Very light yellow
        (183, 183, 47),     # Light yellow
        (183, 183, 55),     # Pale yellow
        (207, 207, 111),    # Very pale yellow
        (223, 223, 159),    # Almost white
        (239, 239, 199),    # Near white
        (255, 255, 255)     # White
    ]
    palette = np.array(colors, dtype=np.uint8)

    # Convert palette to HSV
    palette_hsv = cv2.cvtColor(palette[np.newaxis, :, :], cv2.COLOR_RGB2HSV)

    # Shift hue
    palette_hsv[0, :, 0] = (palette_hsv[0, :, 0] + hue_shift) % 180  # Hue is in [0,179] in OpenCV

    # Convert back to RGB
    shifted_palette = cv2.cvtColor(palette_hsv, cv2.COLOR_HSV2RGB)[0]

    return shifted_palette

# Initialize the flame palette with a random hue shift
hue_shift = random.randint(0, 179)
FLAME_PALETTE = create_flame_palette(hue_shift)

def smooth_mask(current_mask, prev_mask, temporal_factor=0.8):
    """Apply temporal and spatial smoothing to the mask"""
    # Apply spatial smoothing
    kernel_size = 7
    current_mask = cv2.GaussianBlur(current_mask, (kernel_size, kernel_size), 0)

    # Threshold the mask to make it more definitive
    current_mask = cv2.threshold(current_mask, 0.5, 1, cv2.THRESH_BINARY)[1]

    # Apply morphological operations to fill holes and smooth edges
    kernel = np.ones((3, 3), np.uint8)
    current_mask = cv2.morphologyEx(current_mask, cv2.MORPH_CLOSE, kernel)
    current_mask = cv2.morphologyEx(current_mask, cv2.MORPH_OPEN, kernel)

    # Apply temporal smoothing if we have a previous mask
    if prev_mask is not None:
        return cv2.addWeighted(current_mask, 1 - temporal_factor, prev_mask, temporal_factor, 0)
    return current_mask

def generate_flame_effect(segmentation_mask):
    """Generate a flame effect within the segmentation mask."""
    global flame_buffer

    # Initialize flame_buffer if necessary
    if flame_buffer is None or flame_buffer.shape != segmentation_mask.shape:
        flame_buffer = np.zeros_like(segmentation_mask, dtype=np.float32)

    # Add random ignition points within the segmentation mask
    ignition_chance = 0.05  # Adjust for more or fewer ignition points
    random_ignition = (np.random.rand(*segmentation_mask.shape) < ignition_chance).astype(np.float32)
    flame_buffer += random_ignition * segmentation_mask

    # Ensure flame_buffer does not exceed maximum intensity
    np.clip(flame_buffer, 0, 1, out=flame_buffer)

    # Propagate flames to neighboring pixels
    flame_up = np.roll(flame_buffer, -1, axis=0)
    flame_left = np.roll(flame_buffer, -1, axis=1)
    flame_right = np.roll(flame_buffer, 1, axis=1)
    flame_down = np.roll(flame_buffer, 1, axis=0)

    flame_buffer = (
        flame_buffer +
        flame_up +
        flame_left +
        flame_right +
        flame_down
    ) / 5.0

    # Apply cooling
    cooling = 0.02  # Adjust for faster or slower cooling
    flame_buffer -= cooling

    # Apply segmentation mask
    flame_buffer *= segmentation_mask

    # Ensure flame_buffer stays within bounds
    np.clip(flame_buffer, 0, 1, out=flame_buffer)

    return flame_buffer

def apply_flame_effect(frame, segmentation_mask):
    """Apply flame effect to the frame using the segmentation mask."""
    # Generate flame effect
    flame_intensity = generate_flame_effect(segmentation_mask)

    # Map the flame intensity to color indices
    indices = np.clip(flame_intensity * (len(FLAME_PALETTE) - 1), 0, len(FLAME_PALETTE) - 1).astype(np.uint8)

    # Create the flame image
    flame_image = FLAME_PALETTE[indices]

    return flame_image

def draw_flame_view(frame):
    """Creates a visualization with flame effect."""
    global previous_mask

    # Flip the frame horizontally to create mirror effect
    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.segmentation_mask is not None:
        # Process segmentation mask
        current_mask = cv2.resize(results.segmentation_mask,
                                  (SCREEN_WIDTH, SCREEN_HEIGHT))

        # Apply temporal and spatial smoothing
        smoothed_mask = smooth_mask(current_mask, previous_mask)
        previous_mask = smoothed_mask.copy()

        # Apply flame effect
        flame_frame = apply_flame_effect(frame_rgb, smoothed_mask)

        # Convert to pygame surface
        flame_surface = pygame.surfarray.make_surface(flame_frame.swapaxes(0, 1))
        return flame_surface

    else:
        # Return an empty surface if no segmentation mask is available
        return pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

def main():
    """Main loop for flame camera effect display."""
    global FLAME_PALETTE
    running = True
    clock = pygame.time.Clock()

    # Initialize palette change timing
    palette_change_interval = random.randint(10, 15) * 1000  # milliseconds
    next_palette_change = pygame.time.get_ticks() + palette_change_interval

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
            flame_surface = draw_flame_view(frame)
            screen.blit(flame_surface, (0, 0))
            pygame.display.flip()

        except Exception as e:
            print(f"Error processing frame: {e}")
            continue

        # Check if it's time to change the palette
        current_time = pygame.time.get_ticks()
        if current_time >= next_palette_change:
            # Generate a new hue shift
            hue_shift = random.randint(0, 179)
            FLAME_PALETTE = create_flame_palette(hue_shift)
            # Schedule next palette change
            palette_change_interval = random.randint(10, 15) * 1000  # milliseconds
            next_palette_change = current_time + palette_change_interval

        clock.tick(30)

    cap.release()
    pygame.quit()
    pose.close()

if __name__ == '__main__':
    main()

import cv2
import mediapipe as mp
import pygame
import numpy as np
from scipy.ndimage import distance_transform_edt
import matplotlib.cm as cm
import random
import math
import threading

# Configuration variables for fractals
config = {
    'num_attractors': 10,
    'fade_effect': True,
    'fade_amount': 15,
    'scale_factor_min': 20,
    'scale_factor_max': 260,
    'point_color': (255, 255, 255),
    'background_color': (0, 0, 0),
    'line_thickness': 1,
    'attractor_iterations': 500,
    'max_fps': 60,
}

# Initialize pygame and set up display window.
pygame.init()

# Define the initial screen size.
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
width, height = SCREEN_WIDTH, SCREEN_HEIGHT
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption('Core-to-Edge Thermal Camera Effect with Fractal Background')

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

    # Create alpha channel from segmentation mask
    alpha_channel = (segmentation_mask * 255).astype(np.uint8)

    # Stack thermal_image and alpha_channel to create RGBA image
    thermal_image_rgba = np.dstack((thermal_image, alpha_channel))

    return thermal_image_rgba

def draw_thermal_view(frame, results):
    """Creates a visualization with thermal camera effect."""
    global previous_mask

    # Flip the frame horizontally.
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if results.segmentation_mask is not None:
        current_mask = results.segmentation_mask.astype(np.float32)

        # Apply smoothing.
        smoothed_mask = smooth_mask(current_mask, previous_mask)
        previous_mask = smoothed_mask.copy()

        # Apply thermal effect.
        thermal_frame_rgba = apply_thermal_effect(frame_rgb, smoothed_mask)

        # Convert to pygame surface.
        thermal_surface = pygame.image.frombuffer(
            thermal_frame_rgba.tobytes(), thermal_frame_rgba.shape[1::-1], 'RGBA').convert_alpha()
        return thermal_surface
    else:
        # Return a blank transparent surface if no segmentation mask is available.
        return pygame.Surface((frame.shape[1], frame.shape[0]), pygame.SRCALPHA)

# Class to represent a De Jong attractor
class DeJongAttractor:
    def __init__(self, anchorX, anchorY):
        self.anchorX = anchorX
        self.anchorY = anchorY

        # Randomize parameters for the attractor
        self.aOffset = random.uniform(-1, 1)
        self.bOffset = random.uniform(-1, 1)
        self.cOffset = random.uniform(-1, 1)
        self.dOffset = random.uniform(-1, 1)

        self.sx = random.uniform(-1, 1)
        self.sy = random.uniform(-1, 1)
        self.scale = random.uniform(config['scale_factor_min'], config['scale_factor_max'])

        self.msx = 1 / (10 + random.uniform(0, 1000))
        self.msy = 1 / (100 + random.uniform(0, 1000))

        self.offset_x = 0

        # Choose random trigonometric functions
        self.functions = [random.choice(['sin', 'cos']) for _ in range(4)]

        self.time = 0
        rotation = random.uniform(0, 2 * math.pi)
        self.cos_rot = math.cos(rotation)
        self.sin_rot = math.sin(rotation)

    def draw(self):
        global width, height

        self.time += 0.0001

        # Remove movement influence
        mx = 0
        my = 0

        # Update parameters without movement data
        a = 1.4 + self.aOffset + (mx + self.offset_x) * self.msx * 100
        b = -2.3 + self.bOffset + my * self.msy * 100
        c = 2.4 + self.cOffset + my * self.msy * 100
        d = -2.1 + self.dOffset - (mx + self.offset_x) * self.msx * 100

        x = self.sx + self.time
        y = self.sy + self.time

        for _ in range(config['attractor_iterations']):
            # Calculate new positions using De Jong equations
            newX = getattr(math, self.functions[0])(a * y) - getattr(math, self.functions[2])(b * x)
            newY = getattr(math, self.functions[1])(c * x) - getattr(math, self.functions[3])(d * y)
            x, y = newX, newY

            plotX = x * self.scale
            plotY = y * self.scale

            # Apply rotation
            rotatedX = plotX * self.cos_rot - plotY * self.sin_rot
            rotatedY = plotX * self.sin_rot + plotY * self.cos_rot

            # Apply translation
            screenX = rotatedX + self.anchorX
            screenY = rotatedY + self.anchorY

            # Draw the point if within screen bounds
            if 0 <= screenX < width and 0 <= screenY < height:
                if config['line_thickness'] <= 1:
                    screen.set_at((int(screenX), int(screenY)), config['point_color'])
                else:
                    pygame.draw.circle(screen, config['point_color'], (int(screenX), int(screenY)), config['line_thickness'])

# List to hold attractors
attractors = []

# Function to generate new attractors
def generate_attractors():
    global attractors
    attractors = [DeJongAttractor(random.uniform(0, width), random.uniform(0, height)) for _ in range(config['num_attractors'])]

# Generate initial attractors
generate_attractors()

def main():
    """Main loop for thermal camera effect display."""
    running = True
    clock = pygame.time.Clock()
    global SCREEN_WIDTH, SCREEN_HEIGHT, screen, width, height

    # Fill the screen with background color initially
    screen.fill(config['background_color'])

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                SCREEN_WIDTH, SCREEN_HEIGHT = event.size
                width, height = SCREEN_WIDTH, SCREEN_HEIGHT
                screen = pygame.display.set_mode(
                    (SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE
                )
                # Update attractors with new dimensions
                generate_attractors()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Regenerate attractors on mouse click
                generate_attractors()

        ret, frame = cap.read()
        if not ret:
            break

        try:
            # Process frame for pose detection and thermal effect
            frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            # Draw the fading effect if enabled
            if config['fade_effect']:
                fade_surface = pygame.Surface((width, height), pygame.SRCALPHA)
                fade_surface.fill((*config['background_color'], config['fade_amount']))
                screen.blit(fade_surface, (0, 0))
            else:
                # If no fade effect, clear the screen each frame
                screen.fill(config['background_color'])

            # Draw each attractor
            for attractor in attractors:
                attractor.draw()

            # Draw the thermal effect on top of the fractal background
            thermal_surface = draw_thermal_view(frame, results)
            thermal_surface = pygame.transform.scale(
                thermal_surface, (SCREEN_WIDTH, SCREEN_HEIGHT))
            screen.blit(thermal_surface, (0, 0))

            pygame.display.flip()
        except Exception as e:
            print(f"Error processing frame: {e}")
            continue

        clock.tick(config['max_fps'])

    cap.release()
    pygame.quit()
    pose.close()

if __name__ == '__main__':
    main()

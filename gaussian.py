import cv2
import mediapipe as mp
import pygame
import numpy as np
from scipy.ndimage import distance_transform_edt

# Initialize pygame and set up display window.
pygame.init()

# Define the initial screen size.
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption('Gradient Map with Silhouette Effect')

# Initialize MediaPipe Selfie Segmentation.
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Initialize webcam.
cap = cv2.VideoCapture(0)

# Initialize previous mask for temporal smoothing.
previous_mask = None


def create_gradient_colormap():
    """Creates a gradient colormap from blue to red."""
    colors = [
        (0, 0, 255),    # Blue
        (0, 255, 0),    # Green
        (255, 255, 0),  # Yellow
        (255, 0, 0),    # Red
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


GRADIENT_COLORMAP = create_gradient_colormap()


def smooth_mask(current_mask, prev_mask, temporal_factor=0.8):
    """Applies temporal and spatial smoothing to the mask."""
    # Apply strong spatial smoothing first.
    kernel_size = 15  # Increased kernel size for stronger smoothing.
    current_mask = cv2.GaussianBlur(current_mask, (kernel_size, kernel_size), 0)

    # Threshold the mask to make it more definitive.
    current_mask = cv2.threshold(current_mask, 0.5, 1, cv2.THRESH_BINARY)[1]

    # Apply morphological operations to fill holes and smooth edges.
    kernel = np.ones((5, 5), np.uint8)
    current_mask = cv2.morphologyEx(current_mask, cv2.MORPH_CLOSE, kernel)
    current_mask = cv2.morphologyEx(current_mask, cv2.MORPH_OPEN, kernel)

    # Apply temporal smoothing if we have a previous mask.
    if prev_mask is not None:
        return cv2.addWeighted(
            current_mask, 1 - temporal_factor, prev_mask, temporal_factor, 0)
    return current_mask


def create_silhouette_heatmap(segmentation_mask):
    """Creates a heatmap over the silhouette using distance transform."""
    # Create binary mask.
    binary_mask = (segmentation_mask > 0.5).astype(np.uint8)

    # Calculate distance from background for every point inside the mask.
    distance_map = distance_transform_edt(binary_mask)

    # Normalize the distance map to [0, 1].
    if distance_map.max() > 0:
        distance_map = distance_map / distance_map.max()

    # Apply gamma correction to adjust gradient intensity.
    gamma = 1.0
    heatmap = np.power(distance_map, gamma)

    # Smooth the heatmap for a softer gradient.
    heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)

    return heatmap


def apply_gradient_effect(frame, heatmap):
    """Applies gradient effect to the frame using the heatmap."""
    # Ensure heatmap is normalized to [0, 1]
    heatmap = np.clip(heatmap, 0, 1)

    # Convert heatmap to 0-255
    normalized_heatmap = (heatmap * 255).astype(np.uint8)

    # Create the gradient image
    gradient_image = np.zeros_like(frame)
    for i in range(3):
        gradient_image[:, :, i] = GRADIENT_COLORMAP[normalized_heatmap][:, :, i]

    # Create background (optional: you can use any background here)
    background = np.zeros_like(frame)  # Pure black background

    # Blend gradient image with background
    mask_3d = np.stack([heatmap > 0] * 3, axis=-1)
    result = gradient_image * mask_3d + background * (1 - mask_3d)

    return result.astype(np.uint8)


def draw_thermal_view(frame):
    """Creates a visualization with gradient map over the entire silhouette."""
    global previous_mask

    # Flip the frame horizontally to create mirror effect.
    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = selfie_segmentation.process(frame_rgb)

    # Get frame dimensions.
    frame_height, frame_width = frame.shape[:2]

    # Create surface for drawing.
    final_surface = pygame.Surface((frame_width, frame_height))
    final_surface.fill((0, 0, 0))  # Black background.

    if results.segmentation_mask is not None:
        # Process segmentation mask.
        current_mask = results.segmentation_mask

        # Apply temporal and spatial smoothing.
        smoothed_mask = smooth_mask(current_mask, previous_mask)
        previous_mask = smoothed_mask.copy()

        # Generate heatmap over the silhouette
        heatmap = create_silhouette_heatmap(smoothed_mask)

        # Apply gradient effect
        gradient_frame = apply_gradient_effect(frame_rgb, heatmap)

        # Convert to pygame surface and rotate to correct orientation.
        gradient_frame = np.rot90(gradient_frame, k=-1)
        gradient_surface = pygame.surfarray.make_surface(gradient_frame)
        final_surface.blit(gradient_surface, (0, 0))

    return np.rot90(pygame.surfarray.array3d(final_surface))


def main():
    """Main loop for gradient map with silhouette effect display."""
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
            output_array = draw_thermal_view(frame)
            surface = pygame.surfarray.make_surface(output_array.transpose(1, 0, 2))
            surface = pygame.transform.scale(surface, (SCREEN_WIDTH, SCREEN_HEIGHT))
            screen.blit(surface, (0, 0))
            pygame.display.flip()
        except Exception as e:
            print(f"Error processing frame: {e}")
            continue

        clock.tick(30)

    cap.release()
    pygame.quit()
    selfie_segmentation.close()


if __name__ == '__main__':
    main()

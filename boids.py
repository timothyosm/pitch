import cv2
import mediapipe as mp
import pygame
import numpy as np

# Initialize pygame and set up display window
pygame.init()

# Define the screen size
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Realistic Full-Body Flame Effect with Boids')

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

# Flame buffers for flame effect
flame_buffer_person = None
flame_buffer_boids = None

def create_flame_palette():
    """Create a palette of colors representing flames."""
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
    return palette

def create_blue_flame_palette():
    """Create a palette of colors representing blue flames."""
    colors = [
        (0, 0, 0),          # Black
        (0, 0, 7),          # Very dark blue
        (0, 0, 31),         # Dark blue
        (0, 0, 47),         # Darker blue
        (0, 0, 71),         # Blue
        (0, 0, 87),         # Bright blue
        (0, 0, 103),        # Brighter blue
        (0, 0, 119),        # Even brighter blue
        (0, 0, 143),        # Brightest blue
        (0, 0, 159),        # Even brighter blue
        (0, 0, 175),        # Very bright blue
        (0, 0, 191),        # Almost cyan
        (0, 0, 199),        # Even more cyan
        (0, 0, 223),        # Very cyan
        (0, 0, 223),        # Very cyan
        (0, 0, 223),        # Very cyan
        (0, 0, 215),        # Cyan
        (0, 0, 215),        # Cyan
        (0, 7, 215),        # Cyan with a hint of green
        (0, 15, 207),       # Cyan-green
        (0, 23, 207),       # Greenish cyan
        (0, 31, 207),       # Light greenish cyan
        (0, 39, 199),       # Lighter greenish cyan
        (0, 47, 199),       # Lighter greenish cyan
        (0, 55, 191),       # Light cyan-green
        (0, 63, 191),       # Light cyan-green
        (0, 71, 191),       # Lighter cyan-green
        (0, 71, 191),       # Lighter cyan-green
        (0, 79, 191),       # Lighter cyan-green
        (0, 79, 191),       # Lighter cyan-green
        (0, 87, 191),       # Very light cyan-green
        (0, 87, 183),       # Very light cyan-green
        (0, 95, 183),       # Light cyan-green
        (0, 103, 183),      # Pale cyan-green
        (0, 111, 207),      # Very pale cyan-green
        (0, 159, 223),      # Almost white cyan
        (0, 199, 239),      # Near white cyan
        (0, 255, 255)       # Cyan
    ]
    palette = np.array(colors, dtype=np.uint8)
    return palette

FLAME_PALETTE = create_flame_palette()
BLUE_FLAME_PALETTE = create_blue_flame_palette()

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

def generate_flame_effect(segmentation_mask, flame_buffer):
    """Generate a flame effect within the segmentation mask."""
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

def apply_flame_effect(frame, segmentation_mask, flame_palette=FLAME_PALETTE):
    """Apply flame effect to the frame using the segmentation mask."""
    # Generate flame effect
    flame_intensity = generate_flame_effect(segmentation_mask, None)

    # Map the flame intensity to color indices
    indices = np.clip(flame_intensity * (len(flame_palette) - 1), 0, len(flame_palette) - 1).astype(np.uint8)

    # Create the flame image
    flame_image = flame_palette[indices]

    return flame_image

# Boid class for boid simulation
MAX_SPEED = 4.0
MAX_FORCE = 0.1
PERCEPTION_RADIUS = 50
AVOIDANCE_WEIGHT = 1.0
ALIGNMENT_WEIGHT = 1.0
COHESION_WEIGHT = 1.0
SEPARATION_WEIGHT = 1.5

class Boid:
    def __init__(self, position, velocity):
        self.position = np.array(position, dtype=np.float32)
        self.velocity = np.array(velocity, dtype=np.float32)
        self.acceleration = np.zeros(2, dtype=np.float32)

    def update(self, boids, mask):
        self.acceleration = np.zeros(2, dtype=np.float32)
        self.flock(boids, mask)
        # Update velocity
        self.velocity += self.acceleration
        # Limit speed
        speed = np.linalg.norm(self.velocity)
        if speed > MAX_SPEED:
            self.velocity = (self.velocity / speed) * MAX_SPEED
        # Update position
        self.position += self.velocity
        # Wrap around edges
        self.position[0] %= SCREEN_WIDTH
        self.position[1] %= SCREEN_HEIGHT

    def flock(self, boids, mask):
        alignment = self.align(boids) * ALIGNMENT_WEIGHT
        cohesion = self.cohere(boids) * COHESION_WEIGHT
        separation = self.separate(boids) * SEPARATION_WEIGHT
        avoidance = self.avoid_obstacles(mask) * AVOIDANCE_WEIGHT
        self.acceleration += alignment + cohesion + separation + avoidance

    def align(self, boids):
        steering = np.zeros(2, dtype=np.float32)
        total = 0
        avg_velocity = np.zeros(2, dtype=np.float32)
        for other in boids:
            if other is not self:
                distance = np.linalg.norm(self.position - other.position)
                if distance < PERCEPTION_RADIUS:
                    avg_velocity += other.velocity
                    total += 1
        if total > 0:
            avg_velocity /= total
            # Steer towards average velocity
            steering = avg_velocity - self.velocity
            # Limit force
            if np.linalg.norm(steering) > MAX_FORCE:
                steering = (steering / np.linalg.norm(steering)) * MAX_FORCE
        return steering

    def cohere(self, boids):
        steering = np.zeros(2, dtype=np.float32)
        total = 0
        center_of_mass = np.zeros(2, dtype=np.float32)
        for other in boids:
            if other is not self:
                distance = np.linalg.norm(self.position - other.position)
                if distance < PERCEPTION_RADIUS:
                    center_of_mass += other.position
                    total += 1
        if total > 0:
            center_of_mass /= total
            desired = center_of_mass - self.position
            # Steer towards center of mass
            steering = desired - self.velocity
            # Limit force
            if np.linalg.norm(steering) > MAX_FORCE:
                steering = (steering / np.linalg.norm(steering)) * MAX_FORCE
        return steering

    def separate(self, boids):
        steering = np.zeros(2, dtype=np.float32)
        total = 0
        for other in boids:
            if other is not self:
                distance = np.linalg.norm(self.position - other.position)
                if distance < PERCEPTION_RADIUS / 2:
                    diff = self.position - other.position
                    diff /= distance
                    steering += diff
                    total += 1
        if total > 0:
            steering /= total
            # Limit force
            if np.linalg.norm(steering) > MAX_FORCE:
                steering = (steering / np.linalg.norm(steering)) * MAX_FORCE
        return steering

    def avoid_obstacles(self, mask):
        x = int(self.position[0])
        y = int(self.position[1])
        if 0 <= x < SCREEN_WIDTH and 0 <= y < SCREEN_HEIGHT:
            if mask[y, x] > 0.5:
                # Boid is inside the person mask, steer away
                # Compute gradient of the mask to get the direction
                gradient_x = mask[y, min(x+1, SCREEN_WIDTH-1)] - mask[y, max(x-1, 0)]
                gradient_y = mask[min(y+1, SCREEN_HEIGHT-1), x] - mask[max(y-1, 0), x]
                avoidance_force = np.array([gradient_x, gradient_y])
                if np.linalg.norm(avoidance_force) > 0:
                    avoidance_force = (avoidance_force / np.linalg.norm(avoidance_force)) * MAX_FORCE
                return avoidance_force
        return np.zeros(2)

def main():
    """Main loop for flame camera effect display."""
    running = True
    clock = pygame.time.Clock()

    # Initialize boids
    NUM_BOIDS = 50
    boids = [Boid(position=np.random.rand(2) * np.array([SCREEN_WIDTH, SCREEN_HEIGHT]),
                  velocity=(np.random.rand(2) - 0.5) * MAX_SPEED * 2)
             for _ in range(NUM_BOIDS)]

    # Initialize flame buffers
    global flame_buffer_person, flame_buffer_boids, previous_mask
    flame_buffer_person = None
    flame_buffer_boids = None
    previous_mask = None

    while running:
        # Event handling
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

                # Update boids
                for boid in boids:
                    boid.update(boids, smoothed_mask)

                # Create boid mask
                boid_mask = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH), dtype=np.float32)
                for boid in boids:
                    x = int(boid.position[0])
                    y = int(boid.position[1])
                    if 0 <= x < SCREEN_WIDTH and 0 <= y < SCREEN_HEIGHT:
                        cv2.circle(boid_mask, (x, y), 2, 1.0, -1)

                # Generate flame effects
                flame_buffer_person = generate_flame_effect(smoothed_mask, flame_buffer_person)
                flame_frame_person = apply_flame_effect(frame_rgb, smoothed_mask, flame_palette=FLAME_PALETTE)

                flame_buffer_boids = generate_flame_effect(boid_mask, flame_buffer_boids)
                flame_frame_boids = apply_flame_effect(frame_rgb, boid_mask, flame_palette=BLUE_FLAME_PALETTE)

                # Combine the two flame frames
                combined_flame_frame = np.maximum(flame_frame_person, flame_frame_boids)

                # Convert to pygame surface
                flame_surface = pygame.surfarray.make_surface(combined_flame_frame.swapaxes(0,1))

                screen.blit(flame_surface, (0, 0))
                pygame.display.flip()
            else:
                # If no segmentation mask is available, display original frame
                frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0,1))
                screen.blit(frame_surface, (0, 0))
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

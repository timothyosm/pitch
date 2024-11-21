import pygame
import random
import math
import threading
import cv2
import mediapipe as mp

# Configuration variables
config = {
    'num_attractors': 10,        # Number of attractors
    'fade_effect': True,         # Enable or disable the fading effect
    'fade_amount': 15,           # Transparency value for fading effect (0-255)
    'scale_factor_min': 20,      # Minimum scaling factor for attractors
    'scale_factor_max': 260,     # Maximum scaling factor for attractors
    'point_color': (255, 255, 255),  # Color of the attractor points (white)
    'background_color': (0, 0, 0),   # Background color (black)
    'line_thickness': 1,         # Thickness of the points/lines drawn
    'media_pipe_landmark': 'NOSE',    # Landmark to use for control ('NOSE', 'LEFT_SHOULDER', etc.)
    'attractor_iterations': 500, # Number of iterations per attractor draw call
    'max_fps': 60,               # Maximum frames per second
}

# Initialize Pygame
pygame.init()

# Set initial window dimensions
width, height = 800, 600
screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
pygame.display.set_caption('De Jong Attractors with MediaPipe Integration')

# Initialize movement variables
movement_data = {'mx': 0, 'my': 0}

# Lock for thread-safe access to movement_data
data_lock = threading.Lock()

# MediaPipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)  # Open default webcam

# Map landmark names to MediaPipe PoseLandmark enum
landmark_mapping = {
    'NOSE': mp_pose.PoseLandmark.NOSE,
    'LEFT_SHOULDER': mp_pose.PoseLandmark.LEFT_SHOULDER,
    'RIGHT_SHOULDER': mp_pose.PoseLandmark.RIGHT_SHOULDER,
    'LEFT_ELBOW': mp_pose.PoseLandmark.LEFT_ELBOW,
    'RIGHT_ELBOW': mp_pose.PoseLandmark.RIGHT_ELBOW,
    # Add other landmarks as needed
}

def mediapipe_thread():
    global movement_data, cap, pose
    while True:
        success, image = cap.read()
        if not success:
            continue

        # Convert the BGR image to RGB and process it with MediaPipe Pose
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            # Extract the specified landmark
            landmark_name = config['media_pipe_landmark']
            landmark_enum = landmark_mapping.get(landmark_name, mp_pose.PoseLandmark.NOSE)
            landmark = results.pose_landmarks.landmark[landmark_enum]
            # Normalize coordinates to be in the range [-1, 1]
            mx = landmark.x * 2 - 1
            my = landmark.y * 2 - 1

            with data_lock:
                movement_data['mx'] = mx
                movement_data['my'] = my

# Start MediaPipe in a separate thread
threading.Thread(target=mediapipe_thread, daemon=True).start()

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
        global movement_data, width, height

        self.time += 0.0001

        with data_lock:
            mx = movement_data['mx']
            my = movement_data['my']

        # Update parameters based on movement data
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

# Main loop setup
running = True
clock = pygame.time.Clock()

# Fill the screen with background color initially
screen.fill(config['background_color'])

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            generate_attractors()

        elif event.type == pygame.VIDEORESIZE:
            width, height = event.w, event.h
            screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
            screen.fill(config['background_color'])
            generate_attractors()

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

    # Update the display and maintain specified FPS
    pygame.display.flip()
    clock.tick(config['max_fps'])

# Clean up resources
cap.release()
cv2.destroyAllWindows()
pygame.quit()

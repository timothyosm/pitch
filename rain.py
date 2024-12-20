import cv2
import numpy as np
import pygame
import random

class Particle:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.x = random.randint(0, width)
        self.y = 0
        self.speed = 0
        self.velocity = random.random() * 0.7
        self.size = random.random() * 2 + 0.1

    def update(self, grid):
        # Get grid position
        grid_x = min(int(self.x / detail), len(grid[0]) - 1)
        grid_y = min(int(self.y / detail), len(grid) - 1)
        self.speed = grid[grid_y][grid_x]
        
        # Update position
        movement = (2.5 - self.speed) + self.velocity
        self.y += movement
        
        # Reset if particle reaches bottom
        if self.y >= self.height:
            self.y = 0
            self.x = random.randint(0, self.width)

    def draw(self, screen):
        alpha = int(self.speed * 255 * 0.3)
        alpha = max(0, min(255, alpha))  # Clamp between 0 and 255
        
        # Create a temporary surface for the particle
        particle_surface = pygame.Surface((int(self.size * 2), int(self.size * 2)), pygame.SRCALPHA)
        pygame.draw.circle(particle_surface, (255, 255, 255, alpha), 
                         (int(self.size), int(self.size)), int(self.size))
        screen.blit(particle_surface, (int(self.x - self.size), int(self.y - self.size)))

def calculate_brightness(frame, x, y):
    b, g, r = frame[y, x]
    return np.sqrt(
        (r * r) * 0.299 +
        (g * g) * 0.587 +
        (b * b) * 0.114
    ) / 100

# Initialize
pygame.init()
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Webcam Particle Effect")
clock = pygame.time.Clock()

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Particles setup
detail = 10
number_of_particles = 50000
particles = [Particle(width, height) for _ in range(number_of_particles)]

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    # Read webcam
    ret, frame = cap.read()
    if not ret:
        continue
        
    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Create brightness grid
    grid = []
    for y in range(0, height, detail):
        row = []
        for x in range(0, width, detail):
            if y < frame.shape[0] and x < frame.shape[1]:
                brightness = calculate_brightness(frame, x, y)
                row.append(brightness)
            else:
                row.append(0)
        grid.append(row)

    # Draw background with alpha for trail effect
    dark_surface = pygame.Surface(screen.get_size())
    dark_surface.fill((0, 0, 0))
    dark_surface.set_alpha(25)
    screen.blit(dark_surface, (0, 0))

    # Update and draw particles
    for particle in particles:
        particle.update(grid)
        particle.draw(screen)

    pygame.display.flip()
    clock.tick(60)

# Cleanup
cap.release()
pygame.quit()
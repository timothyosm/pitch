import pygame
import random
import math

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH = 800
HEIGHT = 600

# Create the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Optimized Boids Simulation")

# Clock to control frame rate
clock = pygame.time.Clock()

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Boid settings
NUM_BOIDS = 300  # Increase this number based on performance
MAX_SPEED = 4
MAX_FORCE = 0.1
PERCEPTION_RADIUS = 50

# Grid settings for spatial partitioning
GRID_SIZE = 50

# Boid class
class Boid:
    def __init__(self):
        self.position = pygame.Vector2(random.uniform(0, WIDTH), random.uniform(0, HEIGHT))
        angle = random.uniform(0, 2 * math.pi)
        self.velocity = pygame.Vector2(math.cos(angle), math.sin(angle))
        self.velocity.scale_to_length(random.uniform(2, MAX_SPEED))
        self.acceleration = pygame.Vector2(0, 0)

    def edges(self):
        # Bounce off the edges
        if self.position.x >= WIDTH:
            self.position.x = WIDTH
            self.velocity.x *= -1
        elif self.position.x <= 0:
            self.position.x = 0
            self.velocity.x *= -1

        if self.position.y >= HEIGHT:
            self.position.y = HEIGHT
            self.velocity.y *= -1
        elif self.position.y <= 0:
            self.position.y = 0
            self.velocity.y *= -1

    def align(self, boids):
        steering = pygame.Vector2(0, 0)
        total = 0
        for other in boids:
            if self.position.distance_squared_to(other.position) < PERCEPTION_RADIUS ** 2:
                steering += other.velocity
                total += 1
        if total > 0:
            steering /= total
            steering -= self.velocity
            if steering.length_squared() > MAX_FORCE ** 2:
                steering.scale_to_length(MAX_FORCE)
            return steering
        else:
            return pygame.Vector2(0, 0)

    def cohesion(self, boids):
        steering = pygame.Vector2(0, 0)
        total = 0
        for other in boids:
            if self.position.distance_squared_to(other.position) < PERCEPTION_RADIUS ** 2:
                steering += other.position
                total += 1
        if total > 0:
            steering /= total
            steering -= self.position
            if steering.length_squared() > 0:
                steering.scale_to_length(MAX_SPEED)
                steering -= self.velocity
                if steering.length_squared() > MAX_FORCE ** 2:
                    steering.scale_to_length(MAX_FORCE)
                return steering
            else:
                return pygame.Vector2(0, 0)
        else:
            return pygame.Vector2(0, 0)

    def separation(self, boids):
        steering = pygame.Vector2(0, 0)
        total = 0
        for other in boids:
            distance = self.position.distance_to(other.position)
            if 0 < distance < PERCEPTION_RADIUS / 2:
                diff = self.position - other.position
                diff /= distance
                steering += diff
                total += 1
        if total > 0:
            steering /= total
            if steering.length_squared() > 0:
                steering.scale_to_length(MAX_SPEED)
                steering -= self.velocity
                if steering.length_squared() > MAX_FORCE ** 2:
                    steering.scale_to_length(MAX_FORCE)
                return steering
            else:
                return pygame.Vector2(0, 0)
        else:
            return pygame.Vector2(0, 0)

    def update(self, boids):
        alignment = self.align(boids)
        cohesion = self.cohesion(boids)
        separation = self.separation(boids)

        self.acceleration = pygame.Vector2(0, 0)
        self.acceleration += alignment
        self.acceleration += cohesion
        self.acceleration += separation

        self.velocity += self.acceleration
        if self.velocity.length_squared() > MAX_SPEED ** 2:
            self.velocity.scale_to_length(MAX_SPEED)
        self.position += self.velocity
        self.edges()

    def draw(self, screen):
        # Draw boid as a small circle (dot)
        pygame.draw.circle(screen, WHITE, (int(self.position.x), int(self.position.y)), 2)

# Function to create spatial grid
def create_spatial_grid(boids):
    grid = {}
    for boid in boids:
        cell_x = int(boid.position.x // GRID_SIZE)
        cell_y = int(boid.position.y // GRID_SIZE)
        key = (cell_x, cell_y)
        if key not in grid:
            grid[key] = []
        grid[key].append(boid)
    return grid

# Function to get nearby boids
def get_nearby_boids(boid, grid):
    nearby_boids = []
    cell_x = int(boid.position.x // GRID_SIZE)
    cell_y = int(boid.position.y // GRID_SIZE)
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            key = (cell_x + dx, cell_y + dy)
            if key in grid:
                nearby_boids.extend(grid[key])
    return nearby_boids

# Create boids
boids = [Boid() for _ in range(NUM_BOIDS)]

# Main loop
running = True
while running:
    clock.tick(60)
    screen.fill(BLACK)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Create spatial grid for this frame
    grid = create_spatial_grid(boids)

    # Update and draw boids
    for boid in boids:
        nearby_boids = get_nearby_boids(boid, grid)
        boid.update(nearby_boids)
        boid.draw(screen)

    # Update the display
    pygame.display.flip()

pygame.quit()

import pygame
import numpy as np
from noise import pnoise3
import random

class ParticleSystem:
    def __init__(self):
        pygame.init()
        self.width = 1024
        self.height = 768
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Particle System")
        
        self.num_particles = 100000
        self.positions = np.random.uniform(-1, 1, (self.num_particles, 3)).astype(np.float32)
        self.velocities = np.random.uniform(-0.1, 0.1, (self.num_particles, 3)).astype(np.float32)
        
        self.clock = pygame.time.Clock()
        self.start_time = pygame.time.get_ticks()

    def update_particles(self, time):
        # Add random offset to each particle
        random_offset = np.random.uniform(-0.1, 0.1, (self.num_particles, 3))
        offset = np.array([time * 5, 0, 0]) + random_offset
        
        # Update velocities using Perlin noise
        for i in range(self.num_particles):
            pos = self.positions[i]
            noise_scale = 3 + np.cos(time)
            
            # Generate Perlin noise for velocity changes
            noise_x = pnoise3(pos[0] * noise_scale + offset[i][0],
                            pos[1] * noise_scale + offset[i][1],
                            pos[2] * noise_scale + offset[i][2])
            noise_y = pnoise3(pos[0] * 2 + 10 - offset[i][0],
                            pos[1] * 2 + 10 - offset[i][1],
                            pos[2] * 2 + 10 - offset[i][2])
            
            self.velocities[i][0] += noise_x * 0.2
            self.velocities[i][1] += noise_y * 0.2
            
        # Dampen velocities
        self.velocities *= 0.9
        np.clip(self.velocities, -1, 1, out=self.velocities)
        
        # Update positions
        self.positions += self.velocities * 0.01 + random_offset * 0.05
        
        # Wrap particles around screen edges
        for i in range(3):
            wrap_indices = self.positions[:, i] < -1.1
            self.positions[wrap_indices, i] = 1.0
            self.velocities[wrap_indices, i] *= -1.0
            
            wrap_indices = self.positions[:, i] > 1.1
            self.positions[wrap_indices, i] = -1.0
            self.velocities[wrap_indices, i] *= -1.0

    def draw(self):
        self.screen.fill((0, 0, 0))
        
        # Convert positions from [-1,1] to screen coordinates
        screen_positions = np.zeros((self.num_particles, 2))
        screen_positions[:, 0] = (self.positions[:, 0] + 1) * self.width / 2
        screen_positions[:, 1] = (self.positions[:, 1] + 1) * self.height / 2
        
        # Draw particles
        for pos in screen_positions:
            if 0 <= pos[0] < self.width and 0 <= pos[1] < self.height:
                pygame.draw.circle(self.screen, (255, 255, 255), pos.astype(int), 1)
        
        pygame.display.flip()

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            current_time = (pygame.time.get_ticks() - self.start_time) * 0.0001
            self.update_particles(current_time)
            self.draw()
            self.clock.tick(60)
        
        pygame.quit()

if __name__ == "__main__":
    particle_system = ParticleSystem()
    particle_system.run()
import pygame
import numpy as np
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import pygame_gui
import time

class CubeWaveApp:
    def __init__(self):
        # Initialize pygame and OpenGL
        pygame.init()
        self.width, self.height = 800, 600
        pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("🎲 Cubes Wave")

        # OpenGL settings
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION,  (0, 1, 2, 0))
        glClearColor(0.93, 0.10, 0.13, 1)  # Background color

        # Camera settings
        self.camera_radius = 100
        self.camera_theta = np.pi / 4
        self.camera_phi = np.pi / 4
        self.camera_center = np.array([0, 0, 0])
        self.mouse_down = False
        self.last_mouse_pos = None

        # Animation parameters
        self.angle = 0
        self.grid_size = 30
        self.velocity = 0.1
        self.amplitude = -1
        self.wave_length = 242

        # Cube data
        self.cubes = []
        self.create_cubes()

        # GUI manager
        self.manager = pygame_gui.UIManager((self.width, self.height))
        self.create_gui()

        # FPS calculation
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 18)

    def create_gui(self):
        # Sliders for amplitude, velocity, wavelength, and grid size
        self.amplitude_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((10, 10), (200, 20)),
            start_value=self.amplitude,
            value_range=(-10, 0.2),
            manager=self.manager)
        self.amplitude_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((220, 10), (100, 20)),
            text=f'Amplitude: {self.amplitude:.2f}',
            manager=self.manager)

        self.velocity_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((10, 40), (200, 20)),
            start_value=self.velocity,
            value_range=(0, 0.5),
            manager=self.manager)
        self.velocity_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((220, 40), (100, 20)),
            text=f'Velocity: {self.velocity:.2f}',
            manager=self.manager)

        self.wavelength_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((10, 70), (200, 20)),
            start_value=self.wave_length,
            value_range=(100, 500),
            manager=self.manager)
        self.wavelength_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((220, 70), (100, 20)),
            text=f'Wavelength: {self.wave_length:.0f}',
            manager=self.manager)

        self.grid_size_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((10, 100), (200, 20)),
            start_value=self.grid_size,
            value_range=(24, 150),
            manager=self.manager)
        self.grid_size_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((220, 100), (100, 20)),
            text=f'Grid Size: {self.grid_size:.0f}',
            manager=self.manager)

    def create_cubes(self):
        # Create cube positions
        self.cubes = []
        size = 1
        height = 5
        offset = self.grid_size * size / 2
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x = i * size - offset
                z = j * size - offset
                self.cubes.append([x, 0, z, 1])  # x, y, z, scale_y

    def draw_cubes(self):
        # Draw cubes with wave animation
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, [0.1, 0.39, 0.93, 1])
        glPushMatrix()
        for cube in self.cubes:
            x, _, z, scale_y = cube
            glPushMatrix()
            glTranslatef(x, 2.5 * scale_y, z)
            glScalef(1, scale_y, 1)
            glutSolidCube(5)
            glPopMatrix()
        glPopMatrix()

    def update_wave(self):
        # Update cube scales based on wave function
        for idx, cube in enumerate(self.cubes):
            x, _, z, _ = cube
            distance = np.hypot(x, z)
            offset = np.interp(distance, [0, self.wave_length], [-100, 100])
            angle = self.angle + offset
            scale_y = np.interp(np.sin(angle), [-1, self.amplitude], [0.001, 1])
            cube[3] = scale_y
        self.angle -= self.velocity

    def run(self):
        running = True
        while running:
            time_delta = self.clock.tick(60)/1000.0
            fps = self.clock.get_fps()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.mouse_down = True
                        self.last_mouse_pos = pygame.mouse.get_pos()
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.mouse_down = False
                elif event.type == pygame.MOUSEMOTION:
                    if self.mouse_down:
                        x, y = pygame.mouse.get_pos()
                        dx = x - self.last_mouse_pos[0]
                        dy = y - self.last_mouse_pos[1]
                        self.camera_theta -= dx * 0.01
                        self.camera_phi -= dy * 0.01
                        self.camera_phi = max(0.1, min(np.pi - 0.1, self.camera_phi))
                        self.last_mouse_pos = (x, y)

                self.manager.process_events(event)

            # Update GUI
            self.manager.update(time_delta)
            amplitude = self.amplitude_slider.get_current_value()
            if amplitude != self.amplitude:
                self.amplitude = amplitude
                self.amplitude_label.set_text(f'Amplitude: {self.amplitude:.2f}')

            velocity = self.velocity_slider.get_current_value()
            if velocity != self.velocity:
                self.velocity = velocity
                self.velocity_label.set_text(f'Velocity: {self.velocity:.2f}')

            wave_length = self.wavelength_slider.get_current_value()
            if wave_length != self.wave_length:
                self.wave_length = wave_length
                self.wavelength_label.set_text(f'Wavelength: {self.wave_length:.0f}')

            grid_size = self.grid_size_slider.get_current_value()
            if grid_size != self.grid_size:
                self.grid_size = int(grid_size)
                self.grid_size_label.set_text(f'Grid Size: {self.grid_size}')
                self.create_cubes()

            self.update_wave()

            # Clear screen
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Set camera
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            x = self.camera_radius * np.sin(self.camera_phi) * np.cos(self.camera_theta)
            y = self.camera_radius * np.cos(self.camera_phi)
            z = self.camera_radius * np.sin(self.camera_phi) * np.sin(self.camera_theta)
            gluLookAt(x, y, z, *self.camera_center, 0, 1, 0)

            # Draw scene
            self.draw_cubes()

            # Draw GUI
            glDisable(GL_DEPTH_TEST)
            glDisable(GL_LIGHTING)
            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            glOrtho(0, self.width, self.height, 0, -1, 1)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()

            self.manager.draw_ui(pygame.display.get_surface())

            # Display FPS
            fps_text = self.font.render(f'FPS: {fps:.2f}', True, pygame.Color('white'))
            pygame.display.get_surface().blit(fps_text, (self.width - 100, 10))

            glEnable(GL_DEPTH_TEST)
            glEnable(GL_LIGHTING)
            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)

            # Update display
            pygame.display.flip()

            # Handle window resize
            self.handle_resize()

        pygame.quit()

    def handle_resize(self):
        new_size = pygame.display.get_window_size()
        if (self.width, self.height) != new_size:
            self.width, self.height = new_size
            glViewport(0, 0, self.width, self.height)
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective(45, (self.width / self.height), 0.1, 1000.0)
            glMatrixMode(GL_MODELVIEW)
            self.manager.set_window_resolution(new_size)

if __name__ == '__main__':
    # Import GLUT for solid cube
    from OpenGL.GLUT import glutInit, glutSolidCube
    glutInit()
    app = CubeWaveApp()
    app.run()

import pygame
from noise import pnoise2
import random
import math
import time
import colorsys

def hsla_to_rgba(h, s, l, a):
    h = h % 360
    h = h / 360.0
    s = s / 100.0
    l = l / 100.0
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    r = int(r * 255)
    g = int(g * 255)
    b = int(b * 255)
    return (r, g, b, int(a * 255))

def ease_in_out_quad(t):
    if t < 0.5:
        return 2 * t * t
    else:
        return -1 + (4 - 2 * t) * t

def draw_dot(dot, screen):
    x = dot['x']
    y = dot['y']
    r = dot['r']
    h = dot['h']
    a = dot['a']
    # Convert hsla to rgba
    color = hsla_to_rgba(h, 100, 50, a)
    # Create a surface with per-pixel alpha
    circle_surface = pygame.Surface((int(2*r), int(2*r)), pygame.SRCALPHA)
    pygame.draw.circle(circle_surface, color, (int(r), int(r)), int(r))
    # Blit the circle surface onto the main screen
    screen.blit(circle_surface, (x - r, y - r))

def main():
    pygame.init()
    cw = ch = 1200
    screen = pygame.display.set_mode((cw, ch))
    pygame.display.set_caption("Perlin Noise Dots")
    clock = pygame.time.Clock()

    dots = []
    step = 12
    noise_offset_x = random.uniform(0, 1000)
    noise_offset_y = random.uniform(0, 1000)

    for x in range(0, cw, step):
        for y in range(0, ch, step):
            r = (pnoise2((x + noise_offset_x)/200.0, (y + noise_offset_y)/200.0) + 1) * 5
            r = max(0.5, min(10, r))
            h = random.uniform(300, 400)
            dot = {
                'r': r,
                'x': x + 5,
                'y': y + 5,
                'h': h,
                'a': 1,
                'h0': h,
                'r0': r,
                'h1': h,
                'r1': r
            }
            dots.append(dot)

    total_animation_duration = 4.0  # seconds
    animation_start_time = time.time()
    noise_seed_time = animation_start_time
    noise_seed_interval = 3.0

    running = True
    while running:
        current_time = time.time()
        animation_elapsed = current_time - animation_start_time
        animation_progress = (animation_elapsed % total_animation_duration) / total_animation_duration
        phase = animation_progress * 2  # phase goes from 0 to 2
        if phase > 1.0:
            phase = 2.0 - phase  # yoyo effect

        # Update dots
        t = phase
        eased_t = ease_in_out_quad(t)

        for dot in dots:
            dot['h'] = dot['h0'] + (dot['h1'] - dot['h0']) * eased_t
            dot['r'] = dot['r0'] + (dot['r1'] - dot['r0']) * eased_t

        # Check if we need to start a new animation cycle
        if animation_elapsed > total_animation_duration:
            animation_start_time = current_time
            # Update initial and target properties
            for dot in dots:
                dot['h0'] = dot['h1']
                dot['r0'] = dot['r1']
                dot['h1'] = random.uniform(180, 240)
                dot['r1'] = 0.5

        # Randomize noise seed every 3 seconds
        if current_time - noise_seed_time > noise_seed_interval:
            noise_seed_time = current_time
            noise_offset_x = random.uniform(0, 1000)
            noise_offset_y = random.uniform(0, 1000)
            # Update the dots
            for dot in dots:
                dot['r'] = (pnoise2((dot['x'] + noise_offset_x)/200.0, (dot['y'] + noise_offset_y)/200.0) + 1) * 5
                dot['r'] = max(0.5, min(20, dot['r']))
                dot['h'] = random.uniform(300, 400)
                # Reset animation
                dot['h0'] = dot['h']
                dot['r0'] = dot['r']
                dot['h1'] = random.uniform(180, 240)
                dot['r1'] = 0.5
            animation_start_time = current_time

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Drawing
        screen.fill((9, 9, 34))  # Background color #090922

        for dot in dots:
            draw_dot(dot, screen)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == '__main__':
    main()

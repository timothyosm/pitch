import pygame
import sys
from noise import pnoise3
import sounddevice as sd
import numpy as np
import threading

# Configuration
WIDTH, HEIGHT = 300, 800     # Window size
SCALE = 5                     # Size of each grid cell
OCTAVES = 1                   # Number of noise octaves (static)
PERSISTENCE = 1               # Noise persistence (static)
LACUNARITY = 2.0              # Noise lacunarity (static)
DEFAULT_SPEED = 0.00000000000001  # Base speed of animation
FPS = 60                      # Frames per second

# Speed Multiplier
SPEED_MULTIPLIER = 1/8       # Reduce speed to 1/8th

# Audio Configuration
AUDIO_RATE = 44100            # Sampling rate
AUDIO_CHUNK = 1024            # Number of samples per chunk

# Initialize global variables for audio processing
audio_data = np.zeros(AUDIO_CHUNK)
audio_lock = threading.Lock()

def hex_to_rgb(hex_color):
    """
    Converts a hex color string to an RGB tuple.
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def generate_violet_palette(num_colors=256):
    """
    Generates a smooth Violet color palette based on predefined color stops.
    """
    # Define the color stops with their corresponding levels
    color_stops = [
        (50, "#000000"),
        (100, "#000000"),
        (200, "#000000"),
        (300, "#000000"),
        (400, "#000000"),
        (500, "#f43f5e"),
        (600, "#e11d48"),
        (700, "#be123c"),
        (800, "#9f1239"),
        (900, "#881337"),
        (950, "#4c0519"),
    ]

    # Convert hex codes to RGB
    color_stops_rgb = [(level, hex_to_rgb(hex_code)) for level, hex_code in color_stops]

    # Sort the color stops by level just in case
    color_stops_rgb.sort(key=lambda x: x[0])

    # Normalize the levels to [0, 1]
    levels = [level for level, _ in color_stops_rgb]
    min_level = min(levels)
    max_level = max(levels)
    normalized_stops = [( (level - min_level) / (max_level - min_level), color) for level, color in color_stops_rgb]

    # Initialize the palette
    palette = []

    # Current stop index
    current_stop = 0

    for i in range(num_colors):
        # Normalize index to [0,1]
        t = i / (num_colors - 1)

        # Find the two stops t is between
        while current_stop < len(normalized_stops) - 1 and t > normalized_stops[current_stop + 1][0]:
            current_stop += 1

        # Get the two colors to interpolate between
        t0, color0 = normalized_stops[current_stop]
        if current_stop < len(normalized_stops) - 1:
            t1, color1 = normalized_stops[current_stop + 1]
        else:
            t1, color1 = normalized_stops[current_stop]

        # Compute the ratio between the two stops
        if t1 - t0 == 0:
            ratio = 0
        else:
            ratio = (t - t0) / (t1 - t0)

        # Interpolate RGB values
        r = int(color0[0] + (color1[0] - color0[0]) * ratio)
        g = int(color0[1] + (color1[1] - color0[1]) * ratio)
        b = int(color0[2] + (color1[2] - color0[2]) * ratio)

        # Append to palette
        palette.append((r, g, b))

    return palette

def audio_callback(indata, frames, time, status):
    """
    This callback is called for each audio block.
    It updates the global audio_data array with the latest audio samples.
    """
    global audio_data
    if status:
        print(status, file=sys.stderr)
    with audio_lock:
        # Convert to mono by averaging channels if necessary
        if indata.shape[1] > 1:
            audio_data = indata.mean(axis=1)
        else:
            audio_data = indata[:, 0]

def start_audio_stream():
    """
    Starts the audio input stream in a separate thread.
    """
    stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=AUDIO_RATE, blocksize=AUDIO_CHUNK)
    stream.start()
    return stream

def map_audio_to_speed():
    """
    Analyzes the current audio_data and maps it to the SPEED parameter.
    Returns the updated speed.
    """
    with audio_lock:
        data = np.copy(audio_data)

    # Compute the RMS (volume)
    rms = np.sqrt(np.mean(data**2))
    # Normalize RMS to [0, 1]
    volume = min(rms * 1000, 1.0)  # Adjust multiplier as needed for sensitivity

    # Scale the speed by SPEED_MULTIPLIER
    # Map volume to speed
    # For example, SPEED = DEFAULT_SPEED + (volume * 0.05) * SPEED_MULTIPLIER
    speed = DEFAULT_SPEED + (volume * 0.05) * SPEED_MULTIPLIER
    return speed

def interpolate_color(value, palette):
    """
    Maps a normalized value [0,1] to a color in the given palette.
    """
    index = int(value * (len(palette) - 1))
    index = max(0, min(index, len(palette) - 1))
    return palette[index]

def main():
    global DEFAULT_SPEED

    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Perlin Noise Morphing Background with Sound-Controlled Speed")
    clock = pygame.time.Clock()

    # Start audio stream
    stream = start_audio_stream()

    # Generate the Violet palette
    color_palette = generate_violet_palette()

    # Calculate number of columns and rows
    cols = WIDTH // SCALE + 1
    rows = HEIGHT // SCALE + 1

    z = 0.0  # Time component for animation

    try:
        # Main loop
        running = True
        while running:
            clock.tick(FPS)  # Limit to FPS

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Map audio input to SPEED
            SPEED = map_audio_to_speed()

            # Fill the screen with black initially
            screen.fill((0, 0, 0))

            # Draw the grid
            for y in range(rows):
                for x in range(cols):
                    # Calculate the position of the current grid cell
                    rect_x = x * SCALE
                    rect_y = y * SCALE

                    # Get Perlin noise value for the current position and time
                    noise_val = pnoise3(x * 0.1, y * 0.1, z,
                                        octaves=OCTAVES,
                                        persistence=PERSISTENCE,
                                        lacunarity=LACUNARITY,
                                        repeatx=1024,
                                        repeaty=1024,
                                        base=0)

                    # Normalize noise value from [-1, 1] to [0, 1]
                    noise_norm = (noise_val + 1) / 2

                    # Map normalized noise to violet color
                    color = interpolate_color(noise_norm, color_palette)

                    # Draw the rectangle
                    rect = pygame.Rect(rect_x, rect_y, SCALE, SCALE)
                    pygame.draw.rect(screen, color, rect)

            # Update the display
            pygame.display.flip()

            # Increment z with the scaled SPEED
            z += SPEED

    except KeyboardInterrupt:
        # Allow the user to exit with Ctrl+C
        pass
    finally:
        # Clean up
        stream.stop()
        stream.close()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    main()

import pygame
import random
import sys
from collections import deque

# Grid dimensions
GRID_WIDTH = 20
GRID_HEIGHT = 20
CELL_SIZE = 32  # Size of each cell in pixels

# Screen dimensions
SCREEN_WIDTH = GRID_WIDTH * CELL_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * CELL_SIZE + 100  # Extra space for UI

# Colors (R, G, B)
GREEN = (34, 139, 34)    # Trees
GRAY = (128, 128, 128)   # Rocks
RED = (255, 0, 0)        # Berry bushes
BROWN = (139, 69, 19)    # Empty space
BLUE = (0, 0, 255)       # NPC
WHITE = (255, 255, 255)  # Text
BLACK = (0, 0, 0)        # Background

# Hunger settings
MAX_HUNGER = 100
HUNGER_DEPLETION_RATE = 0.05  # Depletion per frame
HUNGER_RESTORE_AMOUNT = 20    # Restored per successful forage
FORAGE_TIME = 120             # Frames to forage (assuming 60 FPS, 2 seconds)
FORAGE_SUCCESS_RATE = 0.25    # 25% chance of successful forage

def generate_forest():
    grid = []
    for y in range(GRID_HEIGHT):
        row = []
        for x in range(GRID_WIDTH):
            rand = random.random()
            if rand < 0.1:
                cell = {'type': 'tree'}
            elif rand < 0.2:
                cell = {'type': 'rock'}
            elif rand < 0.25:
                cell = {'type': 'berry', 'foraged': False}
            else:
                cell = {'type': 'empty'}
            row.append(cell)
        grid.append(row)
    return grid

class NPC:
    def __init__(self, grid):
        self.grid = grid
        self.hunger = MAX_HUNGER
        self.position = self.spawn_npc()
        self.path = []
        self.foraging = False
        self.foraging_counter = 0
        self.berries_collected = 0

    def spawn_npc(self):
        while True:
            x = random.randint(0, GRID_WIDTH - 1)
            y = random.randint(0, GRID_HEIGHT - 1)
            if self.grid[y][x]['type'] == 'empty':
                return x, y

    def find_nearest_berry(self):
        visited = [[False for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        queue = deque()
        queue.append((self.position, []))
        while queue:
            (x, y), path = queue.popleft()
            if visited[y][x]:
                continue
            visited[y][x] = True
            cell = self.grid[y][x]
            if cell['type'] == 'berry' and not cell['foraged']:
                return path + [(x, y)]
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
                    neighbor_cell = self.grid[ny][nx]
                    if neighbor_cell['type'] in ['empty', 'berry'] and not visited[ny][nx]:
                        queue.append(((nx, ny), path + [(nx, ny)]))
        return []

    def move(self):
        if self.foraging:
            self.foraging_counter += 1
            if self.foraging_counter >= FORAGE_TIME:
                self.foraging = False
                self.foraging_counter = 0
                if random.random() < FORAGE_SUCCESS_RATE:
                    self.berries_collected += 1
                    self.hunger = min(self.hunger + HUNGER_RESTORE_AMOUNT, MAX_HUNGER)
                    self.grid[self.position[1]][self.position[0]]['foraged'] = True
                    display_message("Foraging successful!")
                else:
                    display_message("Foraging failed.")
        else:
            if not self.path:
                self.path = self.find_nearest_berry()
                if not self.path:
                    display_message("No berries left. NPC is wandering.")
                    self.random_move()
                    return
            next_pos = self.path.pop(0)
            self.position = next_pos
            cell = self.grid[self.position[1]][self.position[0]]
            if cell['type'] == 'berry' and not cell['foraged']:
                self.foraging = True

    def random_move(self):
        x, y = self.position
        possible_moves = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
                cell_type = self.grid[ny][nx]['type']
                if cell_type == 'empty':
                    possible_moves.append((nx, ny))
        if possible_moves:
            self.position = random.choice(possible_moves)

    def update_hunger(self):
        self.hunger -= HUNGER_DEPLETION_RATE
        if self.hunger < 0:
            self.hunger = 0

    def is_alive(self):
        return self.hunger > 0

messages = []

def display_message(text):
    messages.append(text)
    if len(messages) > 5:
        messages.pop(0)

def draw_ui(npc):
    # Draw hunger bar
    hunger_ratio = npc.hunger / MAX_HUNGER
    hunger_bar_width = int(hunger_ratio * SCREEN_WIDTH)
    hunger_bar_rect = pygame.Rect(0, GRID_HEIGHT * CELL_SIZE + 10, hunger_bar_width, 20)
    pygame.draw.rect(screen, RED, hunger_bar_rect)
    pygame.draw.rect(screen, WHITE, (0, GRID_HEIGHT * CELL_SIZE + 10, SCREEN_WIDTH, 20), 2)
    hunger_text = font.render(f"Hunger: {int(npc.hunger)}", True, WHITE)
    screen.blit(hunger_text, (10, GRID_HEIGHT * CELL_SIZE + 35))

    # Display messages
    for i, msg in enumerate(messages):
        msg_surface = font.render(msg, True, WHITE)
        screen.blit(msg_surface, (10, GRID_HEIGHT * CELL_SIZE + 60 + i * 20))

def draw_grid(grid):
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            cell = grid[y][x]
            if cell['type'] == 'tree':
                color = GREEN
            elif cell['type'] == 'rock':
                color = GRAY
            elif cell['type'] == 'berry':
                if cell['foraged']:
                    color = BROWN
                else:
                    color = RED
            else:
                color = BROWN
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, BLACK, rect, 1)  # Cell border

def draw_npc(npc):
    x, y = npc.position
    rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, BLUE, rect)

def game_over_screen(npc):
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        screen.fill(BLACK)
        game_over_text = large_font.render("NPC has starved. Game Over.", True, WHITE)
        stats_text = font.render(f"Berries Collected: {npc.berries_collected}", True, WHITE)
        survival_time = pygame.time.get_ticks() // 1000
        time_text = font.render(f"Survival Time: {survival_time} seconds", True, WHITE)
        screen.blit(game_over_text, (SCREEN_WIDTH // 2 - game_over_text.get_width() // 2,
                                     SCREEN_HEIGHT // 2 - 60))
        screen.blit(stats_text, (SCREEN_WIDTH // 2 - stats_text.get_width() // 2,
                                 SCREEN_HEIGHT // 2 - 20))
        screen.blit(time_text, (SCREEN_WIDTH // 2 - time_text.get_width() // 2,
                                SCREEN_HEIGHT // 2 + 20))
        pygame.display.flip()
        clock.tick(60)

def main():
    global screen, clock, font, large_font
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Forest Foraging NPC Game")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)
    large_font = pygame.font.SysFont(None, 48)

    grid = generate_forest()
    npc = NPC(grid)
    running = True

    while running:
        clock.tick(60)  # 60 FPS

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        npc.update_hunger()
        if not npc.is_alive():
            display_message("NPC has starved. Game Over.")
            running = False

        npc.move()

        # Drawing
        screen.fill(BLACK)
        draw_grid(grid)
        draw_npc(npc)
        draw_ui(npc)
        pygame.display.flip()

    # Game Over Screen
    game_over_screen(npc)

if __name__ == "__main__":
    main()

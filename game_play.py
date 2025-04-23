import pygame
import numpy as np

from envs import *

# Initialize pygame and the environment
pygame.init()
env = TankEnv()
env.reset()

# Set up the display
screen_size = (env.max_x * 20, env.max_y * 20)  # Scale up the game screen
screen = pygame.display.set_mode(screen_size)
clock = pygame.time.Clock()

running = True
while running:
    # Check for events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    action = 4  # Default action (stay)

    if keys[pygame.K_LEFT]:
        action = 0
    elif keys[pygame.K_DOWN]:
        action = 1
    elif keys[pygame.K_RIGHT]:
        action = 2
    elif keys[pygame.K_UP]:
        action = 3
    elif keys[pygame.K_SPACE]:
        action = 5  # Shoot

    # Update the environment
    state, reward, done, _= env.step(action)
    if done:
        env.reset()

    # Render the game state
    frame = env.render()
    frame = np.repeat(np.repeat(frame, 18, axis=0), 18, axis=1)  # Scale up the frame for visibility
    surface = pygame.surfarray.make_surface(frame)
    screen.blit(surface, (0, 0))
    pygame.display.flip()

    # Cap the frame rate
    clock.tick(9)

pygame.quit()

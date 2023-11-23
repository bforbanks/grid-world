# Grid World: AI-controlled play

# Instructions:
#   Move up, down, left, or right to move the character. The
#   objective is to find the key and get to the door
#
# Control:
#    arrows  : Merge up, down, left, or right
#    s       : Toggle slow play
#    a       : Toggle AI player
#    d       : Toggle rendering
#    r       : Restart game
#    q / ESC : Quit

from GridWorld import GridWorld
import numpy as np
import pygame
from collections import defaultdict


# Initialize the environment
env = GridWorld()
env.reset()
x, y, has_key = env.get_state()

# Definitions and default settings
actions = ["left", "right", "up", "down"]
exit_program = False
action_taken = False
slow = False
runai = True
render = False
done = False

# Game clock
clock = pygame.time.Clock()

# this is the loop that will loop through the n obervations we want to make
for i in range(200):
    consecutive_wins = 0
    attemps_used = 0
    q_table = defaultdict(lambda: [0, 0, 0, 0])

    # this is the game loop, that will continue for the same observation,
    # until 10 consecutive wins have been made
    while not exit_program:
        if render:
            env.render()

        # Slow down rendering to 5 fps
        if slow and runai:
            clock.tick(5)

        # Automatic reset environment in AI mode
        if done and runai:
            env.reset()
            x, y, has_key = env.get_state()

        # Process game events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit_program = True
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                    exit_program = True
                if event.key == pygame.K_UP:
                    best_action, action_taken = "up", True
                if event.key == pygame.K_DOWN:
                    best_action, action_taken = "down", True
                if event.key == pygame.K_RIGHT:
                    best_action, action_taken = "right", True
                if event.key == pygame.K_LEFT:
                    best_action, action_taken = "left", True
                if event.key == pygame.K_r:
                    env.reset()
                if event.key == pygame.K_d:
                    render = not render
                if event.key == pygame.K_s:
                    slow = not slow
                if event.key == pygame.K_a:
                    runai = not runai
                    clock.tick(5)

        # AI controller (enable/disable by pressing 'a')
        if runai:
            # save the current state, so we can access it after we have made the step
            old_x, old_y, old_has_key = x, y, has_key

            # get the
            tile_actions = q_table[(x, y, has_key)]

            # the best action for this tile
            best_action = np.argmax(tile_actions)

            # do the best action
            (x, y, has_key), reward, done = env.step(actions[best_action])

            # set the tile_actions q-score in the direction to the sum of:
            #   - the reward we got
            #   - the gamma-adjusted max q-score of the tile we arrived at
            tile_actions[best_action] = reward + 0.18 * max(q_table[(x, y, has_key)])
            q_table[(old_x, old_y, old_has_key)] = tile_actions

            # count the consecetive wins we have, and if 10, break the loop for this observation
            if done and env.score > 100:
                consecutive_wins += 1
            elif done:
                consecutive_wins = 0

            if consecutive_wins == 10:
                break

            if done:
                attemps_used += 1

        # Human controller
        else:
            if action_taken:
                (x, y, has_key), reward, done = env.step(best_action)
                action_taken = False

    print(attemps_used)

env.close()

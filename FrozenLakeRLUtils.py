import pygame
import numpy as np # linear algebra
import gymnasium as gym
from gym.envs.toy_text.frozen_lake import generate_random_map

INT_TO_ASCII_ARROWS = {0: '←', 1: '↓', 2: '→', 3: '↑'}

# Input: an array of action space size with corrresponding Q values
# Output: the action index with maximum Q value, choosen uniform random from all the max indices
def ChooseRandomOptimalAction(qvalues):
    epsilon = 1e-10    # precision 
    indices = (qvalues >= np.max(qvalues) - epsilon).nonzero()[0]
    return np.random.choice(indices)
    

# a -> 0: LEFT  1: DOWN  2: RIGHT  3: UP
def next_pos(row_idx, col_idx, a, MAP_DIMENSION): 
    direction = {0:[0, -1], 1:[1, 0], 2:[0, 1], 3:[-1, 0]}

    updated_row_idx = direction[a][0] + row_idx
    updated_col_idx = direction[a][1] + col_idx

    out_of_bounds = (updated_row_idx<0 or updated_row_idx>=MAP_DIMENSION) or (updated_col_idx<0 or updated_col_idx>=MAP_DIMENSION)
    return updated_row_idx, updated_col_idx, out_of_bounds
    

# Define rewards for FrozenLake
def FrozenLakeReward(h, s, a, roadmap, MAP_DIMENSION):
    row_idx, col_idx = s//MAP_DIMENSION, s%MAP_DIMENSION
    updated_row_idx, updated_col_idx, out_of_bounds = next_pos(row_idx, col_idx, a, MAP_DIMENSION)
    next_state = roadmap[updated_row_idx][updated_col_idx] if not out_of_bounds else ''
    
    bonus = (a == 1 or a == 2)*0.01
    if next_state== 'G':
        return 1
    elif next_state == 'F':
        return 0.1+bonus
    elif next_state == 'H':
        return -1
    elif out_of_bounds:
        return -1
    else:
        return 0


def print_policy(pi, H, state_size):
    for s in range(state_size):
        print(f"State {s}: ", end="")
        for h in range(H):
            print(INT_TO_ASCII_ARROWS[int(pi[h, s])], end=" ")
        print("")


def evaluate_policy(pi, env, H, episodes=10):
    cumul_reward = 0.0
    for e in range(episodes):
        s, _ = env.reset()
        done = False
        r_sum = 0
        t = 0
        while not done and t < H:
            t += 1
            a = int(pi[t-1, s])
            s, r, done, _, _ = env.step(a)
            r_sum += r
            if done or t == H:
                break
        cumul_reward += r_sum
    # print(f"Average reward: {cumul_reward / episodes}")
    return cumul_reward / episodes

def visualise_policy(pi, map, H):
    render_mode = None
    try:
        pygame.display.init()
        render_mode = "human"
    except pygame.error as e:
        if str(e) == "No available video device":
            print("No video device available. Continuing without visualisation.")
        else:
            print("Pygame error:", e)

    env_vis = gym.make('FrozenLake-v1', desc = map, is_slippery = False, render_mode = render_mode)
    
    for e in range(20):
        s, _ = env_vis.reset()
        done = False
        r_sum = 0
        t = 0
        while not done and t < H:
            t += 1
            a = int(pi[t-1, s])
            s, r, done, _, _ = env_vis.step(a)
            r_sum += r
            if done or t == H:
                print(f"Episode {e+1} finished with reward {r_sum}")
                break

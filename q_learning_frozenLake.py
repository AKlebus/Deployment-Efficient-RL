# Author: Andrea Pierré
# License: MIT License

from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

from Hyperparameters import default

CUSTOMIZED_REWARD = False

N_STEP = default["H"]


class Params(NamedTuple):
    total_episodes: int  # Total episodes
    learning_rate: float  # Learning rate
    gamma: float  # Discounting rate
    epsilon: float  # Exploration probability
    map_size: int  # Number of tiles of one side of the squared environment
    seed: int  # Define a seed so that we get reproducible results
    is_slippery: bool  # If true the player will move in intended direction with probability of 1/3 else will move in either perpendicular direction with equal probability of 1/3 in both directions
    n_runs: int  # Number of runs
    action_size: int  # Number of possible actions
    state_size: int  # Number of possible states
    proba_frozen: float  # Probability that a tile is frozen
    savefig_folder: Path  # Root folder where plots are saved


params = Params(
    total_episodes=10,
    learning_rate=0.8,
    gamma=0.95,
    epsilon=0.1,
    map_size=4,
    seed=123,
    is_slippery=False,
    n_runs=1,
    action_size=None,
    state_size=None,
    proba_frozen=0.8,
    savefig_folder=Path("../img/"),
    )
params

rng = np.random.default_rng(params.seed)


class Qlearning:
    def __init__(self, learning_rate, gamma, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.reset_qtable()

    def update(self, state, action, reward, new_state):
        """Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]"""
        delta = (
            reward
            + self.gamma * np.max(self.qtable[new_state, :])
            - self.qtable[state, action]
        )
        q_update = self.qtable[state, action] + self.learning_rate * delta
        return q_update

    def reset_qtable(self):
        """Reset the Q-table."""
        self.qtable = np.zeros((self.state_size, self.action_size))


class EpsilonGreedy:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def choose_action(self, action_space, state, qtable):
        """Choose an action `a` in the current world state (s)."""
        # First we randomize a number
        explor_exploit_tradeoff = rng.uniform(0, 1)

        # Exploration
        if explor_exploit_tradeoff < self.epsilon:
            action = action_space.sample()

        # Exploitation (taking the biggest Q-value for this state)
        else:
            # Break ties randomly
            # If all actions are the same for this state we choose a random one
            # (otherwise `np.argmax()` would always take the first one)
            if np.all(qtable[state, :]) == qtable[state, 0]:
                action = action_space.sample()
            else:
                action = np.argmax(qtable[state, :])
        return action

    

# a -> 0: LEFT  1: DOWN  2: RIGHT  3: UP
def next_pos(row_idx, col_idx, a, MAP_DIMENSION): 
    direction = {0:[0, -1], 1:[1, 0], 2:[0, 1], 3:[-1, 0]}

    updated_row_idx = direction[a][0] + row_idx
    updated_col_idx = direction[a][1] + col_idx

    out_of_bounds = (updated_row_idx<0 or updated_row_idx>=MAP_DIMENSION) or (updated_col_idx<0 or updated_col_idx>=MAP_DIMENSION)
    return updated_row_idx, updated_col_idx, out_of_bounds
    

# Define rewards for FrozenLake
def FrozenLakeReward_NewState(new_state, a, roadmap, MAP_DIMENSION):
    row_idx, col_idx = new_state//MAP_DIMENSION, new_state%MAP_DIMENSION
    out_of_bounds = (row_idx<0 or row_idx>=MAP_DIMENSION) or (col_idx<0 or col_idx>=MAP_DIMENSION)
    next_state = roadmap[row_idx][col_idx] if not out_of_bounds else ''
    
    bonus = (a == 1 or a == 2)*0.01
    if next_state== 'G':
        return 1
    #elif next_state == 'F':
    #    return 0.1+bonus
    elif next_state == 'H':
        return -1
    elif out_of_bounds:
        return -1
    else:
        return 0
    


def run_env(env, params, learner, explorer, lakemap):
    rewards = np.zeros((params.total_episodes, params.n_runs))
    steps = np.zeros((params.total_episodes, params.n_runs))
    episodes = np.arange(params.total_episodes)
    qtables = np.zeros((params.n_runs, params.state_size, params.action_size))
    all_states = []
    all_actions = []

    for run in range(params.n_runs):  # Run several times to account for stochasticity
        learner.reset_qtable()  # Reset the Q-table between runs

        for episode in tqdm(
            episodes, desc=f"Run {run}/{params.n_runs} - Episodes", leave=False
        ):
            state = env.reset(seed=params.seed)[0]  # Reset the environment
            step = 0
            done = False
            total_rewards = 0

            while not (done or step >= N_STEP):
                action = explorer.choose_action(
                    action_space=env.action_space, state=state, qtable=learner.qtable
                )

                # Log all states and actions
                all_states.append(state)
                all_actions.append(action)

                # Take the action (a) and observe the outcome state(s') and reward (r)
                
                new_state, reward, terminated, truncated, info = env.step(action)
                
                # custom reward
                custom_reward = FrozenLakeReward_NewState(new_state, action, lakemap, params.map_size)

                done = terminated or truncated

                if CUSTOMIZED_REWARD:
                    learner.qtable[state, action] = learner.update(
                        state, action, custom_reward, new_state
                )
                else:
                    learner.qtable[state, action] = learner.update(
                        state, action, reward, new_state
                )

                        

                total_rewards += reward
                step += 1

                # Our new state is state
                state = new_state

            # Log all rewards and steps
            rewards[episode, run] = total_rewards
            steps[episode, run] = step
            
            #print(f"Training Episode {episode} finished with reward {total_rewards}")
            #print("****", learner.qtable)
            
        qtables[run, :, :] = learner.qtable
        

    return rewards, steps, episodes, qtables, all_states, all_actions



def train(params, lakemap):
    # Set the seed
    #rng = np.random.default_rng(params.seed)

    #map_descs = {4: ["SFFF", "FHFH", "FFFH", "HFFG"], 2: ["SH", "FG"]}
        
    env = gym.make(
    "FrozenLake-v1",
    is_slippery=params.is_slippery,
    #render_mode="rgb_array",
    desc = lakemap,
    #desc=generate_random_map(
    #    size=params.map_size, p=params.proba_frozen, seed=params.seed
    #    ),
    #render_mode = "human",
    )
    
    params = params._replace(action_size=env.action_space.n)
    params = params._replace(state_size=env.observation_space.n)
    #print(f"Action size: {params.action_size}")
    #print(f"State size: {params.state_size}")

    learner = Qlearning(
        learning_rate=params.learning_rate,
        gamma=params.gamma,
        state_size=params.state_size,
        action_size=params.action_size,
    )
    
    explorer = EpsilonGreedy(
        epsilon=params.epsilon,
    )
    
    _, _, _, qtables, _, _ = run_env(env, params, learner, explorer, lakemap)
    
    return qtables


# Input: an array of action space size with corrresponding Q values
# Output: the action index with maximum Q value, choosen uniform random from all the max indices
def ChooseRandomOptimalAction(qvalues):
    epsilon = 1e-10    # precision 
    indices = (qvalues >= np.max(qvalues) - epsilon).nonzero()[0]
    return np.random.choice(indices)


if __name__=="__main__":
    
    # Set the seed
    #rng = np.random.default_rng(params.seed)
    
    #map_descs = {4: ["SFFF", "FHFH", "FFFH", "HFFG"], 2: ["SH", "FG"]}
    lakemap = generate_random_map(
        size=params.map_size, p=params.proba_frozen, seed=params.seed)


    # qtables : (params.n_runs, params.state_size, params.action_size)
    qtables = train(params, lakemap)
    
    # evaluation
    env_vis = gym.make('FrozenLake-v1', desc = lakemap, is_slippery = False)#, render_mode = "human")
    
    for e in range(10): # 5 episodes
        s, _ = env_vis.reset()
        done = False
        r_sum = 0
        t = 0
        while not done:
            t += 1
            a = ChooseRandomOptimalAction(qtables[0,s]) # n_runs = 1
            s, r, terminated, truncated, _ = env_vis.step(a)
            done = terminated or truncated
            r_sum += r
            if done:
                print(f"Evaluation Episode {e+1} finished with reward {r_sum}")
                break

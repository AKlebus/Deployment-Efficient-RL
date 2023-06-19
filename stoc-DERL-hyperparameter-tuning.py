import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import FrozenLakeRLUtils as utils
from StochasticDERLAgent import StochasticDERLAgent as Agent
from joblib import Parallel, delayed
from itertools import product
from collections import defaultdict
from typing import Dict, Any
import hashlib
import json

# parse the value of MAP_DIM from the --dim arg
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dim", type=int, default=4, help="Dimension of the map")
args = parser.parse_args()
MAP_DIM = args.dim


N_JOBS = 48
N_TRAIN_MAPS = 5
N_TEST_MAPS = 10

# train_map_dims = [2, 3, 4, 5, 7, 9]
# test_map_dims = [2, 3, 4, 5, 7, 9]
train_map_dims = np.full((N_TRAIN_MAPS), MAP_DIM)
test_map_dims = np.full((N_TEST_MAPS), MAP_DIM)



print("Starting")

def dict_hash(dictionary: Dict[str, Any]) -> str:
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()

def build_map(dim):
    return generate_random_map(size=dim, p=0.5)

train_maps = [build_map(map_dim) for map_dim in train_map_dims]
test_maps = [build_map(map_dim) for map_dim in test_map_dims]
train_envs = [(map, gym.make('FrozenLake-v1', desc=map, is_slippery=False)) for map in train_maps]
test_envs = [(map, gym.make('FrozenLake-v1', desc=map, is_slippery=False)) for map in test_maps]

# params for n = 5
n4_hyperparameters = {
    "epsilon": [0.9],
    "i_max": [10, 50, 100],
    # "beta": [1e-6, 1e-5, 0.0001, 0.001, 0.01, 0.1, 1] ,
    # "beta": [0.1, 1, 2] ,
    "beta": [0.1, 1, 2],
    "H": [10, 20, 40],
    "N": [100],   
    "v_min_squared": [0.01, 0.1, 1]
}


n5_hyperparameters = {
    "epsilon": [0.9],
    "i_max": [10, 50, 100],
    # "beta": [1e-6, 1e-5, 0.0001, 0.001, 0.01, 0.1, 1] ,
    "beta": [0.1, 1, 2] ,
    "H": [20, 40, 80],
    "N": [100],   
    "v_min_squared": [0.01, 0.1, 1]
}

n7_hyperparameters = {
    "epsilon": [0.9],
    "i_max": [10, 50, 100],
    # "beta": [1e-6, 1e-5, 0.0001, 0.001, 0.01, 0.1, 1] ,
    "beta": [0.1, 1, 2],
    "H": [20, 40, 80],
    "N": [100],   
    "v_min_squared": [0.01, 0.1, 1]
}

n10_hyperparameters = {
    "epsilon": [0.9],
    "i_max": [10, 50, 100],
    # "beta": [1e-6, 1e-5, 0.0001, 0.001, 0.01, 0.1, 1] ,
    "beta": [0.1, 1, 2],
    "H": [80, 120, 160],
    "N": [100],   
    "v_min_squared": [0.01, 0.1, 1]
}

n0_hyperparameters = {
    'delta': [0.2],
    'epsilon': [0.9],
    'c_beta': [1e-5],
    'H': [10],
    'K': [10],
    'N': [1],
    'explore_prob': [0.6, 0.8, 1]
}

# map hyperparameters to use depending on MAP_DIM
hyperparameters_selector = {
    4: n4_hyperparameters,
    5: n5_hyperparameters,
    7: n7_hyperparameters,
    10: n10_hyperparameters
}

hyperparameters = hyperparameters_selector[MAP_DIM]


def evaluate_configuration(env, map_desc, epsilon, i_max, beta, H, N, v_min_squared):

    print(f"Training for map {map_desc} with params {epsilon, i_max, beta, H, N, v_min_squared}", flush=True)
    # Initialize the agent with the current hyperparameters
    params = {'epsilon': epsilon, 'i_max': i_max, 'beta': beta, 'H': H, 'N': N, 'v_min_squared': v_min_squared}
    # params = {'delta': delta, 'epsilon': epsilon, 'c_beta': c_beta, 'H': H, 'K': K, 'N': N, 'explore_prob': explore_prob}
    agent = Agent(env, utils.FrozenLakeReward, roadmap=map_desc, params=params)

    # Train the agent
    agent.train()
    pi = agent.pi

    # Evaluate the agent's policy
    reward = utils.evaluate_policy(pi, env, H, episodes=1)

    # return reward, pi, params
    print(f"Reward: {reward}; Params: {params}", flush=True)

    return reward, pi, params

grid_search_params = list(product(
    hyperparameters['epsilon'],
    hyperparameters['i_max'],
    hyperparameters['beta'],
    hyperparameters['H'],
    hyperparameters['N'],
    hyperparameters['v_min_squared']
))

# for map_desc, env in train_envs:
#     for epsilon, i_max, beta, H, N, v_min_squared in grid_search_params:
#         print(f"Training for map {map_desc} with params {epsilon, i_max, beta, H, N, v_min_squared}")
#         evaluate_configuration(env, map_desc, epsilon, i_max, beta, H, N, v_min_squared)

results = Parallel(n_jobs=N_JOBS)(
    delayed(evaluate_configuration)(env, map_desc, epsilon, i_max, beta, H, N, v_min_squared)
    for epsilon, i_max, beta, H, N, v_min_squared in grid_search_params
    for map_desc, env in train_envs
)

cumul_score_per_policy = defaultdict(lambda: 0)
hash_to_params = {}
params_to_pi = {}

for reward, pi, params in results:
    hashed_params = dict_hash(params)
    hash_to_params[hashed_params] = params
    cumul_score_per_policy[hashed_params] += reward
    params_to_pi[hashed_params] = pi

# Create a list of dictionaries representing each row in the DataFrame
data = []
for hashed_params, reward in cumul_score_per_policy.items():
    row = hash_to_params[hashed_params]
    row['Reward'] = reward
    row['AvgReward'] = reward / N_TRAIN_MAPS
    data.append(row)

# Create the DataFrame
df = pd.DataFrame(data)
# sort by highest avg reward
df = df.sort_values(by='Reward', ascending=False)

print(df)

output_file = f"output_stoc_{MAP_DIM}.csv"  
df.to_csv(output_file, index=False)

    
best_pi = None
best_reward = float('-inf')
best_params = None

for hashed_params, reward in cumul_score_per_policy.items():
    if reward > best_reward:
        best_reward = reward
        best_params = hash_to_params[hashed_params]
        best_pi = params_to_pi[hashed_params]

print("", flush=True)
print("Best policy:")
print(best_pi)
print("Best reward:", best_reward)
print("Best parameters:", best_params)

def test(hyperparameters):
    delta = hyperparameters['delta']
    epsilon = hyperparameters['epsilon']
    c_beta = hyperparameters['c_beta']
    H = hyperparameters['H']
    K = hyperparameters['K']
    N = hyperparameters['N']
    explore_prob = hyperparameters['explore_prob']

    results = Parallel(n_jobs=N_JOBS)(
        delayed(evaluate_configuration)(env, map_desc, delta, epsilon, c_beta, H, K, N, explore_prob)
        for map_desc, env in test_envs
    )

    rewards = (reward for reward, pi, params in results)
    avg_reward = np.fromiter(rewards, dtype=float).mean()
    print("Overall average reward:", avg_reward)

# test(best_params)

# utils.visualise_policy(best_pi, map, best_params['H'])

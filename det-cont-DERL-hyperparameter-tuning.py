import pandas as pd
import numpy as np
import gymnasium as gym
from ContinuousDiscreteDERLAgent import ContinuousDiscreteDERLAgent as Agent
from joblib import Parallel, delayed
from itertools import product
from collections import defaultdict
from typing import Dict, Any
import hashlib
import json
from utils.CartPole import CartPoleEnv

DEBUG = False
N_JOBS = 48
NUM_TEST_EPISODES = 10


print("Starting")

def dict_hash(dictionary: Dict[str, Any]) -> str:
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()

explore_prob = [p for p in np.arange(0, 1.2, 0.2)]
explore_prob.append(-1)

hyperparameters = {
    'delta': [0.2],
    'epsilon': [0.9],
    'c_beta': [1e-4, 1e-5, 1e-6],
    'H': [200],
    'K': [30, 50, 100],
    'N': [100, 250, 500, 750],
    'explore_prob': explore_prob
}

# hyperparameters = {
#     'delta': [0.2],
#     'epsilon': [0.9],
#     'c_beta': [1e-5],
#     'H': [3],
#     'K': [3],
#     'N': [3],
#     'explore_prob': [0.6]
# }

def evaluate_policy(agent, env, H):
    r_total_pi = 0
    for e in range(NUM_TEST_EPISODES): 
        s = env.reset()
        done = False
        r_sum = 0
        t = 0
        while not done and t < H:
            t += 1
            a = agent.act(t, s)
            print(a)  if DEBUG else None
            s, r, done, _ = env.step(a)
            r_sum += r
            if done or t == H:
                print(f"Episode {e+1} finished with reward {r_sum}") if DEBUG else None
                r_total_pi += r_sum
                if done:
                    print(f"Episode finished after {t} timesteps") if DEBUG else None
                break
    print(f"Average reward: {r_total_pi / NUM_TEST_EPISODES}") if DEBUG else None
    return r_total_pi / NUM_TEST_EPISODES

    # cumul_reward = 0.0
    # for e in range(episodes):
    #     s, _ = env.reset()
    #     done = False
    #     r_sum = 0
    #     t = 0
    #     while not done and t < H:
    #         t += 1
    #         a = int(pi[t-1, s])
    #         s, r, done, _, _ = env.step(a)
    #         r_sum += r
    #         if done or t == H:
    #             break
    #     cumul_reward += r_sum
    # # print(f"Average reward: {cumul_reward / episodes}")
    # return cumul_reward / episodes

def evaluate_configuration(delta, epsilon, c_beta, H, K, N, explore_prob):
    env = CartPoleEnv(max_ep_len=H, seed = 0, rew_shift=0, append=True) # Modified env for training
    params = {'delta': delta, 'epsilon': epsilon, 'c_beta': c_beta, 'H': H, 'K': K, 'N': N, 'explore_prob': explore_prob}
    agent = Agent(env = env, reward = env.reward)

    # Train the agent
    agent.train()

    # Evaluate the agent's policy
    reward = evaluate_policy(agent, env, H)

    # return reward, pi, params
    print(f"Reward: {reward}; Params: {params}", flush=True)

    return reward, agent, params

grid_search_params = list(product(
    hyperparameters['delta'],
    hyperparameters['epsilon'],
    hyperparameters['c_beta'],
    hyperparameters['H'],
    hyperparameters['K'],
    hyperparameters['N'],
    hyperparameters['explore_prob'],
))


# for delta, epsilon, c_beta, H, K, N, explore_prob in grid_search_params:
#     evaluate_configuration(delta, epsilon, c_beta, H, K, N, explore_prob)

results = Parallel(n_jobs=N_JOBS)(
    delayed(evaluate_configuration)(delta, epsilon, c_beta, H, K, N, explore_prob)
    for delta, epsilon, c_beta, H, K, N, explore_prob in grid_search_params
)

cumul_score_per_policy = defaultdict(lambda: 0)
hash_to_params = {}
# params_to_pi = {}

for reward, agent, params in results:
    hashed_params = dict_hash(params)
    hash_to_params[hashed_params] = params
    cumul_score_per_policy[hashed_params] += reward
    # params_to_pi[hashed_params] = agent

# Create a list of dictionaries representing each row in the DataFrame
data = []
for hashed_params, reward in cumul_score_per_policy.items():
    row = hash_to_params[hashed_params]
    row['Reward'] = reward
    row['AvgReward'] = reward 
    data.append(row)

# Create the DataFrame
df = pd.DataFrame(data)
# sort by highest avg reward
df = df.sort_values(by='Reward', ascending=False)

print(df)

output_file = f"output_algo1_cont.csv"  
df.to_csv(output_file, index=False)

    
# best_pi = None
best_reward = float('-inf')
best_params = None

for hashed_params, reward in cumul_score_per_policy.items():
    if reward > best_reward:
        best_reward = reward
        best_params = hash_to_params[hashed_params]
        # best_pi = params_to_pi[hashed_params]

print("", flush=True)
# print("Best policy:")
# print(best_pi)
print("Best reward:", best_reward)
print("Best parameters:", best_params)

# def test(hyperparameters):
#     delta = hyperparameters['delta']
#     epsilon = hyperparameters['epsilon']
#     c_beta = hyperparameters['c_beta']
#     H = hyperparameters['H']
#     K = hyperparameters['K']
#     N = hyperparameters['N']
#     explore_prob = hyperparameters['explore_prob']

#     results = Parallel(n_jobs=N_JOBS)(
#         delayed(evaluate_configuration)(env, map_desc, delta, epsilon, c_beta, H, K, N, explore_prob)
#     )

#     rewards = (reward for reward, pi, params in results)
#     avg_reward = np.fromiter(rewards, dtype=float).mean()
#     print("Overall average reward:", avg_reward)

# # test(best_params)

# # utils.visualise_policy(best_pi, map, best_params['H'])

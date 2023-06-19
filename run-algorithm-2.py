import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gymnasium as gym # for environment
from gym.envs.toy_text.frozen_lake import generate_random_map
import FrozenLakeRLUtils as utils
from StochasticDERLAgent import StochasticDERLAgent as Agent

# Changes the size of the frozen lake map
MAP_DIMENSION = 4

# if __name__=="__main__":
#     map_descs = {4: ["SFFF", "FHFH", "FFFH", "HFFG"], 2: ["SH", "FG"]}
#     map_descs.setdefault(MAP_DIMENSION, generate_random_map(size=MAP_DIMENSION, p=0.5))

#     map = map_descs[MAP_DIMENSION]
#     env = gym.make('FrozenLake-v1', desc = map, is_slippery = True)
#     agent = Agent(env = env, reward = utils.FrozenLakeReward, roadmap=map)  

#     pi = agent.train() # Train agent

#     utils.print_policy(pi, agent.H, agent.state_size) # Print policy
    
#     env_vis = gym.make('FrozenLake-v1', desc = map, is_slippery = False, render_mode = "human")
    
#     for e in range(10):
#         s, _ = env_vis.reset()
#         done = False
#         r_sum = 0
#         t = 0
#         while not done and t < agent.H:
#             t += 1
#             a = int(pi[t-1, s])
#             s, r, done, _, _ = env_vis.step(a)
#             r_sum += r
#             if done or t == agent.H:
#                 print(f"Episode {e+1} finished with reward {r_sum}")
#                 break

if __name__=="__main__":
    map = generate_random_map(size=MAP_DIMENSION, p=0.5)
    env = gym.make('FrozenLake-v1', desc = map, is_slippery = False)
    agent = Agent(env = env, reward = utils.FrozenLakeReward, roadmap=map)  

    agent.train() # Train agent

    utils.print_policy(agent.pi, agent.H, agent.state_size) # Print policy
    
    
    # See if just layer 1 policy works well enough
    #env_vis = gym.make('FrozenLake-v1', desc = map_descs[MAP_DIMENSION], is_slippery = False, render_mode = "human")
    env_vis = gym.make('FrozenLake-v1', desc = map, is_slippery = False, render_mode = "human")
    
    for e in range(10): # 10 episodes
        s, _ = env_vis.reset()
        done = False
        r_sum = 0
        t = 0
        while not done and t < agent.H:
            t += 1
            a = int(agent.pi[t-1, s])
            s, r, done, _, _ = env_vis.step(a)
            r_sum += r
            if done or t == agent.H:
                print(f"Episode {e+1} finished with reward {r_sum}")
                break
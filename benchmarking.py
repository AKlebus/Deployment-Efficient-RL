import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gymnasium as gym # for environment
from pathlib import Path
from gym.envs.toy_text.frozen_lake import generate_random_map
from q_learning_frozenLake import Params, train, ChooseRandomOptimalAction
from DeterministicDiscreteDERLAgent import DeterministicDiscreteDERLAgent
from FrozenLakeRLUtils import next_pos
from vis_benchmarking import plot_benchmarking, plot_mapsize_comp
from Hyperparameters import default

MAP_DIMENSION = default["map_dimension"]

N_Runs = 100

N_Episodes = 20

ProbabilityFrozen = [0.5, 0.6, 0.7, 0.8, 0.9]

Map_Sizes = [4,5,7,10]

tuned_params = {
    4 : {
    "map_dimension": 4   , 
    "delta":         0.2 ,# failure probability
    "epsilon":       0.9 ,# target accuracy
    "c_beta":        1e-4,# scales beta, but should be > 0
    "H":             20  ,# time horizon
    "K":             25  ,# number of deployments
    "N":             1   ,# batch size
    "explore_prob":  0.1  # explore probability
    },
    
    5: {
    "map_dimension": 5   , 
    "delta":         0.2 ,# failure probability
    "epsilon":       0.9 ,# target accuracy
    "c_beta":        1e-6,# scales beta, but should be > 0
    "H":             30  ,# time horizon
    "K":             30  ,# number of deployments
    "N":             1   ,# batch size
    "explore_prob":  0.5  # explore probability
    },
    
    7: {
    "map_dimension": 7   , 
    "delta":         0.2 ,# failure probability
    "epsilon":       0.9 ,# target accuracy
    "c_beta":        1e-6,# scales beta, but should be > 0
    "H":             30  ,# time horizon
    "K":             30  ,# number of deployments
    "N":             1   ,# batch size
    "explore_prob":  0.2  # explore probability
    }, 
    
    10: {
    "map_dimension": 5   , 
    "delta":         0.2 ,# failure probability
    "epsilon":       0.9 ,# target accuracy
    "c_beta":        1e-6,# scales beta, but should be > 0
    "H":             50  ,# time horizon
    "K":             50  ,# number of deployments
    "N":             1   ,# batch size
    "explore_prob":  0.1  # explore probability
    }
}


# Script to compare algorithm1 and q-learning

# We need to use the same reward function for Q-Learning and Algorithm 1
# Define rewards for FrozenLake
def FrozenLakeReward(h, s, a, roadmap, MAP_DIMENSION):
    row_idx, col_idx = s//MAP_DIMENSION, s%MAP_DIMENSION
    updated_row_idx, updated_col_idx, out_of_bounds = next_pos(row_idx, col_idx, a, MAP_DIMENSION)
    next_state = roadmap[updated_row_idx][updated_col_idx] if not out_of_bounds else ''
    
    bonus = (a == 1 or a == 2)*0.01
    if next_state== 'G':
        return 1
    elif next_state == 'F':
        return 0.1 + bonus
    elif next_state == 'H':
        return -1
    elif out_of_bounds:
        return -1
    else:
        return 0


def evaulate_algorithm1(env_vis, agent, pi_algo1):
    success_algorithm1 = 0
    #print(f"Evaluating 1 Episodes on Algorithm 1: ")
    for e in range(1): # determinstic
        s, _ = env_vis.reset()
        done = False
        r_sum = 0
        t = 0
        while not done and t < agent.H:
            t += 1
            a = int(pi_algo1[t-1, s])
            s, r, done, _, _ = env_vis.step(a)
            r_sum += r
            if done or t == agent.H:
                #print(f"Episode {e+1} finished with reward {r_sum}")
                success_algorithm1 += r_sum
                break
            
    #print(f"Algorithm 1 success rate: {success_algorithm1}/{1}")
    return success_algorithm1


def benchmarking(probability_frozen):
    
    # parameter for q learning
    params = Params(
    total_episodes=10,
    learning_rate=0.8,
    gamma=0.95,
    epsilon=0.1,
    map_size=MAP_DIMENSION,
    seed=123,
    is_slippery=False,
    n_runs=1,
    action_size=None,
    state_size=None,
    proba_frozen=probability_frozen,
    savefig_folder=Path("./img/"),
    )
    params
        
    lakemap = generate_random_map(size=MAP_DIMENSION, p=probability_frozen)  
    
    
    # Q learning
    qtables = train(params, lakemap)

    # Algorithm 1
    env = gym.make('FrozenLake-v1', desc = lakemap, is_slippery = False)
    agent = DeterministicDiscreteDERLAgent(env = env, reward = FrozenLakeReward, roadmap=lakemap)  
    pi_algo1 = agent.train() # Train agent
    
    # evaluation
    env_vis = gym.make('FrozenLake-v1', desc = lakemap, is_slippery = False)#, render_mode = "human")

    success_q_learning = 0
    
    print(f"Evaluating {N_Episodes} Episodes on Q-Learning: ")
    for e in range(N_Episodes): #  episodes
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
                    #print(f"Evaluation Episode {e+1} finished with reward {r_sum}")
                    success_q_learning += r_sum
                    break
    print(f"Q-Learning success rate: {success_q_learning}/{N_Episodes}")

    success_algorithm1 = evaulate_algorithm1(env_vis, agent, pi_algo1)
    
    """ success_algorithm1 = 0
    print(f"Evaluating 1 Episodes on Algorithm 1: ")
    for e in range(1): # determinstic
        s, _ = env_vis.reset()
        done = False
        r_sum = 0
        t = 0
        while not done and t < agent.H:
            t += 1
            a = int(pi_algo1[t-1, s])
            s, r, done, _, _ = env_vis.step(a)
            r_sum += r
            if done or t == agent.H:
                #print(f"Episode {e+1} finished with reward {r_sum}")
                success_algorithm1 += r_sum
                break
            
    print(f"Algorithm 1 success rate: {success_algorithm1}/{1}") """
    
    return success_q_learning, success_algorithm1


def run_benchmarking():
    print(f"For map size {MAP_DIMENSION}: ")
    
    sum_success_q_learning = []
    sum_success_algorithm1 = []
    
    for prob in ProbabilityFrozen:
        print(f"For Frozen probability {prob}: ")
        
        success_q_learning, success_algorithm1 = 0, 0
        
        for _ in range(N_Runs):
            single_q_learning, single_alorithm1 = benchmarking(prob)
            success_q_learning += single_q_learning
            success_algorithm1 += single_alorithm1
        
        sum_success_q_learning.append(success_q_learning/(N_Episodes*N_Runs))
        sum_success_algorithm1.append(success_algorithm1/(1*N_Runs))
    
    print("Frozen Probability: ", ProbabilityFrozen)
    print(f"Evaulated on {N_Episodes} Episodes ")
    print("Q Learning Success Rate: ", sum_success_q_learning)    
    print("Algorithm 1 Success Rate: ", sum_success_algorithm1)    

    # visualization
    d = {"Frozen Probability": ProbabilityFrozen*2, "Success Rate": sum_success_q_learning + sum_success_algorithm1, "Algorithm": ["Q-Learning"]*5 + ["Algorithm 1"]*5}
    data = pd.DataFrame(data = d)
    plot_benchmarking(data)



def map_size_success_rate(map_sizes):
    
    success_rate = []
    
    for size in map_sizes:
        print(f"Map size: {size}")
        for prob in ProbabilityFrozen:
            print(f"  Frozen Probability: {prob}")
            sum_success = 0
            for _ in range(N_Runs):
                lakemap = generate_random_map(size=size, p=prob)  

                env = gym.make('FrozenLake-v1', desc = lakemap, is_slippery = False)
                agent = DeterministicDiscreteDERLAgent(env = env, reward = FrozenLakeReward, roadmap=lakemap, params=tuned_params[size])  
                pi_algo1 = agent.train() # Train agent
            
                # evaluation
                success = evaulate_algorithm1(env, agent, pi_algo1)
                sum_success += success
            
            print(f"    Success rate: {sum_success/N_Runs}")
            success_rate.append(sum_success/N_Runs)
            
    return success_rate


def run_map_size_success_rate():
    success_rate = map_size_success_rate(Map_Sizes)
    print("Map sizes: ", Map_Sizes)
    print("Frozen Probability: ", ProbabilityFrozen)
    print("Success Rate:")
    print(success_rate)
    plot_mapsize_comp(success_rate, Map_Sizes)
    
    

if __name__=="__main__":
    
    #run_benchmarking()
    #run_map_size_success_rate()
    
    success_rate = [1.0, 0.96, 0.94, 0.96, 0.96, 0.82, 0.72, 0.8, 0.88, 0.94, 0.7, 0.6, 0.68, 0.74, 0.9, 0.64, 0.4, 0.52, 0.78, 0.96]
    plot_mapsize_comp(success_rate, Map_Sizes)
    
    plot_mapsize_comp(success_rate, Map_Sizes)

    
    
    """ 
    success_rate:
    [1.0, 0.96, 0.94, 0.96, 0.96, 0.82, 0.72, 0.8, 0.88, 0.94, 0.7, 0.6, 0.68, 0.74, 0.9, 0.64, 0.4, 0.52, 0.78, 0.96]

    """
    
        

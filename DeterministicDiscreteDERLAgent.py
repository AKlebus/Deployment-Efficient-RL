from math import sqrt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gymnasium as gym # for environment
from gym.envs.toy_text.frozen_lake import generate_random_map
from collections import deque
import random
import FrozenLakeRLUtils as utils
import Hyperparameters as hp

DEBUG = False

class DeterministicDiscreteDERLAgent: # Deterministic Deployment Efficient Reinforcement Learning Agent (Algorithm 1 from paper)
    def __init__(self, env, reward, roadmap, params = hp.default):
        self.env = env
        self.reward = reward
        self.roadmap = roadmap

        # env dependent parameters
        # self.state_size = env.observation_space.shape[0] # this is for input of neural network node size
        # self.action_size = env.action_space.n # this is for out of neural network node size
        # We hardcode first to get it working then figure out how to generalize
        self.state_size = env.observation_space.n
        self.map_dimension = int(sqrt(self.state_size)) 
        # self.map_dimension**2  # width * height of map
        # self.map_dimension = map_dimension if map_dimension else params.map_dimension
        self.action_size = 4
        self.d = self.state_size * self.action_size # dimension of the (onehot) feature vector

        # hyperparameters
        self.delta = params['delta'] # confidence parameter
        self.epsilon = params['epsilon'] # target accuracy
        self.c_beta = params['c_beta'] # bonus coefficient
        # self.H = int(params['H'] * self.state_size) # time horizon
        self.H = params['H']  # time horizon
        self.K = params['K'] # number of deployments
        self.N = params['N'] # batch size
        self.explore_prob = params.get('explore_prob') if params.get('explore_prob') is not None else 0.6

        self.beta = self.c_beta * self.d * self.H * np.sqrt(np.log(self.d * self.H / (self.delta*self.epsilon ))) # hyperparameter, unclear exactly what it does
             
        # Q, V, etc.
        self.Q = np.zeros((self.K, self.H, self.state_size, self.action_size)) # Q function. First index is deployment, second is layer, third is state, fourth is action
        self.V = np.zeros((self.K, self.H+1, self.state_size)) # V function. First index is deployment, second is layer, third is state
        self.pi = np.zeros((self.H, self.state_size)) # policy. First index is layer, second is state. Value is action taken (policies are deterministic).

        self.lambda_inv = np.zeros((self.H, self.d, self.d)) # inverse of lambda. First index is layer, second and third are indices of the matrix

        self.h = 1 # current layer
        
        # data buffers
        self.state_buffer = np.zeros((self.K, self.N, self.H+1)) # (Num deployments, batch size, time horizon+1). Extra time horizon since we have one extra state at the end
        self.action_buffer = np.zeros((self.K, self.N, self.H)) # (Num deployments, batch size, time horizon)
        self.feature_buffer = np.zeros((self.K, self.N, self.H, self.d)) # (Num deployments, batch size, time horizon, feature vector dimension)
    

    def _get_features(self, state, action):
        # returns the one-hot feature vector for the given state and action
        features = np.zeros(self.d)
        features[state*self.action_size + action] = 1
        return features

    
    def policy_update_step(self, k): #update policy for deployment k
        # Q, V already initialized to 0 properly, so no need to do this here
        for h in range(self.h, 0, -1):
            
            # Compute lambda
            lambda_k = np.eye(self.d)
            for tau in range(1, k):
                for n in range(1, self.N+1):
                    phi = self.feature_buffer[tau-1, n-1, h-1]
                    lambda_k += np.outer(phi, phi)
            #print(lambda_k)
            self.lambda_inv[h-1] = np.linalg.inv(lambda_k) # Store inverse for later use
            
            # Compute u
            u_k = np.zeros((self.state_size, self.action_size))
            for s in range(0, self.state_size):
                for a in range(0, self.action_size):
                    phi = self._get_features(s, a)
                    u_k[s, a] = np.minimum(self.beta * np.sqrt(np.linalg.multi_dot((phi, self.lambda_inv[h-1], phi))), self.H)
            
            # Compute w
            w = np.zeros(self.d)
            for tau in range(1, k):
                for n in range(1, self.N+1):
                    phi = self.feature_buffer[tau-1, n-1, h-1]
                    w += phi * self.V[k-1, h, int(self.state_buffer[tau-1, n-1, h])] # god this indexing......
            w = np.dot(self.lambda_inv[h-1], w)

            # Compute Q
            for s in range(self.state_size):
                for a in range(self.action_size):
                    phi = self._get_features(s, a)
                    self.Q[k-1, h-1, s, a] = np.minimum(np.dot(phi, w) + u_k[s, a] + self.reward(h-1, s, a, self.roadmap, self.map_dimension), self.H) #TODO: This seemed to always be self.H, inhibiting meaningful policy choices. Check why this is
                    self.Q[k-1, h-1, s, a] = np.dot(phi, w) + u_k[s, a] + self.reward(h-1, s, a, self.roadmap, self.map_dimension)
                #print(f"# Q[{k-1}, {h-1}, {s}, :]: {self.Q[k-1, h-1, s, :]}")
                
            # Compute V
            for s in range(self.state_size):
                self.V[k-1, h-1, s] = np.max(self.Q[k-1, h-1, s, :])

            # Compute pi
            for s in range(self.state_size):
                if np.max(self.Q[k-1, h-1, s, :]) == np.min(self.Q[k-1, h-1, s, :]):
                    self.pi[h-1, s] = self.env.action_space.sample()
                else:
                    #self.pi[h-1, s] = np.argmax(self.Q[k-1, h-1, s, :])
                    # Use ChooseRandomOptimalAction because argmax always return the first element which is maximum, 
                    # which is always in favor of action of lower index (0: LEFT  1: DOWN  2: RIGHT  3: UP)
                    self.pi[h-1, s] = utils.ChooseRandomOptimalAction(self.Q[k-1, h-1, s, :])

    
    def act(self, h, state):
        # If h not explored, sample from uniform distribution
        if self.h < h:
            return self.env.action_space.sample()
        else:
            return int(self.pi[h-1, state])

    def continue_step(self, k):
        # Decide what to do next iteration. True means we are done, False means we continue
        delta_k = 0
        for n in range(1, self.N+1):
            for h in range(1, self.h+1):
                phi = self.feature_buffer[k-1, n-1, h-1]
                #delta_k += np.sqrt(np.dot(phi, np.dot(self.lambda_inv[h-1], phi))) #TODO: check if matrix multiplication is correct
                delta_k += np.sqrt(np.linalg.multi_dot((phi, self.lambda_inv[h-1], phi)))

        """
        print(f'********************************')
        print(f"delta_k: {delta_k} ")
        print(f"self.beta: {self.beta} ")
        print(f"delta_k * 4 * self.beta * self.H: {delta_k * 4 * self.beta * self.H} , self.N * self.epsilon * self.h: {self.N * self.epsilon * self.h}")
        print(f'********************************')
        """
        
        #TODO: This seems to always hold, so we never explore futher layers. Check why this is
        if self.explore_prob >= 0:
            explore_cond = random.random() < self.explore_prob
        else:
            explore_cond = delta_k * 4 * self.beta * self.H >= self.N * self.epsilon * self.h

        if explore_cond:  # This is temporary, to force exploration
        # if delta_k * 4 * self.beta * self.H >= self.N * self.epsilon * self.h: #Here we multiply out to avoid division! 
            return False
        elif self.h == self.H:
            return True
        else:
            self.h += 1
            return False
    
    def collect_trajectories_step(self, k):
        # Collect trajectories for deployment k
        for n in range(1, self.N+1):
            s, _ = self.env.reset()
            self.state_buffer[k-1, n-1, 0] = s
            for h in range(1, self.H+1):
                a = self.act(h, s)
                s, _, _, _, _ = self.env.step(a)
                self.state_buffer[k-1, n-1, h] = s
                self.action_buffer[k-1, n-1, h-1] = a
                self.feature_buffer[k-1, n-1, h-1] = self._get_features(s, a)


    def train(self):
        # Full training process. Returns the optimal policy
        for k in range(1, self.K+1):
            # Policy update 
            self.policy_update_step(k)

            # Collect trajectories 
            self.collect_trajectories_step(k)

            # Continue or notS
            if self.continue_step(k):
                break

            print(f"Iteration {k} done") if DEBUG else None
            print(f"Current layer: {self.h}") if DEBUG else None
        
        return self.pi
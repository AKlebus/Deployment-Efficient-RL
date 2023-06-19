import numpy as np # linear algebra
import gymnasium as gym # for environment
from gym.envs.toy_text.frozen_lake import generate_random_map
import random

# Changes the size of the frozen lake map
MAP_DIMENSION = 4
INT_TO_ASCII_ARROWS = {0: '←', 1: '↓', 2: '→', 3: '↑'}

TIME_HORIZON = 10 # time horizon == number of deployments

class SDERLAgent: # Stochastic Deployment Efficient Reinforcement Learning Agent (Algorithm 2 from paper)
    def __init__(self, env, reward, roadmap):
        self.env = env
        self.reward = reward
        self.roadmap = roadmap

        # env dependent parameters
        # Hardcoded for now
        self.state_size = MAP_DIMENSION**2 # width * height of map
        self.action_size = 4
        self.d = self.state_size * self.action_size # dimension of the (onehot) feature vector

        self.H = int(TIME_HORIZON) # time horizon

        # hyperparameters
        self.epsilon = 0.9 # target accuracy
        self.i_max = 10 # number of iterations
        self.beta = 1 # bonus coefficient
        self.N = 100 # batch size
        self.epsilon_0 = 1 / (2*self.d *(self.N + 1)) # resolution for discretization

        self.v_min_squared = 1 # magic coeficient. Use 1 for deterministic, 0.1 / 0.01 for stochastic according to Jiawei

        # Q, V, etc.
        self.Sigma = np.zeros((self.H, self.d, self.d)) # covariance matrix. First index is layer, second and third are indices of the matrix
        self.Sigma_inv = np.zeros((self.H, self.d, self.d)) # inverse covariance matrix, precompute for efficiency
        self.Sigma_tilde = np.zeros((self.d, self.d)) # temporary covariance matrix.
        self.PI = [] # Mixture of policies buffer 
        self.D_state = np.zeros((self.H, self.N, self.H+1), dtype=np.int32) # (Deployment, Batch, Time) Collected state data buffer
        self.D_action = np.zeros((self.H, self.N, self.H), dtype=np.int32) # (Deployment, Batch, Time) Collected action data buffer
        self.pi = np.zeros((self.H, self.state_size)) # final learned policy


    def _get_features(self, state, action):
        # returns the one-hot feature vector for the given state and action
        features = np.zeros(self.d)
        features[int(state*self.action_size + action)] = 1
        return features

    
    def estimate_covariance_matrix(self, h, pi): 
        # Estimate cov matrix. For onehot, we can just do the diagonal I think
        cov_mat = np.zeros((self.d, self.d))
        for i in range(self.d):
            #aux_R = np.ones((self.state_size, self.action_size)) # \phi(s, a)_i * \phi(s, a)_j = (i==j)
            Q = np.ones((self.state_size, self.action_size)) # Q(s, a) = 1
            V = np.ones((self.state_size)) # V(s) = 1 by definition of Q
            for h_bar in range(h-1, 0, -1):

                # Compute w
                w = np.ones((self.d)) #TODO: Implement
                for n in range(1, self.N+1):
                    phi = self._get_features(self.D_state[h_bar-1, n-1, h_bar-1], self.D_action[h_bar-1, n-1, h_bar-1]) #TODO: Check this
                    w += phi * V[self.D_state[h_bar-1, n-1, h_bar]] #TODO: Check this
                w = np.dot(self.Sigma_inv[h_bar-1], w)

                # Compute Q
                for s in range(self.state_size):
                    for a in range(self.action_size):
                        phi = self._get_features(s, a)
                        Q[s, a] = np.minimum(np.dot(w, phi) , 1)

                # Compute V based on Q and pi
                for s in range(self.state_size):
                    V[s] = Q[s, int(pi[h_bar-1, s])]

            cov_mat[i, i] = V[0] # Value at first state
        
        return 2*cov_mat - np.ones((self.d, self.d))


    def solve_opt_Q(self, h):
        # Set up initial values
        Z = np.linalg.inv(2*np.eye(self.d) + self.Sigma_tilde)
        rew = np.zeros((self.state_size, self.action_size))
        for s in range(self.state_size):
            for a in range(self.action_size):
                phi = self._get_features(s, a)
                rew[s, a] = np.sqrt(np.linalg.multi_dot((phi, Z, phi)))
        
        Q = np.zeros((self.state_size, self.action_size))
        V = np.zeros((self.state_size))

        Q[:] = rew[:]
        V[:] = np.max(Q, axis=1)

        pi = np.zeros((self.H, self.state_size))
        pi[h-1, :] = np.argmax(Q, axis=1)

        for h_bar in range(h-1, 0, -1):
            # Compute w
            w = np.zeros((self.d))
            for n in range(1, self.N+1):
                phi = self._get_features(self.D_state[h_bar-1, n-1, h_bar-1], self.D_action[h_bar-1, n-1, h_bar-1])
                w += phi * V[self.D_state[h_bar-1, n-1, h_bar]]
            w = np.dot(self.Sigma_inv[h_bar - 1], w)
        
            # Compute u
            u = np.zeros((self.state_size, self.action_size))
            for s in range(self.state_size):
                for a in range(self.action_size):
                    phi = self._get_features(s, a)
                    u[s, a] = self.beta * np.sqrt(np.linalg.multi_dot((phi, self.Sigma_inv[h_bar-1], phi)))
            
            # Compute Q
            for s in range(self.state_size):
                for a in range(self.action_size):
                    phi = self._get_features(s, a)
                    Q[s, a] = np.dot(w, phi) + u[s, a] # paper says it's min between this and 1, but leaving as is for now
            
            # Compute V based on Q
            V[:] = np.max(Q, axis=1)

            # Compute pi
            pi[h_bar-1, :] = np.argmax(Q, axis=1)
        
        return V[0], pi
        

    def plan_pi(self):
        # After exploration, we can plan the optimal policy

        # Define V, Q, etc.
        V = np.zeros((self.H+1, self.state_size))

        Lambda = np.eye(self.d)

        for h in range(self.H, 0, -1):
            # Compute lambda
            
            for tau in range(1, int(self.H+1)):
                for n in range(1, self.N+1):
                    phi = self._get_features(self.D_state[tau-1, n-1, h-1], self.D_action[tau-1, n-1, h-1]) #TODO: Check this
                    Lambda += np.outer(phi, phi)
            
            Lambda_inv = np.linalg.inv(Lambda)

            # Compute u
            u = np.zeros((self.state_size, self.action_size))
            for s in range(self.state_size):
                for a in range(self.action_size):
                    phi = self._get_features(s, a)
                    u[s, a] = np.minimum(self.beta*np.sqrt(np.linalg.multi_dot((phi, Lambda_inv, phi))) , self.H) #TODO: Check
            
            # Compute w
            w = np.zeros((self.d))
            for tau in range(1, int(self.H+1)):
                phi = self._get_features(self.D_state[tau-1, n-1, h-1], self.D_action[tau-1, n-1, h-1]) #TODO: fix this
                w += phi * V[h, self.D_state[tau-1, n-1, h]] #TODO: fix this
            w = np.dot(Lambda_inv, w)

            # Compute Q
            Q = np.zeros((self.state_size, self.action_size))
            for s in range(self.state_size):
                for a in range(self.action_size):
                    phi = self._get_features(s, a)
                    Q[s, a] = np.minimum(np.dot(w, phi) + self.reward(h-1, s, a, self.roadmap) + u[s, a] , self.H)
            
            # Compute V
            for s in range(self.state_size):
                V[h-1, s] = np.max(Q[s])
            
            # Compute pi
            for s in range(self.state_size):
                self.pi[h-1, s] = np.argmax(Q[s])

    def collect_trajectories(self, h):
        for n in range(1, self.N+1):
            s, _ = self.env.reset()
            self.D_state[h-1][n-1][0] = s
            for t in range(1, self.H+1):
                a = random.sample(self.PI, 1)[0][t-1][s] # Can turn this into a function for clarity
                phi = self._get_features(s, a)
                self.Sigma[h-1] += np.outer(phi, phi)
                s, _, _, _, _ = self.env.step(a)
                self.D_state[h-1][n-1][t-1] = s
                self.D_action[h-1][n-1][t-1] = a
        self.Sigma_inv[h-1] = np.linalg.inv(self.Sigma[h-1])

    
    def train(self):
        # Train the agent from scratch
        for h in range(1, self.H+1):
            self.Sigma_tilde = 2 * np.eye(self.d)
            pi_curr = np.zeros((self.H, self.state_size), dtype=np.int32) # Random deterministic policy
            self.PI = [] # refresh mixture of policies buffer
            
            for i in range(1, self.i_max+1):
                # print(f"Exploring layer {h}, iteration {i}")
                Lambda = self.estimate_covariance_matrix(h, pi_curr)
                # print(f"Completed estimation of covariance matrix for layer {h}, iteration {i}")
                self.Sigma_tilde += Lambda
                v, pi_curr = self.solve_opt_Q(h)
                # print(f"Completed solving for optimal Q for layer {h}, iteration {i}")
                if 8*v <= 3*self.v_min_squared: # TODO: Investigate behaviour of this
                    break
                self.PI.append(pi_curr)
            

            # Collect trajectories
            self.Sigma[h-1] = np.eye(self.d)
            self.collect_trajectories(h)

            print(f"Explored layer {h}")
        

        self.plan_pi()

# a -> 0: LEFT  1: DOWN  2: RIGHT  3: UP
def next_pos(i, j, a, MAP_DIMENSION): 
    direction = {0:[0, -1], 1:[1, 0], 2:[0, 1], 3:[-1, 0]}
    ii, jj = direction[a][0]+i, direction[a][1]+j
    out = (ii<0 or ii>=MAP_DIMENSION) or (jj<0 or jj>=MAP_DIMENSION)
    return ii, jj, out
    
    
# Define rewards for FrozenLake
def FrozenLakeReward(h, s, a, roadmap):
    i, j = s//MAP_DIMENSION, s%MAP_DIMENSION
    ii, jj, out = next_pos(i, j, a, MAP_DIMENSION)
    next_state = roadmap[ii][jj] if not out else ''
    
    bonus = (a == 1 or a == 2)*0.1
    if next_state== 'G':
        return 1
    elif next_state == 'F':
        return 0.1+bonus
    elif next_state == 'H':
        return -1
    elif out:
        return -1
    else:
        return 0


def print_policy(pi, H, state_size):
    for s in range(state_size):
        print(f"State {s}: ", end="")
        for h in range(int(H)):
            print(INT_TO_ASCII_ARROWS[int(pi[h, s])], end=" ")
        print("")
    

if __name__=="__main__":
    map_descs = {4: ["SFFF", "FHFH", "FFFH", "HFFG"], 2: ["SH", "FG"]}
    map_descs.setdefault(MAP_DIMENSION, generate_random_map(size=MAP_DIMENSION, p=0.5))

    #roadmap = map_descs[MAP_DIMENSION]
    roadmap = generate_random_map(size=MAP_DIMENSION, p=0.5)
    env = gym.make('FrozenLake-v1', desc = map_descs[MAP_DIMENSION], is_slippery = False)
    agent = SDERLAgent(env = env, reward = FrozenLakeReward, roadmap=roadmap)  

    agent.train() # Train agent

    print_policy(agent.pi, agent.H, agent.state_size) # Print policy
    
    
    # See if just layer 1 policy works well enough
    #env_vis = gym.make('FrozenLake-v1', desc = map_descs[MAP_DIMENSION], is_slippery = False, render_mode = "human")
    env_vis = gym.make('FrozenLake-v1', desc = roadmap, is_slippery = False, render_mode = "human")
    
    for e in range(10): # 10 episodes
        s, _ = env_vis.reset()
        done = False
        r_sum = 0
        t = 0
        while not done and t < TIME_HORIZON:
            t += 1
            a = int(agent.pi[t-1, s])
            s, r, done, _, _ = env_vis.step(a)
            r_sum += r
            if done or t == TIME_HORIZON:
                print(f"Episode {e+1} finished with reward {r_sum}")
                break
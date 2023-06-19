import numpy as np # linear algebra
from utils.feature import state_feature
from utils.CartPole import CartPoleEnv
import random
import Hyperparameters as hp

DEBUG = False

# TODO: Maybe do some parameter sharing, since the MDPs we work with are not time dependent
class ContinuousDiscreteDERLAgent: # Continuous Deterministic Deployment Efficient Reinforcement Learning Agent (Algorithm 1 from paper)
    def __init__(self, env, reward, params=hp.default_continuous):
        self.env = env
        self.reward = reward

        # env dependent parameters
        # self.state_size = env.observation_space.shape[0] # this is for input of neural network node size
        # self.action_size = env.action_space.n # this is for out of neural network node size
        # We hardcode first to get it working then figure out how to generalize
        self.state_size = 4  # width * height of map
        self.action_size = 2
        self.fourier_order = 1 # order of fourier feature approximation
        self.d = ((self.fourier_order+1)** self.state_size) * self.action_size # dimension of the (fourier) feature vector

        # hyperparameters
        self.delta = params['delta'] # confidence parameter
        self.epsilon = params['epsilon'] # target accuracy
        self.c_beta = params['c_beta'] # bonus coefficient
        self.H = params['H']  # time horizon
        self.K = params['K'] # number of deployments
        self.N = params['N'] # batch size
        self.explore_prob = params.get('explore_prob') if params.get('explore_prob') is not None else 0.6

        self.beta = self.c_beta * self.d * self.H * np.sqrt(np.log(self.d * self.H / (self.delta*self.epsilon ))) # hyperparameter, unclear exactly what it does
             
        # Q, V, etc. not stored anymore, will be computed on the fly
        # Stored values for linear function approximation
        self.lambda_inv = np.zeros((self.d, self.d)) # inverse of lambda. dxd matrix
        self.w = np.zeros((self.H, self.d)) # w vector buffer. Always use latest w vector, so no need for deployment index.

        self.h = 1 # current layer
        
        # data buffers
        self.state_buffer = np.zeros((self.K, self.N, self.H+1, self.state_size)) # (Num deployments, batch size, time horizon+1). Extra time horizon since we have one extra state at the end
        self.action_buffer = np.zeros((self.K, self.N, self.H)) # (Num deployments, batch size, time horizon)
        self.feature_buffer = np.zeros((self.K, self.N, self.H, self.d)) # (Num deployments, batch size, time horizon, feature vector dimension)


        self.c_critic = state_feature(self.fourier_order) # avoids repeated computations

    def _get_features(self, state, action):
        # returns the fourier feature vector for the given state and action
        features = np.zeros((self.d))
        features[action*(self.d//self.action_size):(action+1)*(self.d//self.action_size)] = np.cos(np.pi * self.c_critic.dot(state))
        return features

    def policy_update_step(self, k): #update policy for deployment k
        # Q, V already initialized to 0 properly, so no need to do this here
        lambda_k = np.eye(self.d)
        for h in range(self.h, 0, -1):
            
            # Compute lambda
            for tau in range(1, k):
                for n in range(1, self.N+1):
                    phi = self.feature_buffer[tau-1, n-1, h-1]
                    lambda_k += np.outer(phi, phi)
            #print(lambda_k)
            self.lambda_inv = np.linalg.inv(lambda_k) # Store inverse for later use
            
            # Compute w
            w = np.zeros(self.d)
            for tau in range(1, k):
                for n in range(1, self.N+1):
                    phi = self.feature_buffer[tau-1, n-1, h-1] # maybe ditch feature buffer and just compute on the fly?
                    w += phi * self.compute_V(h+1, self.state_buffer[tau-1, n-1, h]) # TODO: Check if this is correct
            self.w[h-1] = np.dot(self.lambda_inv, w)
        
        # Don't compute anything else. All else done on the fly when needed.


    def compute_u(self, h, state, action):
        # Compute u for a given layer, state, and action
        phi = self._get_features(state, action)
        return np.minimum(self.beta * np.sqrt(np.linalg.multi_dot((phi, self.lambda_inv, phi))), self.H)

    def compute_Q(self, h, state, action):
        # Compute Q for a given layer, state, and action
        phi = self._get_features(state, action)
        return np.dot(phi, self.w[h-1]) + self.compute_u(h, state, action) + self.reward(state, action) # TODO: Fix this
        # TODO: Do we do the min with H?
    
    def compute_V(self, h, state):
        # Compute V for a given layer and state
        if h > self.h:
            return 0 # for computing w
        return np.max(np.asarray([self.compute_Q(h, state, action) for action in range(self.action_size)]))
    
    def act(self, h, state):
        # If h not explored, sample from uniform distribution
        if self.h < h:
            return self.env.action_space.sample()
        else:
            return 1 if self.compute_Q(h, state, 0) < self.compute_Q(h, state, 1) else 0 # Specific to cartpole, can be generalized by taking highest value aciton

    def continue_step(self, k):
        # Decide what to do next iteration. True means we are done, False means we continue
        delta_k = 0
        for n in range(1, self.N+1):
            for h in range(1, self.h+1):
                phi = self.feature_buffer[k-1, n-1, h-1]
                #delta_k += np.sqrt(np.dot(phi, np.dot(self.lambda_inv[h-1], phi))) #TODO: check if matrix multiplication is correct
                delta_k += np.sqrt(np.linalg.multi_dot((phi, self.lambda_inv, phi)))

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

        # if random.random() > 0:
        #     self.h += 1
        #     return self.h-1 == self.H 
        # else:
        #     return False

    
    def collect_trajectories_step(self, k):
        # Collect trajectories for deployment k
        for n in range(1, self.N+1):
            s = self.env.reset()
            self.state_buffer[k-1, n-1, 0] = s
            for h in range(1, self.H+1):
                a = self.act(h, s)
                s, _, _, _ = self.env.step(a)
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
    

# if __name__=="__main__":
#     env = CartPoleEnv(max_ep_len=TIME_HORIZON, seed = 0, rew_shift=0, append=True) # Modified env for training
#     agent = Agent(env = env, reward = env.reward) # Initialize agent

#     pi = agent.train() # Train agent
    
    
#     env_vis = CartPoleEnv(max_ep_len=TIME_HORIZON, seed = 0, rew_shift=0, append=True) # Maybe do actual carpole env here
    
#     r_total_pi = 0
#     for e in range(10):
#         s = env_vis.reset()
#         done = False
#         r_sum = 0
#         t = 0
#         while not done and t < TIME_HORIZON:
#             # env_vis.render()
#             t += 1
#             a = agent.act(t, s)
#             print(a)
#             s, r, done, _ = env_vis.step(a)
#             r_sum += r
#             if done or t == TIME_HORIZON:
#                 print(f"Episode {e+1} finished with reward {r_sum}")
#                 r_total_pi += r_sum
#                 if done:
#                     print(f"Episode finished after {t} timesteps")
#                 break
    
#     # Comparing to random policy:
#     r_total_random = 0
#     for e in range(10):
#         print("Episode: ", e)
#         s = env_vis.reset()
#         done = False
#         r_sum = 0
#         t = 0
#         while not done and t < TIME_HORIZON:
#             env_vis.render()
#             t += 1
#             a = random.sample([0,1], 1)[0]
#             s, r, done, _ = env_vis.step(a)
#             r_sum += r
#             if done or t == TIME_HORIZON:
#                 print(f"Episode {e+1} finished with reward {r_sum}")
#                 r_total_random += r_sum
#                 if done:
#                     print(f"Episode finished after {t} timesteps")
#                 break
    
#     print(f"Random policy: {r_total_random/10}")
#     print(f"Our policy: {r_total_pi/10}")


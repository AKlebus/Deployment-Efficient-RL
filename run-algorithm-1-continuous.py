import random
from ContinuousDiscreteDERLAgent import ContinuousDiscreteDERLAgent as Agent
from utils.CartPole import CartPoleEnv

DEBUG = False
TIME_HORIZON = 100

NUM_TEST_EPISODES = 200

if __name__=="__main__":
    env = CartPoleEnv(max_ep_len=TIME_HORIZON, seed = 0, rew_shift=0, append=True) # Modified env for training
    agent = Agent(env = env, reward = env.reward)

    agent.train() 
    
    env_vis = CartPoleEnv(max_ep_len=TIME_HORIZON, seed = 0, rew_shift=0, append=True) # Maybe do actual carpole env here
    
    r_total_pi = 0
    for e in range(NUM_TEST_EPISODES): 
        s = env_vis.reset()
        done = False
        r_sum = 0
        t = 0
        while not done and t < TIME_HORIZON:
            # env_vis.render()
            t += 1
            a = agent.act(t, s)
            print(a)
            s, r, done, _ = env_vis.step(a)
            r_sum += r
            if done or t == TIME_HORIZON:
                print(f"Episode {e+1} finished with reward {r_sum}") if DEBUG else None
                r_total_pi += r_sum
                if done:
                    print(f"Episode finished after {t} timesteps") if DEBUG else None
                break
    
    # Comparing to random policy:
    r_total_random = 0
    for e in range(NUM_TEST_EPISODES): 
        print("Episode: ", e) if DEBUG else None
        s = env_vis.reset()
        done = False
        r_sum = 0
        t = 0
        while not done and t < TIME_HORIZON:
            print("timestep: ", t) if DEBUG else None
            #env_vis.render()  # TODO: vis seems to be broken -- Thomas
            t += 1
            a = random.sample([0,1], 1)[0]
            s, r, done, _ = env_vis.step(a)
            r_sum += r
            if done or t == TIME_HORIZON:
                print(f"Episode {e+1} finished with reward {r_sum}") if DEBUG else None
                r_total_random += r_sum
                if done:
                    print(f"Episode finished after {t} timesteps") if DEBUG else None
                break
    
    print(f"Random policy: {r_total_random/NUM_TEST_EPISODES}")
    print(f"Our policy: {r_total_pi/NUM_TEST_EPISODES}")

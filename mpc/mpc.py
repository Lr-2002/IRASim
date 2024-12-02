import numpy as np
import gym


class CEM_MPC:
    def __init__(self, env, horizon=200, pop_size=100, elite_frac=0.2, max_iters=5):
        self.env = env
        self.horizon = horizon
        self.pop_size = pop_size
        self.elite_frac = elite_frac
        self.max_iters = max_iters
        
        self.action_dim = env.action_space.n  
        self.state_dim = env.observation_space.shape[0]
        
        self.mean = np.zeros((self.horizon, self.action_dim))
        self.std = np.ones((self.horizon, self.action_dim))
    
    def evaluate_action_sequences(self, sequences):
        """
            irasim will input  a batch of action sequences and a batch of image  return a batch of rewards
        """
        rewards = []
        for seq in sequences:
            obs = self.env.reset()
            total_reward = 0.0
            for action in seq:
                obs, reward, done, _ = self.env.step(np.argmax(action)) 
                total_reward += reward
                if done:
                    break
            rewards.append(total_reward)
        return np.array(rewards)
 

    def evaluate_in_world_model(self, init_state, action_sequences, worldmodel=None):
        """
            wm input:
                init_state: c, h, w 
                action_sequences: b, t, action_dim
                
            wm output:
                reward 

        """
        if worldmodel is not None:
            self.worldmodel = worldmodel    
        assert hasattr(self, 'worldmodel')
        rewards = worldmodel.evaluate(init_state, action_sequences) 

        pass

    def optimize(self, observation):
        for _ in range(self.max_iters):
            sequences = np.random.normal(self.mean, self.std, (self.pop_size, self.horizon, self.action_dim))
            
            rewards = self.evaluate_action_sequences(sequences)

            elite_idx = rewards.argsort()[-int(self.pop_size * self.elite_frac):]
            elite_sequences = sequences[elite_idx]
        
            self.mean = elite_sequences.mean(axis=0)
            self.std = elite_sequences.std(axis=0)
        

        return np.argmax(self.mean[0])

if __name__ == '__main__':  
    env_real = gym.make("CartPole-v1") # make lt sim 
    env_model= gym.make("CartPole-v1")
    cem_mpc = CEM_MPC(env_model)

    episodes = 100
    for ep in range(episodes):
        observation = env_real.reset()
        total_reward = 0
        done = False
    
        while not done:
            env_real.render()# no need to render 
            action = cem_mpc.optimize(observation) # op 
            observation, reward, done, _ = env_real.step(action)
            total_reward += reward
    
        print(f"Episode {ep + 1}, Total Reward: {total_reward}")

    env_real.close()
    env_model.close()


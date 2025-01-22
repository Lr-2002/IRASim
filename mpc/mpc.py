import numpy as np
import gym

# from language_table.language_table.environments import blocks
# from language_table.language_table.environments import language_table
# from language_table.language_table.environments.rewards import block2block
from models.world_model import WorldModel
from matplotlib import pyplot as plt

class CEM_MPC:
    def __init__(self, env=None,action_dim=2, horizon=16, pop_size=16, elite_frac=0.2, max_iters=8,discrete=False):
        
        self.horizon = horizon
        self.pop_size = pop_size
        self.elite_frac = elite_frac
        self.max_iters = max_iters
        if env is not None:
            self.env = env
            self.action_dim = env.action_space.shape[0]
        else:
            self.action_dim = action_dim  
        #print(env.action_space.shape)
        #self.state_dim = env.observation_space.shape[0]
        
        self.mean = np.zeros((self.horizon, self.action_dim))
        self.std = 0.1*np.ones((self.horizon, self.action_dim))
        self.discrete = discrete
    
    def evaluate_action_sequences(self, sequences):
        """
            irasim will input  a batch of action sequences and a batch of image  return a batch of rewards
        """
        rewards = []
        for seq in sequences:
            obs = self.env.reset()
            total_reward = 0.0
            for action in seq:
                if self.discrete:
                    obs, reward, done, _ = self.env.step(np.argmax(action)) 
                else:
                    obs, reward, done, _ = self.env.step(action)
                    #print(action)
                total_reward += reward
                if done:
                    break
            rewards.append(total_reward)
            #print(total_reward)
        return np.array(rewards)
 

    def evaluate_in_world_model(self, init_state, action_sequences,goal_image=None, worldmodel=None):
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
        rewards = worldmodel(init_state, action_sequences, goal_image) 

        return rewards

    def optimize(self, observation,worldmodel=None):
        #plt.ion()
        #plt.imshow(observation)
        #plt.show()
        self.mean = np.zeros((self.horizon, self.action_dim))
        self.std = 0.01*np.ones((self.horizon, self.action_dim))
        for _ in range(self.max_iters):
            sequences = np.random.normal(self.mean, self.std, (self.pop_size, self.horizon, self.action_dim))
            #print(sequences.shape)
            if worldmodel is not None:
                # if len(observation.shape) ==3 and observation.shape[2]==3:
                #     observation = np.transpose(observation,(2,0,1))
                # elif len(observation.shape) ==2:
                #     observation = np.expand_dims(observation,axs=0)
                #print(observation)
                rewards = self.evaluate_in_world_model(observation,sequences,worldmodel) #.squeeze(-1).sum(axis=1)
            else:
                rewards = self.evaluate_action_sequences(sequences)
            #print(rewards.shape)

            rewards = np.array(rewards)
            elite_idx = rewards.argsort()[-int(self.pop_size * self.elite_frac):]
            elite_sequences = sequences[elite_idx]
        
            self.mean = elite_sequences.mean(axis=0)
            self.std = elite_sequences.std(axis=0)
            #print(self.mean)
            #print(self.mean)
        
        return self.mean[0],rewards.mean()#np.argmax(self.mean[0])

if __name__ == '__main__':  
    env_real = language_table.LanguageTable(
      block_mode=blocks.LanguageTableBlockVariants.BLOCK_8,
      reward_factory=block2block.BlockToBlockReward,
      control_frequency=10.0,
  )      # make lt sim 
    env_model= language_table.LanguageTable(
      block_mode=blocks.LanguageTableBlockVariants.BLOCK_8,
      reward_factory=block2block.BlockToBlockReward,
      control_frequency=10.0,
  )
    horizon = 16
    cem_mpc = CEM_MPC(env_model)

    episodes = 10
    for ep in range(episodes):
        for _ in range(horizon):
            observation = env_real.reset()
            total_reward = 0
            done = False

            #env_real.render()# no need to render 
            rendered_image = env_real.render()
            action = cem_mpc.optimize(rendered_image) # op 
            observation, reward, done, _ = env_real.step(action)
            #print(reward)
            total_reward += reward
            if done:
                break
    
        print(f"Episode {ep + 1}, Total Reward: {total_reward}")

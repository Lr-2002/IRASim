import numpy as np
from mpc.mpc import CEM_MPC
import torch


class DiscreteMPC(CEM_MPC):
    def __init__(
        self,
        env=None,
        action_dim=2,
        horizon=15,
        pop_size=16,
        elite_frac=0.2,
        max_iters=4,
    ):
        super(DiscreteMPC, self).__init__(
            env=env,
            action_dim=action_dim,
            horizon=horizon,
            pop_size=pop_size,
            elite_frac=elite_frac,
            max_iters=max_iters,
            discrete=True,
        )
        sequences = [np.zeros([15, 2]) for _ in range(8)]
        action1 = [1, 0]
        action2 = [0, 1]
        action3 = [1, 1]
        action4 = [-1, 0]
        action5 = [0, -1]
        action6 = [-1, -1]
        action7 = [1, -1]
        action8 = [-1, 1]
        actions = [
            action1,
            action2,
            action3,
            action4,
            action5,
            action6,
            action7,
            action8,
        ]
        for i in range(8):
            sequences[i][0:5] = np.tile(np.array(actions[i]), (5, 1))
        self.sequences = 0.02 * np.array(sequences)

    #    self.action_dim = env.action_space.n

    # def evaluate_action_sequences(self, sequences):
    #     """
    #         irasim will input  a batch of action sequences and a batch of image  return a batch of rewards
    #     """
    #     rewards = []
    #     for seq in sequences:
    #         obs = self.env.reset()
    #         total_reward = 0.0
    #         for action in seq:
    #             obs, reward, done, _ = self.env.step(action)
    #             total_reward += reward
    #             if done:
    #                 break
    #         rewards.append(total_reward)
    #     return np.array(rewards)
    def sample_multiple_action_trajectories(self, mean, std, num_trajectories):
        sampled_trajectories = [
            self.round_action_to_closest_action_set(torch.normal(mean, std).clip(-1, 1))
            for _ in range(num_trajectories)
        ]
        return torch.stack(sampled_trajectories)

    def round_action_to_closest_action_set(self, action):
        """
        Input is a tensor of shape (T, D), where T is the number of steps, and D is the dimension of the action space
        So each action is a vector of shape (D,), and each dimension the value is in the range of [-1, 1]
        We currently discretize the action space into 3 levels: -1, 0, 1.
        Suppose initially an action is uniformly sampled from [-1,1], we need to ensure the probability of it being round to each level is the same
        So -1 to -1/3 round to -1, -1/3 to 1/3 round to 0, 1/3 to 1 round to 1
        """
        # Define thresholds
        lower_threshold = -1 / 3
        upper_threshold = 1 / 3

        # Create mask for each region
        low_mask = action <= lower_threshold
        mid_mask = (action > lower_threshold) & (action < upper_threshold)
        high_mask = action >= upper_threshold

        # Apply rounding based on masks
        rounded_action = torch.zeros_like(action)
        rounded_action[low_mask] = -1
        rounded_action[mid_mask] = 0
        rounded_action[high_mask] = 1

        # scale the action for the step size, generally a small number like 0.02
        rounded_action = (
            rounded_action * 0.02
        )  # .to(self.device) * self.cfg.algorithm.df_planning.action_set_step_scale
        # print("--------rounded_action",rounded_action)
        return rounded_action

    def optimize(self, observation, goal_image=None, worldmodel=None, start_idx=None):
        """
        irasim will input  a batch of action sequences and a batch of image  return a batch of rewards
        """
        self.worldmodel = worldmodel
        # sequences = self.sequences.copy()
        # print(sequences)

        mean = torch.zeros(self.horizon, self.action_dim)
        std = torch.ones(self.horizon, self.action_dim)

        means = [mean]
        stds = [std]
        iteration = 0
        is_converged = False
        while iteration < self.max_iters:
            if is_converged:
                break
            # print("mean",mean

            sampled_action_trajectories = self.sample_multiple_action_trajectories(
                mean, std, self.pop_size
            )
            # rollout_trajectories = []
            rollout_rewards = []
            rollout_rewards = self.evaluate_in_world_model(
                observation, sampled_action_trajectories, goal_image, worldmodel
            )
            # for action_trajectory in sampled_action_trajectories:
            #     #predicted_slots_trajectory = self.evaluate_in_world_model(observation,action_trajectory, goal_image,worldmodel)
            #     #reward = self.calculate_reward(predicted_slots_trajectory, goal_state_slots, slots_indices_for_reward_calculation)
            #     rewards = self.evaluate_in_world_model(observation,action_trajectory,goal_image,worldmodel)

            #     rollout_trajectories.append(predicted_slots_trajectory)
            #     rollout_rewards.append(reward)

            #     rewards = self.evaluate_in_world_model(observation,action_trajectory,goal_image,worldmodel)
            ranked_trajectories_and_reward = sorted(
                zip(rollout_rewards, sampled_action_trajectories),
                key=lambda pair: pair[0],
                reverse=True,
            )

            ranked_action_trajectories = [
                pair[1] for pair in ranked_trajectories_and_reward
            ]
            top_action_trajectories = ranked_action_trajectories[
                : int(self.pop_size * self.elite_frac)
            ]
            top_action_trajectories = torch.stack(top_action_trajectories)
            mean = (
                torch.mean(top_action_trajectories, dim=0) * 5
            )  # this is to scale the action to the range of [-1,1]
            std = torch.std(top_action_trajectories, dim=0) * 25
            means.append(mean)
            stds.append(std)

            iteration += 1
        print(
            "-------------------top_action_trajs---------------",
            top_action_trajectories,
        )

        return top_action_trajectories[0][0:3], ranked_trajectories_and_reward[0][0]
        # rewards = self.evaluate_in_world_model(observation,sequences,goal_image,worldmodel)
        # # print(rewards)
        # #rewards = np.array(rewards)

        # # Detach rewards tensor and convert to numpy array
        # rewards = np.array([reward.detach().numpy() if isinstance(reward, torch.Tensor) else reward for reward in rewards])

        # elite_idx = rewards.argsort()[-1]
        # # print(elite_idx)

        # # elite_sequences = sequences[elite_idx]
        # # print(sequences)
        # # print(sequences[elite_idx])
        # # print(sequences[elite_idx][0])
        # return sequences[elite_idx][0] , rewards[elite_idx]


if __name__ == "__main__":
    mpc = DiscreteMPC()

    print(mpc.sequences)

    class worldmodel:
        def __init__(self):
            pass

        def __call__(self, init_state, action_sequences):
            return np.random.rand(8)

    actions = mpc.optimize(np.random.rand(3, 3, 3), worldmodel())
    print(actions)

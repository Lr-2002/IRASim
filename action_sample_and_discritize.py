    def sample_multiple_action_trajectories(self, mean, std, num_trajectories ):
        sampled_trajectories = [
             self.round_action_to_closest_action_set(torch.normal(mean, std).clip(-1, 1).to(self.device)) for _ in range(num_trajectories)
        ]
        return torch.stack(sampled_trajectories)

    def round_action_to_closest_action_set(self, action):
        '''
        Input is a tensor of shape (T, D), where T is the number of steps, and D is the dimension of the action space
        So each action is a vector of shape (D,), and each dimension the value is in the range of [-1, 1]
        We currently discretize the action space into 3 levels: -1, 0, 1.
        Suppose initially an action is uniformly sampled from [-1,1], we need to ensure the probability of it being round to each level is the same
        So -1 to -1/3 round to -1, -1/3 to 1/3 round to 0, 1/3 to 1 round to 1
        '''
        # Define thresholds
        lower_threshold = -1/3
        upper_threshold = 1/3
        
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
        rounded_action = rounded_action.to(self.device) * self.cfg.algorithm.df_planning.action_set_step_scale

        return rounded_action

    def sample_action_by_CEM(self, goal_state_slots, cur_state_slots, slots_indices_for_reward_calculation):
        '''
        initialize the mean and std of the action distribution as 0 and 1
        1. get random samples from the distribution
        2. rollout the future trajectories and calculate the reward for each trajectory
        3. rank the trajectories and take the K best trajectory to update the mean and std of the action distribution
        4. iterate the above process for N times until the mean and std of the action distribution converge or reach the max iteration
        '''
        mean = torch.zeros(self.cfg.algorithm.df_planning.rollout_horizon, self.env.action_dim, device=self.device)
        std = torch.ones(self.cfg.algorithm.df_planning.rollout_horizon, self.env.action_dim, device=self.device)

        iteration = 0
        is_converged = False
        means = [mean]
        stds = [std]
        while iteration < self.cfg.algorithm.df_planning.max_cem_iteration:
            if not self.cfg.algorithm.df_planning.fixed_iteration and is_converged:
                break

            sampled_action_trajectories = self.sample_multiple_action_trajectories(mean, std, self.cfg.algorithm.df_planning.num_trajectories_to_sample_each_CEM_iter)
            
            # rollout the future trajectories and calculate the reward for each trajectory
            rollout_trajectories = []
            rollout_rewards = []
            for action_trajectory in sampled_action_trajectories:
                predicted_slots_trajectory = self.world_model_forward(action_trajectory, cur_state_slots)
                reward = self.calculate_reward(predicted_slots_trajectory, goal_state_slots, slots_indices_for_reward_calculation)
                rollout_trajectories.append(predicted_slots_trajectory)
                rollout_rewards.append(reward)

            # rank the trajectories and take the best trajectory first action
            # ranked_rollout_slots_trajectories = sorted(rollout_trajectories, key=lambda x: rollout_rewards[rollout_trajectories.index(x)], reverse=True)
            ranked_trajectories_and_reward = sorted(
                zip(rollout_trajectories, rollout_rewards, sampled_action_trajectories),
                key=lambda pair: pair[1],
                reverse=True
            )
            ranked_rollout_slots_trajectories = [pair[0] for pair in ranked_trajectories_and_reward]
            top_rollout_slots_trajectories = ranked_rollout_slots_trajectories[:self.cfg.algorithm.df_planning.cem_k]
            ranked_action_trajectories = [pair[2] for pair in ranked_trajectories_and_reward]
            top_action_trajectories = ranked_action_trajectories[:self.cfg.algorithm.df_planning.cem_k]

            # update the mean and std of the action distribution
            top_action_trajectories = torch.stack(top_action_trajectories)
            mean = torch.mean(top_action_trajectories, dim=0)
            std = torch.std(top_action_trajectories, dim=0)
            means.append(mean)
            stds.append(std)

            is_converged = self.is_distribution_converged(means, stds)
            iteration += 1
        best_next_action = top_action_trajectories[0][0]

        output_dict = {
            "best_next_action": best_next_action,
            "top_rollout_slots_trajectories": top_rollout_slots_trajectories,
            "means": means,
            "stds": stds
        }
        return output_dict
 

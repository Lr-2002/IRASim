"""
This module contains the main body of the trajectory generation part of the Object-level Action World Model (OAWM) algorithm using Diffusion Forcing method.
This script needs to implement the abstractions from the template https://github.com/buoyancy99/research-template
"""

from tqdm import tqdm
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


import cv2
import sys
sys.path.append('env/language-table')
from dotmap import DotMap
from language_table.language_table.environments import blocks
from language_table.language_table.environments import language_table
from language_table.language_table.environments.rewards import block2block


class EnvWrapperBase:
    def __init__(self, cfg: DictConfig):
        """ Initialize the Env Wrapper Base
            self.env: the dynamic simulation environment, currently using the language table environment
            self.mask_bbox_extractor: the mask and bbox extractor using the MaskBBoxExtractorWithSam class, which uses the SAM2 model to extract masks and bboxes
            self.initial_random_num_steps: the number of random steps to take for generating the observation frames
            self.observation_frames: the list of observation frames, each frame is a tensor of shape (1, C, H, W). These frames are used to extract best reference frame for object mask and bbox extraction, and the first frame is the goal image
            self.agent_index: the object index of the agent in the observation frames
            self.gt_num_objects: the ground truth number of objects in the environment
            self.goal_image: the goal image for the planning algorithm
            

        Args:
            cfg (DictConfig): configuration file
        """
        self.cfg = cfg
        # TODO: in the future, we should use the environment from the config file
        self.env = language_table.LanguageTable(
            block_mode=blocks.LanguageTableBlockVariants.BLOCK_8,
            reward_factory=block2block.BlockToBlockReward,
            control_frequency=10.0,
        )
        self.initial_random_num_steps = cfg.environment.environment.initial_random_num_steps
        self.observation_frames = []
        self.goal_image = None
        self.cur_observation_frame = None

    def reset_env(self)->torch.Tensor:
        """
        Reset the environment
        """
        _ = self.env.reset()
        return self.render_video()
        
              
    def step_env(self, use_random_action:bool=False, action:np.ndarray=None)->None:
        """_summary_

        Args:
            random_action (bool, optional): Use random sampled action to step env or not. Defaults to False.
            action (np.ndarray, optional): Action given to step env. Defaults to None.
        *** Note: random_action and action should be exclusive

        """
        assert (use_random_action==False and action is not None) or (use_random_action==True and action is None), "use_random_action and action should be exclusive"
        if use_random_action:                                          # if action is None, then sample a random action                          
            action = self.env.action_space.sample() 
        self.env.step(action)                                       # the action should in the same format as self.env.action_space.sample()
        return self.render_video()
    
    def render_video(self)->torch.Tensor:
        """rendor video frame from the environment

        Returns:
            torch.Tensor: [1, C, H, W] tensor of the rendered video frame
        """
        video = [self.env.render()]                                  # H, W, C
        return torch.from_numpy(np.stack(video)).permute(0, 3, 1, 2)/255 # 1, C, H, W
    
    def generate_random_cur_frame_and_goal_image(self):
        self.env.reset()
        initial_frame = self.render_video()
        self.env.random_move_an_object_to_random_position()
        goal_frame = self.render_video()
        self.observation_frames = [goal_frame, initial_frame]
        return self.observation_frames
        

    

    def take_random_steps_and_go_back_to_initial_state_task_initialization(self, preserve_random_steps_observation:bool=False):
        '''
        1. Initialize the environment, set it as the goal image, 
        2. then take n_steps random steps for each frame
        '''
        # 1. initialize the environment, set it as the goal image, 
        cur_frame = self.reset_env()
        self.observation_frames.append(cur_frame)
        self.goal_image = cur_frame

        # 2. then take n_steps random steps for each frame
        for i in range(self.initial_random_num_steps):
            cur_frame = self.step_env(use_random_action=True)   
            if preserve_random_steps_observation:
                self.observation_frames.append(cur_frame)

        return self.observation_frames
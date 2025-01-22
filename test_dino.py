
import imageio
import os
import argparse
import torch
import json
import numpy as np
from copy import deepcopy
from imageio import get_writer
from einops import rearrange
import torch.nn.functional as F 
from tqdm import tqdm
from diffusers.models import AutoencoderKL
import sys
sys.path.insert(0, "/home/lr-2002/code/IRASim/")
from models import get_models
from dataset import get_dataset
from util import (get_args, requires_grad)
from evaluate.generate_short_video import generate_single_video
from dataset.video_transforms import Resize_Preprocess, ToTensorVideo
import torchvision.transforms as T

from mpc.mpc import CEM_MPC
from language_table.language_table.environments import blocks
from language_table.language_table.environments import language_table
from language_table.language_table.environments.rewards import block2block
from language_table.language_table.environments.rewards import block1_to_corner
import matplotlib.pyplot as plt
#faulthandler.enable()
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import random
from env_wrapper_base import EnvWrapperBase
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from dino_reward_model import DINO_Reward_Model

if __name__ == "__main__":

    cfg_dict = {'environment': {'environment': {'initial_random_num_steps': 0}}}

    # Convert the dictionary to a DictConfig object
    cfg = OmegaConf.create(cfg_dict)
    
    env_real = EnvWrapperBase(cfg)
    env_real.env._render_text_in_image = False
    #print(env_real._image_size)
    
    #env_real._image_size = (444,640)
    [goal_frame, current_frame] = env_real.generate_random_cur_frame_and_goal_image() # 1, C, H, W
    frames=[]
    world_model = DINO_Reward_Model()
    for step in range(1):
        print('---- video_tensor shape is ', current_frame.shape) #---- video_tensor shape is  torch.Size([1, 3, 180, 320])
        video_array = current_frame.squeeze().permute(1, 2, 0).numpy()  # H, W, C ()
        frames.append(current_frame)
    
    # Convert to image and save
        image = Image.fromarray((video_array * 255).astype(np.uint8))
        image.save(f'./application/test/test_frame_{step}.png')
        reward = world_model.calculate_reward(current_frame, goal_frame)
        #plt.imshow(image)
        #plt.axis('off')
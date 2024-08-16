from einops import rearrange, repeat
import imageio
import os
from sample.pipeline_trajectory2videogen import Trajectory2VideoGenPipeline
import argparse
import json
from dataset import get_dataset
from copy import deepcopy
from einops import rearrange
from models import get_models
from diffusion import create_mask_diffusion
from omegaconf import OmegaConf
from diffusers.models import AutoencoderKL
from util import requires_grad, get_args
from diffusers.schedulers import PNDMScheduler 
import numpy as np
import torch
import json
import os
import random
import warnings
import traceback
import argparse
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms as T
import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
import imageio
from decord import VideoReader, cpu
from dataset.dataset_util import euler2rotm, rotm2euler
from concurrent.futures import ThreadPoolExecutor, as_completed
from einops import rearrange
# from dataset.video_transforms import Resize_Preprocess, ToTensorVideo
from torchvision.transforms import Compose, CenterCrop, ToTensor, Normalize, Resize, ToPILImage
import torchvision.transforms.functional as F
from dataset.dataset_util import euler2rotm, rotm2euler, quat2rotm
from util import update_paths
import os
import re
import imageio
import numpy as np


def generate_latent_image(image_path, vae, device, args):
    image = imageio.imread(image_path)

    first_preprocess = Compose([
            ToPILImage(),  # Convert tensor to PIL Image
            # lambda img: F.crop(img, crop_top_left[0], crop_top_left[1], crop_height, crop_width),  # Dynamic crop
            # lambda img: F.center_crop(img, min(crop_height, crop_width)),  # Replace 'crop_size' with your desired size
            Resize(args.video_size),  # Resize to the desired video size, e.g., (288, 512)
        ])

    resize_frist_frame = first_preprocess(image)

    second_preprocess = Compose([
        ToTensor(),  # Convert back to tensor
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    normalized_first_frame = second_preprocess(resize_frist_frame)
    normalized_first_frame = rearrange(normalized_first_frame, 'c h w -> 1 c h w').contiguous()
    normalized_first_frame = normalized_first_frame.to(device)
    with torch.no_grad():
        latent_first_frame = vae.encode(normalized_first_frame).latent_dist.mode().mul_(vae.config.scaling_factor)
    return normalized_first_frame, latent_first_frame
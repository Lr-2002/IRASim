# Copyright (2024) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Union
import logging
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration class for video processing parameters."""
    video_size: Tuple[int, int] = (288, 512)
    arrow_position: Tuple[int, int] = (200, 240)
    max_frames: int = 5
    fps: int = 4
    valid_actions: List[str] = ('w', 'a', 's', 'd', ' ')
    scale_factors: dict = None

    def __post_init__(self):
        self.scale_factors = {
            'left': 0.5,
            'right': -0.5,
            'up': 0.5,
            'down': -0.5
        }

class VideoProcessor:
    """Class for handling video processing operations."""
    
    def __init__(self, config: Config):
        self.config = config
        self.val_preprocess = self._create_preprocessor()

    def _create_preprocessor(self) -> T.Compose:
        """Create the video preprocessing pipeline."""
        return T.Compose([
            ToTensorVideo(),
            Resize_Preprocess(self.config.video_size),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])

    @staticmethod
    def create_arrow_image(direction: str = 'w') -> np.ndarray:
        """Create an arrow image pointing in the specified direction."""
        try:
            image = imageio.imread('./sample/arrow.jpg')
            if direction == 's':
                image = np.flipud(image)
            elif direction == 'a':
                image = np.rot90(image)
            elif direction == 'd':
                image = np.rot90(image, -1)
            return image
        except Exception as e:
            logger.error(f"Error creating arrow image: {e}")
            raise

    def add_arrows_to_video(self, video_np: np.ndarray, save_path: str, actions: List[str]) -> None:
        """Add direction arrows to the video and save it."""
        if len(actions) != video_np.shape[0]:
            raise ValueError("Actions list length must match number of video frames")

        try:
            writer = get_writer(save_path, fps=self.config.fps)
            for frame, action in zip(video_np, actions):
                if action != ' ':
                    arrow_img = self.create_arrow_image(direction=action)
                    pos_x, pos_y = self.config.arrow_position
                    mask = np.any(arrow_img != 0, axis=-1)
                    frame[pos_x:pos_x+arrow_img.shape[0], pos_y:pos_y+arrow_img.shape[1]][mask] = \
                        arrow_img[mask]
                writer.append_data(frame)
            writer.close()
        except Exception as e:
            logger.error(f"Error adding arrows to video: {e}")
            raise

    def read_actions_from_keyboard(self, num_actions: int = 15) -> List[str]:
        """Read action inputs from keyboard."""
        actions = []
        while len(actions) < num_actions:
            input_actions = input(f"Please enter actions (remaining {num_actions - len(actions)}): ").lower()
            actions.extend(a for a in input_actions if a in self.config.valid_actions and len(actions) < num_actions)
            
            if len(actions) < num_actions:
                logger.info(f"Not enough actions. Please enter the remaining {num_actions - len(actions)} actions.")
        return actions

    @staticmethod
    def reshape(x: torch.Tensor) -> torch.Tensor:
        """Reshape and resize the input tensor."""
        x = x.permute(0, 3, 1, 2)
        x_resized = F.interpolate(x, size=(288, 512), mode='bilinear', align_corners=False)
        return x_resized.permute(0, 2, 3, 1)

def process_video(video_path: str, vae: AutoencoderKL, processor: VideoProcessor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process video file and return processed tensors."""
    try:
        video_reader = imageio.get_reader(video_path)
        video_tensor = []
        
        for idx, frame in enumerate(video_reader):
            if idx > processor.config.max_frames:
                break
            video_tensor.append(torch.tensor(frame))
        
        video_reader.close()
        video_tensor = torch.stack(video_tensor)

        with torch.inference_mode():
            frames = video_tensor.permute(0, 3, 1, 2).cuda()
            frames = processor.val_preprocess(frames)
            frames = vae.encode(frames).latent_dist.sample().mul_(vae.config.scaling_factor)[:2]

        return frames, video_tensor
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise

def main(device: str, model: torch.nn.Module, vae: AutoencoderKL, args: argparse.Namespace, video_id: int = 0):
    """Main function for video processing and generation."""
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    config = Config()
    processor = VideoProcessor(config)

    video_path = Path(f'/home/lr-2002/code/IRASim/val/{video_id}/rgb.mp4')
    ann_file = Path("/home/lr-2002/code/IRASim/robotdata/opensource_robotdata/languagetable/annotation/val/000016.json")

    try:
        with open(ann_file, "r") as f:
            ann = json.load(f)
            logger.info(f'Loaded annotation: {ann}')

        latent_video, video_tensor = process_video(str(video_path), vae, processor)
        
        game_dir = Path('application/languagetable_game_short_action_sim')
        game_dir.mkdir(exist_ok=True)
        logger.info(f'Created game directory: {game_dir}')

        start_idx = 0
        start_image = latent_video[start_idx]
        video_tensor = video_tensor[start_idx:]

        seg_idx = 0

        video_tensor = video_tensor
        video_tensor = processor.reshape(video_tensor)
        seg_video_list = [video_tensor[0:1].numpy()] 

        action_scaler = [20.0, 20.0] 
        action_scaler = np.array(action_scaler)

        while True:
            if seg_idx == 1 : 
                break 
            action = ann['actions']
            action = action * action_scaler 
            if action is not None:
                actions = action
            else:
                env_actions = processor.read_actions_from_keyboard()
                actions = []
                for action in env_actions:
                    if action == 'd':
                        actions.append([0,config.scale_factors['up']])
                    elif action == 'a':
                        actions.append([0,config.scale_factors['down']])
                    elif action == 's':
                        actions.append([config.scale_factors['left'],0])
                    elif action == 'w':
                        actions.append([config.scale_factors['right'],0])
                    else:
                        actions.append([0,0])
            actions = torch.from_numpy(np.array(actions)[:15])

            seg_action = actions
            start_image = start_image.unsqueeze(0).unsqueeze(0)
            seg_action = seg_action.unsqueeze(0)
            seg_video, seg_latents = generate_single_video(args, start_image , seg_action, device, vae, model)
            seg_video = seg_video.squeeze()
            seg_latents = seg_latents.squeeze()
            start_image = seg_latents[-1].clone()

            t_videos = ((seg_video / 2.0 + 0.5).clamp(0, 1) * 255).detach().to(dtype=torch.uint8).cpu().contiguous()
            t_videos = rearrange(t_videos, 'f c h w -> f h w c')
            t_videos = t_videos.numpy()
            seg_video_list.append(t_videos[1:])
            all_video = np.concatenate(seg_video_list,axis=0)
            output_video_path = os.path.join(game_dir,f'{video_id}_{seg_idx}-th.mp4')
            writer = get_writer(output_video_path, fps=4)
            for frame in all_video:
                writer.append_data(frame)
            writer.close()
            logger.info(f'Generated video: {output_video_path}')
            seg_idx += 1
            del actions, seg_video, seg_latents

        exclude_list = [device, model, None, vae]
        for name, obj in globals().items():
            if torch.is_tensor(obj) and obj.is_cuda and obj not in exclude_list:
                logger.info(f'The object is {name}')

        del frames, start_image, latent_video

        del action
        import gc 
        gc.collect()
        torch.cuda.empty_cache()
        for name, obj in globals().items():
            if torch.is_tensor(obj) and obj.is_cuda and obj not in exclude_list:
                logger.info(f'After delete the object is {name}')

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return None
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/evaluation/languagetable/frame_ada.yaml")
    args = parser.parse_args()
    args = get_args(args)
    args.latent_size = [t // 8 for t in args.video_size]
    device = torch.device("cuda", 0)
    model = get_models(args)
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    vae = AutoencoderKL.from_pretrained(args.vae_model_path, subfolder="vae").to(device)
    logger.info(f'Arguments: {args}')
    checkpoint = torch.load(args.evaluate_checkpoint, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:
        logger.info('Using ema ckpt!')
        checkpoint = checkpoint["ema"]

    model_dict = model.state_dict()
    pretrained_dict = {}
    for k, v in checkpoint.items():
        if k in model_dict:
            pretrained_dict[k] = v
        else:
            logger.info(f'Ignoring: {k}')
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.to(device)
    model.eval()

    from tqdm import tqdm
    for video_id in tqdm(os.listdir('./val/')):
        main(device, model, vae, args, video_id)

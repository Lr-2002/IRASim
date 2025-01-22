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

from tqdm import tqdm
import imageio
import os
import argparse
import torch
import json
import torch
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
from util import get_args, requires_grad
from evaluate.generate_short_video import generate_single_video
from dataset.video_transforms import Resize_Preprocess, ToTensorVideo
import torchvision.transforms as T
import wandb


from mpc.mpc import CEM_MPC
from mpc.discretempc import DiscreteMPC

# from language_table.language_table.environments import blocks
# from language_table.language_table.environments import language_table
# from language_table.language_table.environments.rewards import block2block
# from language_table.language_table.environments.rewards import block1_to_corner
import matplotlib.pyplot as plt

# faulthandler.enable()
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import random
from env_wrapper_base import EnvWrapperBase
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from dino_reward_model import DINO_Reward_Model

# from moviepy.editor import VideoFileClip
# IRASim default settings
DEFAULT_TIME_STEPS = 16


def create_arrow_image(direction="w", size=50, color=(255, 0, 0)):
    """
    Create an arrow image pointing in the specified direction.

    Parameters:
    - direction: The direction of the arrow ('up', 'down', 'left', 'right')
    - size: The length of the arrow in pixels
    - color: The color of the arrow (R, G, B)

    Returns:
    - A numpy array representing the arrow image.
    """
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image = imageio.imread("./sample/arrow.jpg")
    if direction == "s":
        image = np.flipud(image)
    elif direction == "a":
        image = np.rot90(image)
    elif direction == "d":
        image = np.rot90(image, -1)
    return image


def add_arrows_to_video(video_np, save_path, actions):
    """
    Add direction arrows to the video based on the actions list and save the modified video.

    Parameters:
    - video_np: A numpy array representing the video (frames, height, width, channels).
    - save_path: Path to save the modified video.
    - actions: A list of actions ('w', 's', 'a', 'd') for each frame.
    """
    if len(actions) != video_np.shape[0]:
        raise ValueError(
            "The length of the actions list must match the number of video frames."
        )

    writer = get_writer(save_path, fps=4)
    for frame, action in zip(video_np, actions):
        if action != " ":
            arrow_img = create_arrow_image(direction=action)
            position = (200, 240)
            for i in range(arrow_img.shape[0]):
                for j in range(arrow_img.shape[1]):
                    if np.any(arrow_img[i, j] != 0):
                        frame[position[0] + i, position[1] + j] = arrow_img[i, j]
        writer.append_data(frame)
    writer.close()


def read_actions_from_keyboard():
    valid_actions = ["w", "a", "s", "d", " "]
    actions = []

    while len(actions) < 15:
        input_actions = input(
            f"Please enter actions (remaining {15 - len(actions)}): "
        ).lower()
        for action in input_actions:
            if action in valid_actions and len(actions) < 15:
                actions.append(action)

        if len(actions) < 15:
            print(
                f"Not enough actions. Please enter the remaining {15 - len(actions)} actions."
            )
    return actions


def reshape(x):
    # 调整张量形状为 (28, 3, 360, 640) 以便进行插值
    x = x.permute(0, 3, 1, 2)  # 变换为 (batch, channels, height, width)

    # 对空间维度 (360, 640) 插值到 (288, 512)
    x_resized = F.interpolate(x, size=(288, 512), mode="bilinear", align_corners=False)

    # 恢复张量形状为 (28, 288, 512, 3)
    x_resized = x_resized.permute(0, 2, 3, 1)
    return x_resized


def predit_video(args, action, image, device, vae, model, game_dir):
    val_preprocess = T.Compose(
        [
            ToTensorVideo(),
            Resize_Preprocess(tuple(args.video_size)),  # 288 512
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )

    action_scaler = [20.0, 20.0]
    action_scaler = np.array(action_scaler)
    action = action * action_scaler
    actions = action
    actions = torch.from_numpy(np.array(actions))  # [:15]
    seg_action = actions
    seg_action = seg_action.unsqueeze(0)

    # plt.imshow(image)
    # plt.axis('off')
    # plt.savefig('text_frame_unworkable.png')

    video_tensor = []
    frame_tensor = image  # torch.tensor(image)/255.0

    # print(f"frame_tensor.shape0{frame_tensor.shape}")

    # frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)  # Shape: (1, 3, 180, 320)

    # Resize the tensor to (360, 640)
    print(f"frame_tensor_shape0{frame_tensor.shape}")
    resized_tensor = F.interpolate(
        frame_tensor, size=(360, 640), mode="bilinear", align_corners=False
    )

    # Remove the batch dimension and rearrange back to original dimensions
    resized_tensor = resized_tensor.squeeze(0).permute(1, 2, 0)  # Shape: (360, 640, 3)
    # print(f"resized_tensor_shape0{resized_tensor.shape}")

    resized_tensor = resized_tensor.clip(0, 1)  # Ensure values are in the range [0, 1]

    # Step 3: Save using matplotlib
    # plt.imshow(resized_tensor.numpy())
    # plt.axis('off')  # Hide axes
    # plt.savefig("test_resized_image.png")

    # video_tensor.append(frame_tensor)
    # print(f"resized_tensor{resized_tensor}")
    # print(resized_tensor.dtype)
    fram_tensor = (resized_tensor * 255).to(torch.uint8)
    # print(f"fram_tensor{fram_tensor}")
    # print(fram_tensor.dtype)
    video_tensor.append(fram_tensor)
    video_tensor = torch.stack(video_tensor)

    os.makedirs(game_dir, exist_ok=True)
    print(f"Game Dir {game_dir} !")

    # image_tensor = T.Resize((288, 512))(image_tensor)  # Example resizing
    # image_tensor = image_tensor.to(device)
    print(f"video_tensor_shape0{video_tensor.shape}")
    print(f"video_tensor_dtype0{video_tensor.dtype}")
    # Encode using VAE
    with torch.inference_mode():
        frames = video_tensor.permute(0, 3, 1, 2).cuda()
        print(f"video_tensor_shape2{video_tensor.shape}")
        frames = val_preprocess(frames)
        print(f"video_tensor_shape2{video_tensor.shape}")
        frames = (
            vae.encode(frames).latent_dist.sample().mul_(vae.config.scaling_factor)[:2]
        )
        print(f"video_tensor_shape3{video_tensor.shape}")
        latent_video = frames
        # encoded_image = vae.encode(image_tensor).latent_dist.sample().mul_(vae.config.scaling_factor)
        # print(f"iencoded_image.shape1{encoded_image.shape}")
    # Ensure batch_start_frame (start_image) is properly dimensioned
    # start_image = encoded_image.unsqueeze(0) # Add batch and time dimensions
    start_idx = 0
    start_image = latent_video[start_idx]
    video_tensor = video_tensor[start_idx:]
    seg_idx = 0

    video_tensor = video_tensor

    imageio.imwrite(os.path.join(game_dir, "first_image.png"), video_tensor[0].numpy())
    print("video_tensor shape is ", video_tensor.shape)

    video_tensor = reshape(video_tensor)
    seg_video_list = [video_tensor[0:1].numpy()]

    seg_action = actions
    start_image = start_image.unsqueeze(0).unsqueeze(0)
    seg_action = seg_action.unsqueeze(0)
    # Ensure actions have the correct shape

    print(f"start_image shape before call: {start_image.shape}")
    print(f"actions shape before call: {seg_action.shape}")
    # print(f"args:{args}")

    seg_video, seg_latents = generate_single_video(
        args, start_image, seg_action, device, vae, model
    )
    return seg_video, seg_latents, seg_video_list


def save_video(seg_video, seg_latents, seg_video_list, game_dir, seg_idx=0):
    seg_video = seg_video.squeeze()
    seg_latents = seg_latents.squeeze()
    start_image = seg_latents[-1].clone()

    t_videos = (
        ((seg_video / 2.0 + 0.5).clamp(0, 1) * 255)
        .detach()
        .to(dtype=torch.uint8)
        .cpu()
        .contiguous()
    )
    t_videos = rearrange(t_videos, "f c h w -> f h w c")
    t_videos = t_videos.numpy()
    print("t_videos.shape si ", t_videos.shape)
    seg_video_list.append(t_videos[1:])
    all_video = np.concatenate(seg_video_list, axis=0)
    output_video_path = os.path.join(game_dir, f"{random.randint(0,100)}-th.mp4")
    writer = get_writer(output_video_path, fps=4)
    for frame in all_video:
        writer.append_data(frame)
    writer.close()
    print(f"generate video: {output_video_path}")
    return all_video


class WorldModel:
    def __init__(self, args, device, vae, model, game_dir):
        self.args = args
        self.device = device
        self.vae = vae
        self.model = model
        self.game_dir = game_dir
        self.reward_model = DINO_Reward_Model()

    @torch.no_grad()
    def __call__(self, current_frame, action_sequences, goal_image, start_idx=None):
        # videos = []
        rewards = []
        for action in action_sequences:
            seg_video, seg_latents, seg_video_list = predit_video(
                args,
                action,
                current_frame,
                self.device,
                self.vae,
                self.model,
                self.game_dir,
            )
            prdicted_video_tensor = save_video(
                seg_video, seg_latents, seg_video_list, self.game_dir
            )
            prdicted_video_tensor = (
                torch.from_numpy(prdicted_video_tensor).permute(0, 3, 1, 2).unsqueeze(1)
            )
            reward = self.reward_model.calculate_reward(
                prdicted_video_tensor, goal_image
            )
            rewards.append(reward)
            # videos.append(-predict_reward(self.args, action, image, self.device, self.vae, self.model, self.game_dir))
        return rewards


if __name__ == "__main__":
    game_dir = "application/languagetable_game_short_action_sim_test"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/evaluation/languagetable/frame_ada.yaml",
    )
    parser.add_argument(
        "--wandb_project", type=str, default="IRASim_with_dino_reward_cem"
    )

    args = parser.parse_args()
    project_name = args.wandb_project
    wandb.init(
        project=args.wandb_project,
        entity="world_model_xh",
        config={"pop_size": 10, "elite_frac": 0.3, "horizon": 15, "max_iters": 2},
    )
    args = get_args(args)
    args.latent_size = [t // 8 for t in args.video_size]
    device = torch.device("cuda", 0)
    model = get_models(args)
    vae = AutoencoderKL.from_pretrained(args.vae_model_path, subfolder="vae").to(device)
    print("-----args", args)
    checkpoint = torch.load(
        args.evaluate_checkpoint, map_location=lambda storage, loc: storage
    )

    model_dict = model.state_dict()
    pretrained_dict = {}
    for k, v in checkpoint.items():
        if k in model_dict:
            pretrained_dict[k] = v
        else:
            print("Ignoring: {}".format(k))
    model_dict.update(pretrained_dict)

    model.load_state_dict(model_dict)
    model.to(device)
    model.eval()

    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    cfg_dict = {"environment": {"environment": {"initial_random_num_steps": 0}}}

    # Convert the dictionary to a DictConfig object
    cfg = OmegaConf.create(cfg_dict)

    env_real = EnvWrapperBase(cfg)
    env_real.env._render_text_in_image = False
    # print(env_real._image_size)

    output_dir = "./mpc_test_dino"
    os.makedirs(output_dir, exist_ok=True)

    # env_real._image_size = (444,640)
    [current_frame, goal_frame] = (
        env_real.generate_random_cur_frame_and_goal_image()
    )  # 1, C, H, W
    goal_image = goal_frame.clone()
    # Convert the tensor to a NumPy array and reshape to (180, 320, 3)
    image_np = goal_image.squeeze(0).permute(1, 2, 0).numpy()

    # Ensure the values are in the range [0, 255]
    image_np = (image_np * 255).astype(np.uint8)

    # Convert to image and save
    image = Image.fromarray(image_np)
    image_path = os.path.join(output_dir, "goal_image.png")
    image.save(image_path)

    # Upload the image to wandb
    wandb.log({"goal_image": wandb.Image(image_path)})

    print(f"Image saved at {image_path} and uploaded to wandb")

    frames = []
    cem_mpc = DiscreteMPC(
        action_dim=2, pop_size=10, elite_frac=0.3, horizon=15, max_iters=2
    )
    world_model = WorldModel(args, device, vae, model, game_dir)
    for step in tqdm(range(40)):
        print(
            "---- video_tensor shape is ", current_frame.shape
        )  # ---- video_tensor shape is  torch.Size([1, 3, 180, 320])
        # video_array = current_frame.squeeze().permute(1, 2, 0).numpy()  # H, W, C ()
        frames.append(current_frame)
        action, model_reward = cem_mpc.optimize(
            current_frame, goal_image=goal_frame, worldmodel=world_model, start_idx=step
        )  # op
        # world_model_reward = world_model()
        print(action)
        wandb.log({"action": action})
        wandb.log({"dino_model_reward": model_reward})
        for i in range(3):
            current_frame = env_real.step_env(action=action[i])
            frames.append(current_frame)

    video_path = os.path.join(output_dir, f"irasim_dino_{project_name}.mp4")
    # print('---- frames shape is ', frames)
    # Convert the tensor to a NumPy array and reshape to (10, 180, 320, 3)
    # frames_np = frames.squeeze(1).permute(0, 2, 3, 1).numpy()

    # Convert the list of tensors to a single tensor
    frames_tensor = torch.stack(frames)

    # Convert the tensor to a NumPy array and reshape to (10, 180, 320, 3)
    frames_np = frames_tensor.squeeze(1).permute(0, 2, 3, 1).numpy()

    # Ensure the values are in the range [0, 255]
    frames_np = (frames_np * 255).astype(np.uint8)

    # Ensure the values are in the range [0, 255]
    # frames_np = (frames_np * 255).astype(np.uint8)
    imageio.mimwrite(video_path, frames_np, fps=2)
    print(f"Video saved at {video_path}")
    wandb.log({"video": wandb.Video(video_path, fps=2, format="mp4")})

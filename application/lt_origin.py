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
import matplotlib.pyplot as plt

def create_arrow_image(direction='w', size=50, color=(255, 0, 0)):
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
    image = imageio.imread('./sample/arrow.jpg')
    if direction == 's':
        image = np.flipud(image)
    elif direction == 'a':
        image = np.rot90(image)
    elif direction == 'd':
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
        raise ValueError("The length of the actions list must match the number of video frames.")

    writer = get_writer(save_path, fps=4)
    for frame, action in zip(video_np, actions):
        if action != ' ':
            arrow_img = create_arrow_image(direction=action)
            position = (200, 240)
            for i in range(arrow_img.shape[0]):
                for j in range(arrow_img.shape[1]):
                    if np.any(arrow_img[i, j] != 0):
                        frame[position[0]+i, position[1]+j] = arrow_img[i, j]
        writer.append_data(frame)
    writer.close()

def read_actions_from_keyboard():
    valid_actions = ['w', 'a', 's', 'd', ' ']
    actions = []

    while len(actions) < 15:
        input_actions = input(f"Please enter actions (remaining {15 - len(actions)}): ").lower()
        for action in input_actions:
            if action in valid_actions and len(actions) < 15:
                actions.append(action)

        if len(actions) < 15:
            print(f"Not enough actions. Please enter the remaining {15 - len(actions)} actions.")
    return actions

def reshape(x):
    # 调整张量形状为 (28, 3, 360, 640) 以便进行插值
    x = x.permute(0, 3, 1, 2)  # 变换为 (batch, channels, height, width)

    # 对空间维度 (360, 640) 插值到 (288, 512)
    x_resized = F.interpolate(x, size=(288, 512), mode='bilinear', align_corners=False)

    # 恢复张量形状为 (28, 288, 512, 3)
    x_resized = x_resized.permute(0, 2, 3, 1)
    return x_resized

def main(device, model,  vae,  args, video_id=0000):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    val_preprocess = T.Compose([
        ToTensorVideo(),
        Resize_Preprocess(tuple(args.video_size)), # 288 512
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    left_scale,right_scale = 0.5,-0.5
    up_scale, down_scale = 0.5, -0.5


    video_path = f'/home/lr-2002/code/IRASim/val/{video_id}/rgb.mp4'
    # video_path = f"/home/lr-2002/code/IRASim/robotdata/opensource_robotdata/languagetable/evaluation_videos/val_sample_videos/{video_id}_0_0.mp4"
    ann_file =  f"/home/lr-2002/code/IRASim/robotdata/opensource_robotdata/languagetable/annotation/val/000016.json"
    with open(ann_file, "rb") as f:
        ann = json.load(f)
        print('---- annotation is ', ann)
    try:
        video_reader = imageio.get_reader(video_path)
    except: 
        return 
    video_tensor = []
    for idx, frame in enumerate(video_reader):
        if idx>0:
            break
        plt.imshow(frame)
        plt.axis('off')
        plt.savefig('text_frame_workable.png')
        frame_tensor = torch.tensor(frame)
        print(f"frame_tensor{frame_tensor}")
        print(frame_tensor.dtype)
        video_tensor.append(frame_tensor)
        print(f"frame_tensor_shape0{frame_tensor.shape}")
    video_reader.close()
    video_tensor = torch.stack(video_tensor)

    game_dir = 'application/languagetable_game_short_action_sim'
    os.makedirs(game_dir,exist_ok=True)
    print(f'Game Dir {game_dir} !')
    
    print(f"video_tensor_shape0{video_tensor.shape}")
    print(f"video_tensor_dtype0{video_tensor.dtype}")
    with torch.inference_mode():
        frames = video_tensor.permute(0, 3, 1, 2).cuda()
        print(f"video_tensor_shape2{video_tensor.shape}")
        frames = val_preprocess(frames)
        print(f"video_tensor_shape2{video_tensor.shape}")
        frames = vae.encode(frames).latent_dist.sample().mul_(vae.config.scaling_factor)[:2]
        print(f"video_tensor_shape3{video_tensor.shape}")
        latent_video = frames
    start_idx = 0
    start_image = latent_video[start_idx]
    video_tensor = video_tensor[start_idx:]
    seg_idx = 0

    video_tensor = video_tensor
   
    imageio.imwrite(os.path.join(game_dir,'first_image.png'), video_tensor[0].numpy())
    print('video_tensor shape is ' , video_tensor.shape)
    video_tensor = reshape(video_tensor)
    seg_video_list = [video_tensor[0:1].numpy()] # TODO
    
    action= None 
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
            env_actions = read_actions_from_keyboard()
            actions = []
            for action in env_actions:
                if action == 'd':
                    actions.append([0,up_scale])
                elif action == 'a':
                    actions.append([0,down_scale])
                elif action == 's':
                    actions.append([left_scale,0])
                elif action == 'w':
                    actions.append([right_scale,0])
                else:
                    actions.append([0,0])
        actions = torch.from_numpy(np.array(actions)[:15])

        seg_action = actions
        start_image = start_image.unsqueeze(0).unsqueeze(0)
        seg_action = seg_action.unsqueeze(0)
        
        print(f"start_image shape before call: {start_image.shape}")
        print(f"actions shape before call: {seg_action.shape}")
        print(f"args:{args}")
        
        seg_video, seg_latents = generate_single_video(args, start_image , seg_action, device, vae, model)
        
        seg_video = seg_video.squeeze()
        seg_latents = seg_latents.squeeze()
        start_image = seg_latents[-1].clone()

        t_videos = ((seg_video / 2.0 + 0.5).clamp(0, 1) * 255).detach().to(dtype=torch.uint8).cpu().contiguous()
        t_videos = rearrange(t_videos, 'f c h w -> f h w c')
        t_videos = t_videos.numpy()
        print('t_videos.shape si ', t_videos.shape)
        seg_video_list.append(t_videos[1:])
        all_video = np.concatenate(seg_video_list,axis=0)
        output_video_path = os.path.join(game_dir,f'{video_id}_{seg_idx}-th.mp4')
        writer = get_writer(output_video_path, fps=4)
        for frame in all_video:
            writer.append_data(frame)
        writer.close()
        print(f'generate video: {output_video_path}')
        seg_idx += 1
        del actions, seg_video, seg_latents
    exclude_list = [device, model, ema, vae]
    for name, obj in globals().items():
        if torch.is_tensor(obj) and obj.is_cuda and obj not in exclude_list:
            print('the obje is ', name)
 
    del frames, start_image, latent_video

    del action
    import gc 
    gc.collect()
    torch.cuda.empty_cache()
    for name, obj in globals().items():
        if torch.is_tensor(obj) and obj.is_cuda and obj not in exclude_list:
            print('after delete the obje is ', name)
 
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
    print('-----args', args)
    checkpoint = torch.load(args.evaluate_checkpoint, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:
        print('Using ema ckpt!')
        checkpoint = checkpoint["ema"]

    model_dict = model.state_dict()
    pretrained_dict = {}
    for k, v in checkpoint.items():
        if k in model_dict:
            pretrained_dict[k] = v
        else:
            print('Ignoring: {}'.format(k))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.to(device)
    model.eval()

   # train_dataset,val_dataset = get_dataset(args)
    # print(val_dataset.ann_files)
    from tqdm import tqdm
    for video_id in tqdm(os.listdir('./val/')):
        main(device, model, vae, args, video_id)
    # with open('./no_hide/filtered_file.pkl', 'rb') as f :
    #     import pickle as pkl 
    #     video_list = pkl.load(f)
    #     print(video_list)
    # for video_id in tqdm(video_list):
    #     main(device, model, vae,  args, str(video_id))

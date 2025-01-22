
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
from mpc.discretempc import DiscreteMPC
from language_table.language_table.environments import blocks
from language_table.language_table.environments import language_table
from language_table.language_table.environments.rewards import block2block
from language_table.language_table.environments.rewards import block1_to_corner
import matplotlib.pyplot as plt
#faulthandler.enable()
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import wandb
import random
from env_wrapper_base import EnvWrapperBase
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from scipy.ndimage import binary_erosion
#from moviepy.editor import VideoFileClip
# IRASim default settings
DEFAULT_TIME_STEPS = 16

def reshape(x):
    # 调整张量形状为 (28, 3, 360, 640) 以便进行插值
    x = x.permute(0, 3, 1, 2)  # 变换为 (batch, channels, height, width)

    # 对空间维度 (360, 640) 插值到 (288, 512)
    x_resized = F.interpolate(x, size=(288, 512), mode='bilinear', align_corners=False)

    # 恢复张量形状为 (28, 288, 512, 3)
    x_resized = x_resized.permute(0, 2, 3, 1)
    return x_resized

def predit_video(args,action,image,device, vae, model,game_dir):
    val_preprocess = T.Compose([
        ToTensorVideo(),
        Resize_Preprocess(tuple(args.video_size)), # 288 512
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    
    action_scaler = [20.0, 20.0] 
    action_scaler = np.array(action_scaler)
    action = action * action_scaler
    actions = action
    actions = torch.from_numpy(np.array(actions))  #[:15]
    seg_action = actions
    seg_action = seg_action.unsqueeze(0)
    
    plt.imshow(image)
    plt.axis('off')
    plt.savefig('text_frame_unworkable.png')

    video_tensor = []
    frame_tensor = torch.tensor(image)/255.0
    
    print(f"frame_tensor.shape0{frame_tensor.shape}")
    
    frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)  # Shape: (1, 3, 180, 320)

    # Resize the tensor to (360, 640)
    resized_tensor = F.interpolate(frame_tensor, size=(360, 640), mode='bilinear', align_corners=False)

    # Remove the batch dimension and rearrange back to original dimensions
    resized_tensor = resized_tensor.squeeze(0).permute(1, 2, 0)  # Shape: (360, 640, 3)
    print(f"resized_tensor_shape0{resized_tensor.shape}")
    
    
    resized_tensor = resized_tensor.clip(0, 1)  # Ensure values are in the range [0, 1]

    # Step 3: Save using matplotlib
    plt.imshow(resized_tensor.numpy())
    plt.axis('off')  # Hide axes
    plt.savefig("test_resized_image.png")

    #video_tensor.append(frame_tensor)
    print(f"resized_tensor{resized_tensor}")
    print(resized_tensor.dtype)
    fram_tensor = (resized_tensor * 255).to(torch.uint8)
    print(f"fram_tensor{fram_tensor}")
    print(fram_tensor.dtype)
    video_tensor.append(fram_tensor)
    video_tensor = torch.stack(video_tensor)
    
    os.makedirs(game_dir,exist_ok=True)
    print(f'Game Dir {game_dir} !')
    
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
        frames = vae.encode(frames).latent_dist.sample().mul_(vae.config.scaling_factor)[:2]
        print(f"video_tensor_shape3{video_tensor.shape}")
        latent_video = frames
        # encoded_image = vae.encode(image_tensor).latent_dist.sample().mul_(vae.config.scaling_factor)
        # print(f"iencoded_image.shape1{encoded_image.shape}")
    # Ensure batch_start_frame (start_image) is properly dimensioned
    #start_image = encoded_image.unsqueeze(0) # Add batch and time dimensions
    start_idx = 0
    start_image = latent_video[start_idx]
    video_tensor = video_tensor[start_idx:]
    seg_idx = 0

    video_tensor = video_tensor
    
    imageio.imwrite(os.path.join(game_dir,'first_image.png'), video_tensor[0].numpy())
    print('video_tensor shape is ' , video_tensor.shape)

    video_tensor = reshape(video_tensor)
    seg_video_list = [video_tensor[0:1].numpy()]
    
    seg_action = actions
    start_image = start_image.unsqueeze(0).unsqueeze(0)
    seg_action = seg_action.unsqueeze(0)
    # Ensure actions have the correct shape

    
    print(f"start_image shape before call: {start_image.shape}")
    print(f"actions shape before call: {seg_action.shape}")
    print(f"args:{args}")
        
    seg_video, seg_latents = generate_single_video(args, start_image , seg_action, device, vae, model)
    return seg_video, seg_latents , seg_video_list

def save_video(seg_video, seg_latents,seg_video_list,game_dir,seg_idx=0):
    seg_video = seg_video.squeeze()
    seg_latents = seg_latents.squeeze()
    start_image = seg_latents[-1].clone()

    t_videos = ((seg_video / 2.0 + 0.5).clamp(0, 1) * 255).detach().to(dtype=torch.uint8).cpu().contiguous()
    t_videos = rearrange(t_videos, 'f c h w -> f h w c')
    t_videos = t_videos.numpy()
    print('t_videos.shape si ', t_videos.shape)
    seg_video_list.append(t_videos[1:])
    all_video = np.concatenate(seg_video_list,axis=0)
    output_video_path = os.path.join(game_dir,f'{random.randint(0,100)}-th.mp4')
    writer = get_writer(output_video_path, fps=4)
    for frame in all_video:
        writer.append_data(frame)
    writer.close()
    print(f'generate video: {output_video_path}')
    return all_video[-1]
    
def compute_distance_to_corner(image):
    # Define the color range for green in RGB
    lower_green = np.array([0, 100, 0])
    upper_green = np.array([100, 255, 100])
    
    
    # Create a mask for green pixels
    mask = np.all((image >= lower_green) & (image <= upper_green), axis=-1)
    
    structure = np.ones((3, 3), dtype=bool)
    mask = binary_erosion(mask, structure=structure)
    
    path="./application/languagetable_game_short_action_mask_test"
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    imageio.imwrite(f'{path}/mask{random.randint(0,100)}.png', (mask * 255).astype(np.uint8))
    
    # Get the coordinates of green pixels
    green_coords = np.argwhere(mask)
    
    center_point = np.mean(green_coords, axis=0)
    
    # Lower-left corner coordinates
    corner = np.array([image.shape[0] - 1, 0])
    
    # Compute the Euclidean distance from each green pixel to the lower-left corner
    distances = np.linalg.norm(center_point - corner) #, axis=1
    
    # Return the minimum distance
    return distances #np.min(distances) if distances.size > 0 else None

def predict_reward(args, action, image, device, vae, model, game_dir):
    seg_video, seg_latents, seg_video_list = predit_video(args, action, image, device, vae, model, game_dir)
    last_frame = save_video(seg_video, seg_latents, seg_video_list, game_dir)
    dist = compute_distance_to_corner(last_frame)
    return dist

class FakeWorldModel:
    def __init__(self,args,device, vae, model, game_dir):
        self.args = args
        self.device = device
        self.vae = vae
        self.model = model
        self.game_dir = game_dir

    def __call__(self, image, action_sequences):
        rewards = []
        for action in action_sequences:
            rewards.append(-predict_reward(self.args, action, image, self.device, self.vae, self.model, self.game_dir))
        return rewards
    
if __name__ == "__main__":
    game_dir = 'application/languagetable_game_short_action_sim_test'
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/evaluation/languagetable/frame_ada.yaml")
    parser.add_argument("--wandb_project", type=str, required=True, help="WandB project name")

    args = parser.parse_args()
    
    #wandb.init(project=args.wandb_project)
    
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
    
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
    # ann_file =  f"/home/lr-2002/code/IRASim/robotdata/opensource_robotdata/languagetable/annotation/val/000016.json"
    # with open(ann_file, "rb") as f:
    #     ann = json.load(f)
    #     print('---- annotation is ', ann)
    rnd_seed = 3
#     random.seed(rnd_seed)    
#     env_real = language_table.LanguageTable(
#       block_mode=blocks.LanguageTableBlockVariants.BLOCK_4,
#       reward_factory=block1_to_corner.Block1ToCornerLocationReward,
#       control_frequency=10.0,
#       seed=rnd_seed 
#   ) 
#     env_real.reset()
#     env_real._render_text_in_image = False
    #print(env_real._image_size)
    #env_real._image_size = (444,640)
    #env_real.reset()
    
    # action = ann['actions'][:15]
    # print(action)
    
    horizon = 15
    episodes = 7
    
    TEST_IMAGE_SIZE = [288, 512]  # Quarter of the original size
    TEST_BATCH_SIZE = 1
    # wandb.init(project=args.wandb_project, entity='world_model_xh', config={
    #     "episodes": episodes,
    #     "horizon": horizon,
    #     "test_image_size": TEST_IMAGE_SIZE,
    #     "test_batch_size": TEST_BATCH_SIZE,
    # })
    
    output_dir = "./mpc_test_videos"
    os.makedirs(output_dir, exist_ok=True)

    cem_mpc = DiscreteMPC(action_dim=2, pop_size=10, elite_frac=0.2,horizon=horizon,max_iters=3)
    fake_world_model = FakeWorldModel(args, device, vae, model, game_dir)
    
    seedlist=[3,0,1,2,6,7,8]
    
    cfg_dict = {'environment': {'environment': {'initial_random_num_steps': 0}}}

    # Convert the dictionary to a DictConfig object
    cfg = OmegaConf.create(cfg_dict)
    
    for ep in range(episodes):
        rnd_seed = seedlist[ep]
        random.seed(rnd_seed)
        # env_real = language_table.LanguageTable(
        # block_mode=blocks.LanguageTableBlockVariants.BLOCK_4,
        # reward_factory=block1_to_corner.Block1ToCornerLocationReward,
        # control_frequency=10.0,
        # seed=rnd_seed) 
        # env_real._render_text_in_image = False
        #print(env_real._image_size)
        #env_real._image_size = (444,640)
        random.seed(rnd_seed)
        # env_real.reset()
        env_real = EnvWrapperBase(cfg)
        env_real.env._render_text_in_image = False
        total_reward = 0
        done = False
        frames = []  # To store video frames
        for step in range(30):
            #env_real.render()# no need to render 
            #rendered_image = env_real.render()
            frame = env_real.render()#mode='rgb_array'
            frames.append(frame)
            action ,world_model_reward = cem_mpc.optimize(frame,worldmodel = fake_world_model) # op 
            #world_model_reward = world_model()
            print(action)
            observation, reward, done, ___ = env_real.step(action)
            #print(reward)
            total_reward += reward
            wandb.log({
                "episode": ep + 1,
                "step": step + 1,
                "world_model_rewards": world_model_reward,
                "lt_reward": reward,
                "frame": wandb.Image(frame)
            })
            if done:
                break
        #video = np.stack(frames, axis=0)
        video_path = os.path.join(output_dir, f"episode_{ep + 1}.mp4")
        clip = ImageSequenceClip(frames, fps=2)
        clip.write_videofile(video_path, codec="libx264", audio=False)
        wandb.log({
            "episode": ep + 1,
            "lt_total_reward": total_reward,
            "video": wandb.Video(video_path, fps=2, format="mp4")
        })
    
        print(f"Episode {ep + 1}, Total Reward: {total_reward}")
        
    

    
     
    
  
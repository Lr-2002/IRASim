import os
import sys
import torch
import numpy as np
import wandb
import hydra

from models.world_model import WorldModel
from models.test_model import TestModel
from models.model_inference import get_args
import torch.nn.functional as F
import faulthandler

from mpc.mpc import CEM_MPC
from language_table.language_table.environments import blocks
from language_table.language_table.environments import language_table
from language_table.language_table.environments.rewards import block2block
from language_table.language_table.environments.rewards import block1_to_corner

faulthandler.enable()
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
#from moviepy.editor import VideoFileClip
# IRASim default settings
DEFAULT_TIME_STEPS = 16

# SAM2 settings
SAM2_CHECKPOINT = "./online_processor/checkpoints/sam2_hiera_large.pt"
SAM2_MODEL_CFG = "sam2_hiera_l.yaml"

# Test settings
TEST_IMAGE_SIZE = [288, 512]  # Quarter of the original size
TEST_BATCH_SIZE = 1

def clear_gpu_memory():
    """Clear GPU memory between tests"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Force garbage collection
        import gc
        gc.collect()
        
def main():
    # Set up arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/evaluation/languagetable/frame_ada.yaml")
    args = parser.parse_args([])
    args = get_args(args)
    args.latent_size = [t // 8 for t in TEST_IMAGE_SIZE]
    args.video_size = TEST_IMAGE_SIZE
    args.action_dim = 2

    print("Initializing world model...")
    # Initialize world model #TestModel
    world_model = TestModel(
        model_inference_args=args,
        sam2_checkpoint=SAM2_CHECKPOINT,
        model_cfg=SAM2_MODEL_CFG,
    )

    import cv2
    #image = cv2.imread('/ssd/lt/processed_dataset/lt_sim_seged/val/video_EiQKGXdvcmtlcl8wMDFfZXBfMTRfMDZfMDZfMjIQIxguIAMw6AkqIGY1MjU4NTI4N2ExNTc2Yjg1ZGZiYzI5OWI0OTgxMWZj/images/00000.jpg')

    # 生成随机actions
    num_frames = 15  # 生成15帧
    #actions = torch.randn(1, num_frames, args.action_dim)  # [1, 15, action_dim]

    env_real = language_table.LanguageTable(
      block_mode=blocks.LanguageTableBlockVariants.BLOCK_1,
      reward_factory=block1_to_corner.Block1ToCornerLocationReward,
      control_frequency=10.0,
  ) 
    horizon = 20

    

    episodes = 100

    # wandb.init(project="language-table-mpc-worldmodel", entity='world_model_xh', config={
    #     "episodes": episodes,
    #     "horizon": horizon,
    #     "test_image_size": TEST_IMAGE_SIZE,
    #     "test_batch_size": TEST_BATCH_SIZE,
    # })

    output_dir = "./mpc_test_videos"
    os.makedirs(output_dir, exist_ok=True)

    cem_mpc = CEM_MPC(action_dim=args.action_dim, pop_size=100, elite_frac=0.05,horizon=horizon)
            
    for ep in range(episodes):
        _ = env_real.reset()
        total_reward = 0
        done = False
        frames = []  # To store video frames
        for step in range(horizon):
            #env_real.render()# no need to render 
            #rendered_image = env_real.render()
            frame = env_real.render(mode='rgb_array')
            frames.append(frame)
            action ,world_model_reward = cem_mpc.optimize(frame,world_model) # op 
            #world_model_reward = world_model()
            #print(action)
            observation, reward, done, ___ = env_real.step(action)
            #print(reward)
            total_reward += reward
            # wandb.log({
            #     "episode": ep + 1,
            #     "step": step + 1,
            #     "world_model_rewards": world_model_reward,
            #     "lt_reward": reward
            # })
            if done:
                break
        #video = np.stack(frames, axis=0)
        video_path = os.path.join(output_dir, f"episode_{ep + 1}.mp4")
        clip = ImageSequenceClip(frames, fps=2)
        clip.write_videofile(video_path, codec="libx264", audio=False)
        # wandb.log({
        #     "episode": ep + 1,
        #     "lt_total_reward": total_reward,
        #     "video": wandb.Video(video_path, fps=2, format="mp4")
        # })
    
        print(f"Episode {ep + 1}, Total Reward: {total_reward}")

    
    #print(world_model(image, actions))
    # # Perform rollout
    # print("Performing rollout...")
    # videos, masks, rewards = world_model.rollout(image, actions)
    
    # # Check and print shapes
    # print("\nOutput shapes:")
    # print(f"Videos shape: {videos.shape}")
    # print(f"Masks shape: {masks.shape}")
    # print(f"Rewards shape: {rewards.shape}")
    
    # # Verify expected shapes
    # expected_shapes = {
    #     "batch_size": TEST_BATCH_SIZE,
    #     "num_frames": num_frames + 1,
    #     "image_size": TEST_IMAGE_SIZE
    # }
    
    # print("\nShape verification:")
    # print(f"Batch size: {'✓' if videos.shape[0] == expected_shapes['batch_size'] else '✗'} (expected {expected_shapes['batch_size']}, got {videos.shape[0]})")
    # print(f"Number of frames: {'✓' if videos.shape[1] == expected_shapes['num_frames'] else '✗'} (expected {expected_shapes['num_frames']}, got {videos.shape[1]})")
    # print(f"Spatial dimensions: {'✓' if list(masks.shape[-2:]) == expected_shapes['image_size'] else '✗'} (expected {expected_shapes['image_size']}, got {list(masks.shape[-2:])})")
    # print(f"Rewards shape: {'✓' if rewards.shape == (TEST_BATCH_SIZE, num_frames + 1) else '✗'} (expected {(TEST_BATCH_SIZE, num_frames + 1)}, got {rewards.shape})")

if __name__ == "__main__":
    main()

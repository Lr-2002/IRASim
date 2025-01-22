import os
import sys
import torch
import numpy as np
from models.world_model import WorldModel
from models.model_inference import get_args
import torch.nn.functional as F
import faulthandler

faulthandler.enable()
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
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/evaluation/languagetable/frame_ada.yaml",
    )
    args = parser.parse_args([])
    args = get_args(args)
    args.latent_size = [t // 8 for t in TEST_IMAGE_SIZE]
    args.video_size = TEST_IMAGE_SIZE
    args.action_dim = 2

    print("Initializing world model...")
    # Initialize world model
    world_model = WorldModel(
        model_inference_args=args,
        sam2_checkpoint=SAM2_CHECKPOINT,
        model_cfg=SAM2_MODEL_CFG,
    )

    import cv2

    image = cv2.imread(
        "/ssd/lt/processed_dataset/lt_sim_seged/val/video_EiQKGXdvcmtlcl8wMDFfZXBfMTRfMDZfMDZfMjIQIxguIAMw6AkqIGY1MjU4NTI4N2ExNTc2Yjg1ZGZiYzI5OWI0OTgxMWZj/images/00000.jpg"
    )

    # 生成随机actions
    num_frames = 15  # 生成15帧
    actions = torch.randn(1, num_frames, args.action_dim)  # [1, 15, action_dim]

    # print(world_model(image, actions))
    # Perform rollout
    print("Performing rollout...")
    videos, masks, rewards = world_model.rollout(image, actions)

    # Check and print shapes
    print("\nOutput shapes:")
    print(f"Videos shape: {videos.shape}")
    print(f"Masks shape: {masks.shape}")
    print(f"Rewards shape: {rewards.shape}")

    # Verify expected shapes
    expected_shapes = {
        "batch_size": TEST_BATCH_SIZE,
        "num_frames": num_frames + 1,
        "image_size": TEST_IMAGE_SIZE,
    }

    print("\nShape verification:")
    print(
        f"Batch size: {'✓' if videos.shape[0] == expected_shapes['batch_size'] else '✗'} (expected {expected_shapes['batch_size']}, got {videos.shape[0]})"
    )
    print(
        f"Number of frames: {'✓' if videos.shape[1] == expected_shapes['num_frames'] else '✗'} (expected {expected_shapes['num_frames']}, got {videos.shape[1]})"
    )
    print(
        f"Spatial dimensions: {'✓' if list(masks.shape[-2:]) == expected_shapes['image_size'] else '✗'} (expected {expected_shapes['image_size']}, got {list(masks.shape[-2:])})"
    )
    print(
        f"Rewards shape: {'✓' if rewards.shape == (TEST_BATCH_SIZE, num_frames + 1) else '✗'} (expected {(TEST_BATCH_SIZE, num_frames + 1)}, got {rewards.shape})"
    )


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardModel:
    def __init__(self):
        """Initialize the reward model"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def compute_reward(self, masks, images):
        """
        Compute reward based on masks and images
        Args:
            masks: Tensor of shape [B, T, N, H, W] where:
                  B is batch size
                  T is number of frames
                  N is number of objects
                  H, W are height and width
            images: Tensor of shape [B, T, H, W]
        Returns:
            rewards: Tensor of shape [B, T] containing reward values for each frame
        """
        # Convert inputs to torch tensors if they aren't already
        if not isinstance(masks, torch.Tensor):
            masks = torch.from_numpy(masks).float()
        if not isinstance(images, torch.Tensor):
            images = torch.from_numpy(images).float()
            
        masks = masks.to(self.device)
        images = images.to(self.device)
        
        print(f"Masks shape: {masks.shape}")
        print(f"Images shape: {images.shape}")
        
        B, T = masks.shape[:2]
        rewards = torch.zeros((B, T), device=self.device)
        
        # Basic reward computation:
        # 1. Coverage reward: encourage masks to cover a reasonable area
        # 2. Consistency reward: encourage temporal consistency between frames
        
        # Coverage reward
        mask_areas = masks.sum(dim=(-1, -2))  # [B, T, N]
        total_area = float(masks.shape[-1] * masks.shape[-2])
        coverage_ratio = mask_areas / total_area
        
        print(f"Coverage ratio shape: {coverage_ratio.shape}")
        
        # Penalize both too small and too large masks
        coverage_reward = -((coverage_ratio - 0.3).abs()).mean(dim=-1)  # [B, T]
        
        print(f"Coverage reward shape: {coverage_reward.shape}")
        
        # Consistency reward
        if T > 1:
            # Sum over object dimension first
            masks_sum = masks.sum(dim=2)  # [B, T, H, W]
            mask_diff = (masks_sum[:, 1:] - masks_sum[:, :-1]).abs().mean(dim=(-1, -2))  # [B, T-1]
            consistency_reward = -mask_diff
            # Pad to match temporal dimension
            consistency_reward = F.pad(consistency_reward, (0, 1), mode='replicate')
        else:
            consistency_reward = torch.zeros_like(coverage_reward)
            
        print(f"Consistency reward shape: {consistency_reward.shape}")
        
        # Combine rewards
        rewards = coverage_reward + consistency_reward
        
        return rewards

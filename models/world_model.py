import torch
import cv2 
import numpy as np
from .model_inference import ModelInference
from .reward_model import RewardModel
from online_processor.online_processor import OnlineProcessor

class WorldModel:
    def __init__(self, model_inference_args, sam2_checkpoint, model_cfg):
        """
        Initialize World Model with its component models
        Args:
            model_inference_args: Arguments for ModelInference
            sam2_checkpoint: Checkpoint path for SAM model
            model_cfg: Configuration for OnlineProcessor
        """
        # Initialize component models
        self.model_inference_args = model_inference_args
        # self.dynamic_model = ModelInference(model_inference_args) #! do not delete
        self.dynamic_model = None
        self.mask_model = OnlineProcessor(model_cfg, sam2_checkpoint)
        self.reward_model = None
        # self.reward_model = RewardModel()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def reset_mask_model(self, frame, text_prompt="object.", confidence_threshold=0.1):
        """Reset the mask model with a new frame"""
        return self.mask_model.reset(frame, text_prompt, confidence_threshold)
    
    def __call__(self, start_frame, actions, text_prompt="object."):
        # acion is b ,t ,2 
        # use b ,t ,1 to return reward noise 
        rewards = torch.randn(actions.shape[0], actions.shape[1], 1) # use -1 -- 1 
        rewards = rewards.clamp(-1, 1)
        # rewards = self.rollout(start_frame, actions, text_prompt)
        return rewards
    
    def rollout(self, start_frame, actions, text_prompt="object."):
        """
        Perform a rollout using the world model
        Args:
            start_frame: Starting frame with shape [B, C, H, W]
            actions: Actions with shape [B, T, C] where T is number of frames
            text_prompt: Text prompt for object detection
        Returns:
            videos: Generated video frames [B, T+1, C, H, W]
            masks: Segmentation masks [B, T+1, N, H, W]
            rewards: Computed rewards [B, T+1]
        """
        # Generate videos using dynamic model
        # 预处理图像：转换为tensor，permute通道，归一化到[-1, 1]

        
        if start_frame.shape[:2] != tuple(self.model_inference_args.video_size):
            start_frame = cv2.resize(start_frame, tuple(self.model_inference_args.video_size))
            start_frame = cv2.transpose(start_frame)
        success, (box, obj_id) = self.mask_model.reset(start_frame, text_prompt, is_rgb=False)
        
        
        start_frame = cv2.cvtColor(start_frame, cv2.COLOR_BGR2RGB)
        start_frame = torch.from_numpy(start_frame).float().permute(2, 0, 1) / 127.5 - 1
        
        # 添加batch维度
        start_frame = start_frame.unsqueeze(0)  # [1, C, H, W]

        videos = self.dynamic_model.forward(start_frame, actions)
        print(videos.shape)
        
        # Convert to numpy format expected by mask model (uint8, 0-255)
        videos_np = ((videos.cpu() / 2.0 + 0.5).clamp(0, 1) * 255).to(torch.uint8).numpy()
        B, T, C, H, W = videos.shape
        
        # Process each video through mask model
        all_masks = []
        
        for b in range(B):
            video_masks = []
            # Reset mask model with first frame (HWC format)
            first_frame = videos_np[b, 0].transpose(1, 2, 0)  # CHW -> HWC
            # success = self.mask_model.reset(first_frame, text_prompt)
            success, _ = self.mask_model.reset_with_bbox(start_frame, box, obj_id)
            if not success:
                print(f"Warning: Failed to initialize tracking for batch {b}")
                video_masks.append(torch.zeros((1, H, W)))
                continue
            
            # Process each frame
            for t in range(T):
                frame = videos_np[b, t].transpose(1, 2, 0)  # CHW -> HWC
                masks = self.mask_model.add_frame(frame)
                if isinstance(masks, np.ndarray):
                    masks = torch.from_numpy(masks).float()  # Convert to float32
                video_masks.append(masks.unsqueeze(0))  # Add time dimension
            
            video_masks = torch.cat(video_masks, dim=0)
            all_masks.append(video_masks)
        
        # Stack batch dimension and convert to float32
        all_masks = torch.stack(all_masks, dim=0).float()  # [B, T, N, H, W]
        
        # Convert videos to grayscale for reward computation
        gray_videos = videos_np.mean(axis=2) / 255.0  # Convert to float and normalize
        gray_videos = torch.from_numpy(gray_videos).float()
        
        # Compute rewards
        rewards = self.reward_model.compute_reward(all_masks, gray_videos)
        
        return rewards

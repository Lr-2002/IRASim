import os
import torch
import numpy as np
from torch.utils.data import Dataset
import decord
import json

class SimpleDataset(Dataset):
    def __init__(self, args, mode='train'):
        """
        Initialize dataset
        Args:
            args: Configuration arguments
            mode: Dataset mode ('train', 'val', 'test')
        """
        self.args = args
        self.mode = mode
        self.video_size = args.video_size
        self.num_frames = args.num_frames
        self.use_mp4 = getattr(args, 'use_mp4', True)  # Default to True for MP4 loading
        
        if self.use_mp4:
            # Load video data from MP4
            self.video_dir = os.path.join(args.video_path, 'videos', mode)
            
            # Get list of video directories
            self.video_dirs = []
            for episode_dir in os.listdir(self.video_dir):
                video_path = os.path.join(self.video_dir, episode_dir, 'rgb.mp4')
                if os.path.exists(video_path):
                    self.video_dirs.append(episode_dir)
            
            if len(self.video_dirs) == 0:
                raise ValueError(f"No valid video files found in {self.video_dir}")
                
            print(f"Found {len(self.video_dirs)} episodes in {mode} set")
        else:
            # Use random data generation mode
            self.dataset_size = 1000 if mode == 'train' else 100
            
    def __len__(self):
        """Return the size of dataset"""
        if self.use_mp4:
            return len(self.video_dirs)
        return self.dataset_size
    
    def load_video_frame(self, video_path, frame_idx):
        """Load a specific frame from video file"""
        vr = decord.VideoReader(video_path)
        frame = vr[frame_idx].asnumpy()
        # Convert to torch tensor and normalize to [-1, 1]
        frame = torch.from_numpy(frame).float().permute(2, 0, 1) / 127.5 - 1
        return frame
    
    def load_action_sequence(self, episode_dir, start_idx):
        """
        Load action sequence for the episode
        Args:
            episode_dir: Directory containing episode data
            start_idx: Starting frame index
        Returns:
            Action sequence tensor [N, C] where N is number of frames
        """
        # For now, generate random actions (2D for LanguageTable dataset)
        # Later, this should be replaced with actual action loading from annotation files
        return torch.randn(self.num_frames - 1, 2)  # Generate num_frames - 1 actions
    
    def __getitem__(self, idx):
        """
        Get a sample from dataset
        Returns:
            Dictionary containing:
            - start_frame: Starting frame tensor [C, H, W]
            - actions: Action sequence tensor [N, C] where N is number of frames
        """
        if self.use_mp4:
            episode_dir = self.video_dirs[idx]
            video_path = os.path.join(self.video_dir, episode_dir, 'rgb.mp4')
            
            # Load video
            vr = decord.VideoReader(video_path)
            total_frames = len(vr)
            
            # Randomly select start frame (ensure enough frames are available)
            max_start_idx = total_frames - self.num_frames
            start_idx = np.random.randint(0, max_start_idx) if max_start_idx > 0 else 0
            
            # Load start frame
            start_frame = self.load_video_frame(video_path, start_idx)
            
            # Load action sequence
            actions = self.load_action_sequence(episode_dir, start_idx)
            
            # Resize if needed
            if start_frame.shape[-2:] != tuple(self.video_size):
                start_frame = torch.nn.functional.interpolate(
                    start_frame.unsqueeze(0),
                    size=tuple(self.video_size),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
        else:
            # Generate random data
            start_frame = torch.randn(3, *self.video_size)  # [C, H, W]
            actions = torch.randn(self.num_frames, 2)
        
        return {
            'start_frame': start_frame,  # [C, H, W]
            'actions': actions,  # [N, 2]
        }

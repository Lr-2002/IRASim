import json
import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as T
from dataset.video_transforms import Resize_Preprocess, ToTensorVideo
import decord

class SimpleDataset(Dataset):
    def __init__(self, args, mode='train'):
        """
        Simple dataset that only loads frames and actions
        Args:
            args: Configuration arguments
            mode: 'train', 'val', or 'test'
        """
        super().__init__()
        self.args = args
        self.mode = mode
        
        # 设置数据集大小
        self.dataset_size = 1000 if mode == 'train' else 100
        
        # 设置图像和动作维度
        self.image_size = tuple(args.video_size)  # [h, w]
        self.seq_len = args.num_frames - 1
        self.action_dim = 2  # 根据实际action维度设置
        
        print(f'{self.dataset_size} samples in total')
        
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset
        Returns:
            dict containing:
                - video: tensor of shape [c, h, w]
                - action: tensor of shape [n, c] where n is number of frames
        """
        # 生成随机图像 [3, h, w]
        video = torch.randn(3, self.image_size[0], self.image_size[1])
        video = torch.clamp(video, -1, 1)  # 归一化到 [-1, 1]
        
        # 生成随机动作序列 [n, c]
        actions = torch.randn(self.seq_len, self.action_dim)
        actions = torch.clamp(actions, -1, 1)  # 归一化到 [-1, 1]
        
        return {
            'video': video,
            'action': actions
        }

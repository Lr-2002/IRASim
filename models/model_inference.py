import os
import sys
sys.path.append('/home/lr-2002/code/IRASim/')
from diffusers.models import AutoencoderKL
from sample.pipeline_trajectory2videogen import Trajectory2VideoGenPipeline
from diffusers.schedulers import DDPMScheduler, PNDMScheduler
from models import get_models
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
from dataset.simple_dataset import SimpleDataset
from util import get_args, update_paths
import torchvision.utils as vutils
import imageio
import numpy as np
from einops import rearrange

class ModelInference:
    def __init__(self, args):
        """
        Initialize the model inference class
        Args:
            config_path: Path to the configuration file
        """
        # Load configuration
        # data_config = OmegaConf.load("configs/base/data.yaml")
        # diffusion_config = OmegaConf.load("configs/base/diffusion.yaml")
        # config = OmegaConf.load(config_path)
        # config = OmegaConf.merge(data_config, config)
        # self.args = OmegaConf.merge(diffusion_config, config)
        self.args = args       
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(self.args.vae_model_path, subfolder="vae").to(self.device)
        
        # Load model
        self.model = get_models(self.args)
        if self.args.evaluate_checkpoint:
            state_dict = torch.load(self.args.evaluate_checkpoint, map_location='cpu')
            if 'ema' in state_dict:
                self.model.load_state_dict(state_dict['ema'])
            else:
                self.model.load_state_dict(state_dict['model'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Initialize scheduler based on sample_method
        if self.args.sample_method == 'PNDM':
            self.scheduler = PNDMScheduler.from_pretrained(
                self.args.scheduler_path, 
                beta_start=self.args.beta_start, 
                beta_end=self.args.beta_end, 
                beta_schedule=self.args.beta_schedule, 
                variance_type=self.args.variance_type
            )
        elif self.args.sample_method == 'DDPM':
            self.scheduler = DDPMScheduler.from_pretrained(
                self.args.scheduler_path, 
                beta_start=self.args.beta_start, 
                beta_end=self.args.beta_end, 
                beta_schedule=self.args.beta_schedule, 
                variance_type=self.args.variance_type
            )
        else:
            raise ValueError(f"Unknown sample_method: {self.args.sample_method}")
        
        # Initialize pipeline
        self.pipeline = Trajectory2VideoGenPipeline(
            vae=self.vae,
            scheduler=self.scheduler,
            transformer=self.model
        )
    
    def forward(self, start_frame, actions):
        """
        Forward pass through the model
        Args:
            start_frame: Starting frame with shape [b, c, h, w]
            actions: Actions with shape [b, n, c] where n is number of frames
        Returns:
            Generated frame with shape [b, c, h, w]
        """
        with torch.no_grad():
            # Move to device
            start_frame = start_frame.to(self.device).to(torch.float32)
            actions = actions.to(self.device).to(torch.float32)
            
            # Encode start frame with VAE
            latent_dist = self.vae.encode(start_frame).latent_dist
            latent = latent_dist.sample().mul_(self.vae.config.scaling_factor)  # [b, c, h/8, w/8]
            
            # Add time dimension to latent: [b, c, h/8, w/8] -> [b, 1, c, h/8, w/8]
            latent = latent.unsqueeze(1)
            
            # Generate video latents
            videos, latents = self.pipeline(
                actions,
                mask_x=latent,
                video_length=actions.shape[1] + 1,  # +1 for start frame
                height=self.args.video_size[0],
                width=self.args.video_size[1],
                num_inference_steps=self.args.infer_num_sampling_steps,
                guidance_scale=self.args.guidance_scale,
                device=self.device,
                output_type='both'  # Return both videos and latents
            )
            
            # Check pipeline output
            if videos is None and latents is None:
                raise ValueError("Pipeline returned None for both videos and latents")
            
            # If we have latents, decode them
            if latents is not None:
                # Decode latents with VAE
                latents = 1 / self.vae.config.scaling_factor * latents
                videos = self.vae.decode(latents.flatten(0, 1)).sample
                videos = videos.reshape(latents.shape[0], latents.shape[1], *videos.shape[1:])
            
            if videos is None:
                raise ValueError("Failed to generate videos")
                
            return videos

    def save_video(self, video_tensor, save_path):
        """
        Save video tensor as mp4
        Args:
            video_tensor: Video tensor [T, C, H, W] or [B, T, C, H, W]
            save_path: Path to save the video
        """
        # Remove batch dimension if it exists
        if len(video_tensor.shape) == 5:
            video_tensor = video_tensor[0]  # [T, C, H, W]
        
        # Create directory if not exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Process video tensor
        video = ((video_tensor / 2.0 + 0.5).clamp(0, 1) * 255).detach().to(dtype=torch.uint8).cpu()
        video = rearrange(video, 't c h w -> t h w c').numpy()
        
        # Save video
        writer = imageio.get_writer(save_path, fps=4)
        for frame in video:
            writer.append_data(frame)
        writer.close()

    def save_frame(self, frame_tensor, save_path):
        """
        Save frame tensor as image
        Args:
            frame_tensor: Frame tensor [C, H, W] or [B, C, H, W]
            save_path: Path to save the frame
        """
        # Remove batch dimension if it exists
        if len(frame_tensor.shape) == 4:
            frame_tensor = frame_tensor[0]  # [C, H, W]
        
        # Create directory if not exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Process frame tensor
        frame = ((frame_tensor / 2.0 + 0.5).clamp(0, 1) * 255).detach().to(dtype=torch.uint8).cpu()
        frame = rearrange(frame, 'c h w -> h w c').numpy()
        
        # Save image
        imageio.imwrite(save_path, frame)

if __name__ == '__main__':
    import argparse
    import imageio.v3 as iio
    import torch
    import numpy as np
    import torch.nn.functional as F

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/evaluation/languagetable/frame_ada.yaml")

    args = parser.parse_args()
    args = get_args(args)
    args.latent_size = [t //8 for t in args.video_size]
    args.action_dim = 2
    # 初始化模型
    model = ModelInference(args)
    save_dir = './samples'
    os.makedirs(save_dir, exist_ok=True)

    # 读取视频第一帧
    video_path = './val/000027/rgb.mp4'
    video = iio.imread(video_path, index=0)  # 只读取第一帧
    
    # 预处理图像：转换为tensor，permute通道，归一化到[-1, 1]
    start_frame = torch.from_numpy(video).float().permute(2, 0, 1) / 127.5 - 1
    
    # Resize到指定大小
    if start_frame.shape[-2:] != tuple(args.video_size):
        start_frame = F.interpolate(
            start_frame.unsqueeze(0),
            size=tuple(args.video_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
    
    # 添加batch维度
    start_frame = start_frame.unsqueeze(0)  # [1, C, H, W]
    
    # 生成随机actions
    num_frames = 15  # 生成15帧
    actions = torch.randn(1, num_frames, args.action_dim)  # [1, 15, action_dim]
    
    # 打印输入shape
    print(f"Start frame shape: {start_frame.shape}")
    print(f"Actions shape: {actions.shape}")
    
    # 使用模型生成视频
    output_video = model.forward(start_frame, actions)  # [1, T, C, H, W]
    
    # 打印输出shape
    print(f"Output video shape: {output_video.shape}")
    
    # 保存视频
    model.save_video(
        output_video[0],  # 移除batch维度
        os.path.join(save_dir, 'generated_video.mp4')
    )
    
    # 保存起始帧
    model.save_frame(
        start_frame,
        os.path.join(save_dir, 'start_frame.png')
    )
            
    print(f"Results saved to {save_dir}")
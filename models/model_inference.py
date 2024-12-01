import torch
import numpy as np
import sys 
sys.path.append('/home/lr-2002/code/IRASim/')
from diffusers.models import AutoencoderKL
from sample.pipeline_trajectory2videogen import Trajectory2VideoGenPipeline
from diffusers.schedulers import DDPMScheduler, PNDMScheduler
from models import get_models
from omegaconf import OmegaConf
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
            # Convert numpy arrays to tensors if needed
            if isinstance(start_frame, np.ndarray):
                start_frame = torch.from_numpy(start_frame)
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions)
            
            # Move to device
            start_frame = start_frame.to(self.device).to(torch.float32)
            actions = actions.to(self.device).to(torch.float32)
            
            # Encode start frame with VAE
            latent_dist = self.vae.encode(start_frame).latent_dist
            latent = latent_dist.sample().mul_(self.vae.config.scaling_factor)  # [b, c, h/8, w/8]
            
            # Add time dimension to latent: [b, c, h/8, w/8] -> [b, 1, c, h/8, w/8]
            latent = latent.unsqueeze(1)
            
            # Generate video
            videos, _ = self.pipeline(
                actions,
                mask_x=latent,
                video_length=actions.shape[1] + 1,  # +1 for start frame
                height=self.args.video_size[0],
                width=self.args.video_size[1],
                num_inference_steps=self.args.infer_num_sampling_steps,
                guidance_scale=self.args.guidance_scale,
                device=self.device,
                return_dict=False,
                output_type='video'
            )
            
            # Return last frame
            return videos[:, -1]  # Shape: [b, c, h, w]

if __name__=='__main__':
    import argparse
    from util import update_paths, get_args
    from dataset.simple_dataset import SimpleDataset
    from torch.utils.data import DataLoader
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/evaluation/languagetable/frame_ada.yaml")

    args = parser.parse_args()
    args = get_args(args)
    args.latent_size = [t //8 for t in args.video_size]
    # update_paths(args)

    # 初始化模型
    model = ModelInference(args)

    # 初始化数据集
    dataset = SimpleDataset(args, mode='val')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    # 使用模型进行推理
    for batch in dataloader:
        start_frame = batch['video']  # [b, c, h, w]
        actions = batch['action']     # [b, n, c]
        
        # 前向传播
        output_frame = model.forward(start_frame, actions)  # [b, c, h, w]
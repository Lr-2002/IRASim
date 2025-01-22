import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models import AutoencoderKL, Transformer2DModel
from diffusers.schedulers import DDIMScheduler, DDPMScheduler
from typing import Tuple, Optional, Union, Dict
import numpy as np
from einops import rearrange, repeat
import logging
from pathlib import Path
from sample.pipeline_trajectory2videogen import Trajectory2VideoGenPipeline
from models import get_models
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WorldModelWrapper:
    """Wrapper class for video generation using the world model."""
    
    def __init__(
        self,
        vae_model_path: str,
        model_path: str,
        scheduler_path: str,
        device: str = "cuda",
        sample_method: str = "DDPM",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        variance_type: str = "fixed_small",
    ):
        """
        Initialize the world model wrapper.
        
        Args:
            vae_model_path: Path to the VAE model
            model_path: Path to the transformer model
            scheduler_path: Path to the scheduler config
            device: Device to run the model on
            sample_method: Sampling method (DDPM or PNDM)
            beta_start: Start value for beta schedule
            beta_end: End value for beta schedule
            beta_schedule: Type of beta schedule
            variance_type: Type of variance to use
        """
        self.device = device
        self.vae = AutoencoderKL.from_pretrained(vae_model_path, subfolder="vae").to(device)
        
        # Initialize scheduler
        if sample_method == "PNDM":
            self.scheduler = DDPMScheduler.from_pretrained(
                scheduler_path,
                beta_start=beta_start,
                beta_end=beta_end,
                beta_schedule=beta_schedule,
                variance_type=variance_type
            )
        else:  # Default to DDPM
            self.scheduler = DDPMScheduler.from_pretrained(
                scheduler_path,
                beta_start=beta_start,
                beta_end=beta_end,
                beta_schedule=beta_schedule,
                variance_type=variance_type
            )
        
        # Load transformer model
        checkpoint = torch.load(model_path, map_location=device)
        if not isinstance(checkpoint, dict):
            raise ValueError("Expected checkpoint to be a dictionary")
            
        # Get model config from the checkpoint and create args
        args = OmegaConf.create({
            'model': 'IRASim-XL/2',  # This should match your model type
            'latent_size': 36,  # 288/8 = 36 to match VAE output size
            'num_frames': 16,
            'learn_sigma': False,  # Set to False as per original config
            'extras': 3,
            'attention_mode': 'math',
            'dataset': 'languagetable',
            'video_size': [288, 512]  # Add video size from config
        })
        
        # Initialize the model using get_models
        self.model = get_models(args).to(device)
            
        # Load state dict
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'ema' in checkpoint:  # Try EMA model if available
            state_dict = checkpoint['ema']
        else:
            raise ValueError("No model state dict found in checkpoint")
            
        # Load state dict
        try:
            msg = self.model.load_state_dict(state_dict, strict=False)  # Use strict=False to allow partial loading
            logger.info(f"Model loading info: {msg}")
        except Exception as e:
            logger.warning(f"Failed to load state dict directly: {e}")
            # Try to match keys by removing 'module.' prefix
            fixed_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            msg = self.model.load_state_dict(fixed_state_dict, strict=False)
            logger.info(f"Model loading info after fixing: {msg}")
        
        # Create pipeline components without moving to device yet
        self.pipeline = Trajectory2VideoGenPipeline(
            vae=self.vae,
            scheduler=self.scheduler,
            transformer=self.model,
        )
        
        # Manually move pipeline components to device
        self.pipeline.vae = self.pipeline.vae.to(device)
        self.pipeline.scheduler = self.pipeline.scheduler
        self.pipeline.transformer = self.pipeline.transformer.to(device)
        
    def generate_video(
        self,
        start_image: torch.Tensor,
        actions: torch.Tensor,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        mask_frame_num: int = 1,
        height: Optional[int] = 288,
        width: Optional[int] = 512,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a video sequence given an initial frame and actions.
        
        Args:
            start_image: Initial frame tensor
            actions: Sequence of actions
            num_inference_steps: Number of denoising steps
            guidance_scale: Scale for classifier-free guidance
            mask_frame_num: Number of frames to mask
            height: Height of the output video frames
            width: Width of the output video frames
            
        Returns:
            Tuple of (generated video frames, latent representations)
        """
        # Prepare mask_x from start_image
        mask_x = start_image.repeat(1, mask_frame_num, 1, 1, 1)
        
        # Calculate video length from actions tensor
        # Actions shape should be [batch_size, num_frames, action_dim]
        video_length = actions.size(1) + mask_frame_num
        
        # Call pipeline with corrected parameters
        return self.pipeline(
            action=actions,
            mask_x=mask_x,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            video_length=video_length,
            device=self.device
        )

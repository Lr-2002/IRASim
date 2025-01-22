import cv2 
import numpy as np
from .model_inference import ModelInference
from .reward_model import RewardModel
from online_processor.online_processor import OnlineProcessor
from .world_model import WorldModel
import torch

class TestModel(WorldModel):
    def __init__(self, model_inference_args, sam2_checkpoint, model_cfg):
        super().__init__(model_inference_args, sam2_checkpoint, model_cfg)
        self.dynamic_model = ModelInference(model_inference_args)
    
    def __call__(self, start_frame, actions, text_prompt="object."):
        # acion is b ,t ,2 
        # use b ,t ,1 to return reward noise 
        if isinstance(start_frame, np.ndarray):
            start_frame = torch.from_numpy(start_frame)
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)
        rewards = torch.randn(actions.shape[0], actions.shape[1], 1) # use -1 -- 1 
        rewards = rewards.clamp(-1, 1)
        # rewards = self.rollout(start_frame, actions, text_prompt)
        videos = self.dynamic_model.forward(start_frame, actions)
        print(videos.shape)
        return rewards
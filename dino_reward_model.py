import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import torch
import torch.nn.functional as F
import numpy as np


class DINOBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_version = "dinov2_vitb14"
        version = self.model_version
        self.version = version
        if "v2" in version:
            self.dino = torch.hub.load("facebookresearch/dinov2", version)
        else:
            self.dino = torch.hub.load("facebookresearch/dino:main", version)

    def forward(self, x):
        if "v2" in self.version:
            enc_out = self.dino.forward_features(x)["x_norm_patchtokens"]
        else:
            enc_out = self.dino.get_intermediate_layers(x, n=1)[0][
                :, 1:
            ]  # ge the last layer features
        return rearrange(
            enc_out, "b (h w ) c -> b c h w", h=int(np.sqrt(enc_out.shape[-2]))
        )


class DINO_Reward_Model:
    def __init__(self):
        self.dino_model = DINOBackbone()
        self.device = torch.device("cpu")

    def distance_to_goal_state(self, goal_image, cur_frame):
        """
        Compute the distance between the goal image and the current frame in DINO distance
        Might need to resize the images
        """
        # The output is fp32 as default, no need to change back
        goal_image_dino_feature = self.dino_model(
            self.resize_tensor(goal_image, 224, 224).to(
                dtype=torch.float, device=self.device
            )
        )
        cur_frame_dino_feature = self.dino_model(
            self.resize_tensor(cur_frame, 224, 224).to(
                dtype=torch.float, device=self.device
            )
        )

        return torch.norm(goal_image_dino_feature - cur_frame_dino_feature) ** 2

    def calculate_reward(self, predicted_video, goal_image):
        """
        Calculate the reward of each video as the minimum distance between the goal image and each frame in the predicted video
        """
        # each_frame_distance = [
        #     self.distance_to_goal_state(goal_image, predicted_video[i])
        #     for i in range(predicted_video.shape[0])
        # ]
        # return min(each_frame_distance)
        return -self.distance_to_goal_state(goal_image, predicted_video[-1])

    def resize_tensor(self, tensor, new_height, new_width):
        """
        调整张量的大小
        :param tensor: 输入张量，形状为 [1, 3, 180, 320]
        :param new_height: 新的高度
        :param new_width: 新的宽度
        :return: 调整大小后的张量
        """
        # 使用 interpolate 函数调整张量的大小
        flatten_tensor = tensor.flatten(0, -4)
        C = flatten_tensor.shape[1]
        resized_tensor = F.interpolate(
            flatten_tensor,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        )
        resized_tensor = resized_tensor.view(
            *tensor.shape[:-3], C, new_height, new_width
        )
        return resized_tensor


if __name__ == "__main__":
    dino_model = DINO_Reward_Model()
    print("dino model loaded")

    # write an test example for this
    goal_image = torch.randn(1, 3, 180, 320)
    predicted_video = torch.randn(10, 1, 3, 180, 320)
    reward = dino_model.calculate_reward(predicted_video, goal_image)
    print(reward)

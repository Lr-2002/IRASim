import os
import torch
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# 自定义数据集类
class ImageDataset(Dataset):
    def __init__(self, base_dataset_path):
        self.image_paths = []
        self.base_dataset_path = base_dataset_path

        # 获取所有视频 ID
        video_ids = [d for d in os.listdir(base_dataset_path) if os.path.isdir(os.path.join(base_dataset_path, d))]
        
        # 收集所有图像路径
        for video_id in video_ids:
            video_path = os.path.join(base_dataset_path, video_id, 'images/')
            if os.path.exists(video_path):
                for root, _, files in os.walk(video_path):
                    for file in files:
                        if file.endswith('.jpg'):
                            self.image_paths.append(os.path.join(root, file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        return transform(img)

# 基础数据集路径
base_dataset_path = './robotdata/opensource_robotdata/languagetable/mask_data/train'
batch_size = 1024  # 设置批处理大小

# 创建数据集和数据加载器
dataset = ImageDataset(base_dataset_path)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 初始化均值、方差和计数
mean = torch.zeros(3, device=device)
std = torch.zeros(3, device=device)
total_images = 0
# 判断是否有可用的 GPU

# 遍历数据加载器中的每个批次
for batch in tqdm(data_loader):
    batch = batch.to(device)
    # 计算当前批次的均值和标准差
    batch_mean = batch.mean(dim=[0, 2, 3])  # 每个通道的均值
    batch_std = batch.std(dim=[0, 2, 3])    # 每个通道的标准差

    # 更新总均值和总标准差
    mean += batch_mean * batch.size(0)
    std += (batch_std ** 2) * batch.size(0)  # 注意这里是平方

    total_images += batch.size(0)

# 计算最终均值和标准差
mean /= total_images
std = torch.sqrt(std / total_images)

print(f'Mean: {mean}')
print(f'Std: {std}')


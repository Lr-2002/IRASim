import os
import shutil

# 定义原始文件夹和目标文件夹
source_folder = './generate_video/'  # 原始文件夹路径
gt_folder = 'robotdata/opensource_robotdata/languagetable/evaluation_videos/val_sample_videos/'  # 目标文件夹路径
target_folder = './gt_video'
# 确保目标文件夹存在
os.makedirs(target_folder, exist_ok=True)

# 获取所有的 mp4 文件
mp4_files = [f for f in os.listdir(source_folder) if f.endswith('.mp4')]

# 提取视频 ID 并复制对应文件
for file in mp4_files:
    # 提取 video_id
    video_id = file.split('_')[0]  # 根据 '_' 分割并取第一个部分
    target_file = f"{video_id}_0_0.mp4"  # 构造目标文件名

    # 检查目标文件是否存在
    if os.path.exists(os.path.join(gt_folder, target_file)):
        # 复制文件到目标文件夹
        shutil.copy(os.path.join(gt_folder, target_file), target_folder)
        print(f"Copied: {target_file} to {target_folder}")
    else:
        print(f"File not found: {target_file}")

print("完成复制。")


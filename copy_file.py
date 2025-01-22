import os
import shutil

# 定义原始文件夹和目标文件夹
ref_folder = './generate_video/'  # 原始文件夹路径
source_folder = '/ssd/opensource_robotdata/languagetable/videos/val/'
target_folder = './source_ref_videos'
# 确保目标文件夹存在
os.makedirs(target_folder, exist_ok=True)
# 获取所有的 mp4 文件
video_ids = [f for f in os.listdir(ref_folder) if os.path.isdir(os.path.join(ref_folder, f))]
print(video_ids)
# 提取视频 ID 并复制对应文件
for video_id in video_ids:
    print(os.path.join(source_folder, video_id+'/rgb.mp4'))
    if os.path.exists(os.path.join(source_folder, video_id+'/rgb.mp4')):
        # 复制文件到目标文件夹
        shutil.copy(os.path.join(source_folder, video_id+'/rgb.mp4'), os.path.join(target_folder, video_id+'.mp4'))
        print(video_id)
print("完成复制。")


import os 
import numpy as np 
import pickle as pkl
videos = [x[:6] for x in os.listdir('./gt_video') if x.endswith('mp4')]
print(videos)
with open('./validate_video_id.pkl', 'wb') as f :
    pkl.dump(videos, f)

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import torch
import subprocess
from options.train_options import TrainOptions
from loaders import aligned_data_loader
from models import pix2pix_model

def extract_video(vid_filename):
    cmd = [
        'ffmpeg',
        '-i', vid_filename,
        'temp/%06d.png',
        '-threads', '4'
    ]

    print(' '.join(cmd))
    try:
        subprocess.call(cmd)
    except OSError:
        print('OSError')

    # 'ffmpeg -i friends_will_hates_rachel_brad_pitt.mp4 out/%06d.png -threads 4'

def make_video(output_path, img_dir, fps=25):
    """
    output_path is the final mp4 name
    img_dir is where the images to make into video are saved.
    """
    cmd = [
        'ffmpeg',
        '-y',
        '-threads', '16',
        '-framerate', str(fps),
        '-i', '{img_dir}/%*.png'.format(img_dir=img_dir),
        '-profile:v', 'baseline',
        '-level', '3.0',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-an',
        # Note that if called as a string, ffmpeg needs quotes around the
        # scale invocation.
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
        output_path,
    ]
    print(' '.join(cmd))
    try:
        subprocess.call(cmd)
    except OSError:
        print('OSError')

BATCH_SIZE = 1

opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

vid_filename = opt.indir
out_vid_filename = 'out.mp4'
folder = 'temp'

os.makedirs(folder, exist_ok=True)

# extract_video(vid_filename)

output_f = 'temp.txt'

image_list = [os.path.join(folder, x)
              for x in os.listdir(folder)
              if x.endswith('.jpg') or x.endswith('.png')]

with open(output_f, 'w') as f:
    for item in image_list:
        f.write(f'{item}\n')

video_list = output_f # 'test_data/test_pt_list.txt'

eval_num_threads = 2
video_data_loader = aligned_data_loader.GenericDataLoader(video_list, BATCH_SIZE)
video_dataset = video_data_loader.load_data()
print('========================= Video dataset #images = %d =========' %
      len(video_data_loader))

model = pix2pix_model.Pix2PixModel(opt)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
best_epoch = 0
global_step = 0

print(
    '=================================  BEGIN VALIDATION ====================================='
)

print('TESTING ON VIDEO')

model.switch_to_eval()
save_path = 'out_temp'
os.makedirs(save_path, exist_ok=True)
print('save_path %s' % save_path)

for i, data in enumerate(video_dataset):
    stacked_img = data[0]
    targets = data[1]
    model.run_and_save_VIDEOS(stacked_img, targets, save_path)

# folder_name = folder.split("/")[-2]

make_video(out_vid_filename, save_path)

os.removedirs('temp')
os.removedirs('out_temp')

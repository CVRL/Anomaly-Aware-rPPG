import os
import torch
import torch.nn.functional as F
import numpy as np


class Dataset():
    def __init__(self, types='both', debug=False, return_video=True):
        if types not in ['attack','real','both']:
            print('types parameter must be in [attack,real,both].')
            sys.exit(-1)

        self.types = types
        self.return_video = return_video
        ## TODO: set this to the root of the data
        self.root = ...
        self.input_root = os.path.join(self.root, 'input_data')

        self.video_paths, self.labels, self.output_paths = self.get_meta()
        if debug:
            self.video_paths = self.video_paths[:3]
            self.labels = self.labels[:3]
            self.output_paths = self.output_paths[:3]

    def get_meta(self):
        ## video naming is <camera_lighting_subject.avi>
        vid_paths = []
        out_paths = []
        vid_labels = []
        if self.types == 'both':
            vid_classes = ['attack', 'real']
        else:
            vid_classes = [self.types]

        for vid_cls in vid_classes:
            vid_label = int(vid_cls == 'real')
            for num in range(1, 13):
                vid_dir = os.path.join(self.input_root, vid_cls, '%02d'%num)
                out_dir = os.path.join(vid_cls, '%02d'%num)
                vid_files = sorted(os.listdir(vid_dir))
                full_paths = [os.path.join(vid_dir, f) for f in vid_files]
                full_out_paths = [os.path.join(out_dir, f) for f in vid_files]
                vid_paths.extend(full_paths)
                out_paths.extend(full_out_paths)
                vid_labels.extend(len(vid_files)*[vid_label])
        return vid_paths, vid_labels, out_paths

    def interpolate_clip(self, clip, length):
        clip = torch.from_numpy(clip)
        clip = F.interpolate(clip, (length, 64, 64), mode='trilinear', align_corners=False)
        return clip

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        output_path = self.output_paths[idx]
        label =  self.labels[idx]

        if self.return_video:
            video = np.load(video_path)['arr_0'] # every video is 10 seconds
            video = video[:,:,:,[2,1,0]] ## convert to RGB
            video = np.transpose(video, (3,0,1,2))[np.newaxis,:]
            video = video.astype(np.float64) / 255
            video = self.interpolate_clip(video, 900) ## upsample to 90 fps
            video = video.float()
        else:
            video = None

        return video, label, output_path


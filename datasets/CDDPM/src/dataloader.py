import os
import torch
import numpy as np
import pandas as pd


class Dataset():
    def __init__(self, split='test', debug=False, return_video=True, return_waveform=False):
        self.return_video = return_video
        self.return_waveform = return_waveform

        ## TODO: set this to the root of the data
        self.root = ...

        self.subject_strs = self.get_subjects(split)
        self.labels = self.get_labels()
        self.input_root = os.path.join(self.root, 'input_data')
        self.video_paths, self.output_paths = self.get_paths(split)
        if debug:
            self.video_paths = self.video_paths[:3]
            self.output_paths = self.output_paths[:3]
            self.labels = self.labels[:3]

    def get_subjects(self, split):
        ## TODO: set this to the path to metadata
        meta_path = ...
        meta = pd.read_csv(meta_path)
        subject_strs = meta[meta['Set'] == split]['Session ID'].to_numpy()
        return subject_strs

    def get_paths(self, split):
        vid_paths = [os.path.join(self.input_root, s, f'{s}.npz') for s in self.subject_strs]
        out_paths = [os.path.join(s, f'{s}.npz') for s in self.subject_strs]
        return vid_paths, out_paths

    def get_labels(self):
        ## TODO: set this to the directory with ground truth or labels
        lab_root = ...
        label_dirs = [os.path.join(lab_root, n) for n in self.subject_strs]
        waves = [np.load(os.path.join(d, 'wave.npy')) for d in label_dirs]
        return waves

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        output_path = self.output_paths[idx]
        wave = self.labels[idx]
        label = 0

        if self.return_video or self.return_waveform:
            data = np.load(video_path)
            video = data['video'] # every video is 10 seconds
            video = video[:,:,:,[2,1,0]] ## convert to RGB
            video = np.transpose(video, (3,0,1,2))[np.newaxis,:]
            video = video.astype(np.float64) / 255
            video = torch.from_numpy(video).float()
            min_len = min(len(wave), video.shape[2])
            wave = wave[:min_len]
            if self.return_waveform:
                label = wave
        else:
            video = None

        return video, label, output_path


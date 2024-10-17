import os
import torch
import numpy as np
import h5py
import pandas as pd


class Dataset():
    def __init__(self, noise_type, split='test', debug=False, return_video=True):
        self.noise_type = noise_type
        self.return_video = return_video
        self.rng = np.random.default_rng(seed=17)
        self.noise_width = 3
        ## This is the root of the input videos that will be modified
        ## Note that we will not write to this root
        ## TODO: set this to the root of the input DDPM data
        self.input_root = ...
        ## TODO: set this to the root of the NDDPM data
        self.root = ... # should be e.g. f'datasets/NDDPM/{noise_type}'
        self.subject_strs = self.get_subjects(split)
        self.video_paths, self.output_paths = self.get_paths(split)
        if debug:
            self.video_paths = self.video_paths[:3]
            self.output_paths = self.output_paths[:3]

    def get_subjects(self, split):
        ## TODO: set this to the path to metadata
        meta_path = ...
        meta = pd.read_csv(meta_path)
        subject_strs = meta[meta['Set'] == split]['Session ID'].to_numpy()
        if split == 'val':
            subject_strs = subject_strs[subject_strs != '2020-043-039']
        return subject_strs

    def get_paths(self, split):
        vid_paths = [os.path.join(self.input_root, f'{s}.hdf5') for s in self.subject_strs]
        out_paths = [os.path.join(s, f'{s}.npz') for s in self.subject_strs]
        return vid_paths, out_paths

    def create_negative_sample(self, clip):
        clip_len = clip.shape[0]
        if self.noise_type == 'shuffle': #Shuffle
            shuffle_idcs = self.rng.permutation(clip_len)
            clip = clip[shuffle_idcs] # shuffle the frame order
        elif self.noise_type == 'normal': #Normal
            frame_idx = self.rng.integers(0, clip_len)
            clip[:] = clip[frame_idx] # constant frame throughout clip
            noise = self.rng.normal(0, self.noise_width, clip.shape)
            clip = clip + noise
        elif self.noise_type == 'uniform': #Uniform
            frame_idx = self.rng.integers(0, clip_len)
            clip[:] = clip[frame_idx] # constant frame throughout clip
            noise = self.rng.uniform(-self.noise_width, self.noise_width, clip.shape)
            clip = clip + noise
        else:
            print('Invalid noise_type. Must be in [shuffle, normal, uniform].')
            sys.exit(-1)
        return clip

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        output_path = self.output_paths[idx]
        label = 0

        if self.return_video:
            video = h5py.File(video_path, 'r')['data'][:]
            video = video[:,:,:,[2,1,0]] ## convert to RGB
            video = video.astype(np.float64)
            video = self.create_negative_sample(video)
            video = np.transpose(video, (3,0,1,2))[np.newaxis,:]
            video = video / 255
            video = torch.from_numpy(video).float()
        else:
            video = None

        return video, label, output_path


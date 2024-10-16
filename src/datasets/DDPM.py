import torch
import numpy as np
import pandas as pd
from torchvision.datasets.vision import VisionDataset
import sys


class DDPM(VisionDataset):
    def __init__(self, split, arg_obj):
        super(DDPM, self).__init__(split, arg_obj)

        self.fps             = int(arg_obj.fps)
        self.debug           = bool(int(arg_obj.debug))
        self.split           = split.lower()
        self.channels        = arg_obj.channels.lower()
        self.frames_per_clip = int(arg_obj.fpc)
        self.step            = int(arg_obj.step)
        self.skip            = int(arg_obj.skip)
        self.dtype           = arg_obj.dtype
        self.w               = int(arg_obj.frame_width)
        self.h               = int(arg_obj.frame_height)
        self.aug             = arg_obj.augmentation.lower()
        self.aug_flip, self.aug_illum, self.aug_gauss = [False]*3
        self.negative_prob   = float(arg_obj.negative_prob)
        self.noise_width     = float(arg_obj.noise_width)

        self.set_augmentations()
        self.load_data()
        self.build_samples()

        print(self.split)
        print('Samples: ', self.samples.shape)
        print('Total frames: ', self.samples.shape[0] * self.frames_per_clip)


    def load_data(self):
        if self.fps == 30:
            meta = pd.read_csv('datasets/metadata/DDPM_30fps.csv')
        elif self.fps == 90:
            meta = pd.read_csv('datasets/metadata/DDPM_90fps.csv')
        else:
            print('Invalid fps for DDPM loader. Must be in [30,90]. Exiting.')
            sys.exit(-1)

        self.subject_strs = meta[meta['Set'] == self.split]['Session ID'].to_numpy()
        if self.debug:
            self.subject_strs = [self.subject_strs[0]]

        data = []
        self.waves = []
        for idx, row in meta.iterrows():
            if row['Set'] == self.split:
                subj_id = row['Session ID']
                npz = np.load(row['path'])
                d = {k: npz[k] for k in npz.files}
                d['id'] = subj_id
                d['path'] = row['path']
                self.waves.append(d['wave'])
                data.append(d)
        self.data = data


    def set_augmentations(self):
        self.aug_flip = False
        self.aug_illum = False
        self.aug_gauss = False
        if self.split == 'train':
            self.aug_flip = True if 'f' in self.aug else False
            self.aug_illum = True if 'i' in self.aug else False
            self.aug_gauss = True if 'g' in self.aug else False


    def build_samples(self):
        start_idcs = self.get_start_idcs()
        ## Want array of size clips with (subj, start_idx) in each element
        samples = []
        for subj in range(len(self.waves)):
            starts = start_idcs[subj]
            subj_rep = np.repeat(subj, len(starts))
            sample = np.vstack((subj_rep, starts))
            samples.append(sample)
        self.samples = np.hstack(samples).T


    def get_start_idcs(self):
        start_idcs = []
        for wave in self.waves:
            slen = len(wave)
            end = slen - self.frames_per_clip
            starts = np.arange(0, end, self.step)
            start_idcs.append(starts)
        start_idcs = np.array(start_idcs, dtype=object)
        return start_idcs


    def __len__(self):
        return self.samples.shape[0]


    def arrange_channels(self, imgs):
        ## Orig is BGRN:
        d = {'b':0, 'g':1, 'r':2, 'n':3}
        channel_order = [d[c] for c in self.channels]
        imgs = imgs[:,:,:,channel_order]
        return imgs


    def create_negative_sample(self, clip, wave, HR):
        ''' For the default DDPM, we don't want to create negative samples.
        '''
        return clip, wave, HR


    def __getitem__(self, idx):
        subj, start_idx = self.samples[idx]
        idcs = np.arange(start_idx, start_idx + (self.frames_per_clip*self.skip), self.skip)

        HR = self.data[subj]['HR'][idcs] # [T]

        wave = self.data[subj]['wave'][idcs] # [T]
        wave = (wave - wave.mean()) / np.std(wave)

        clip = self.data[subj]['video'][idcs] # [T,H,W,C]
        clip = self.arrange_channels(clip)
        clip = clip.astype(np.float64)

        live = 1

        ## Negative sample via shuffling frame order in a clip
        if np.random.rand() < self.negative_prob:
            clip, wave, HR = self.create_negative_sample(clip, wave, HR)
            live = 0

        clip = np.transpose(clip, (3, 0, 1, 2))

        ## Horizontal flip
        if self.aug_flip:
            if np.random.rand() > 0.5:
                clip = np.flip(clip, 3)

        ## Illumination noise
        if self.aug_illum:
            clip += np.random.normal(0, 10)

        ## Gaussian noise for every pixel
        if self.aug_gauss:
            clip += np.random.normal(0, 2, clip.shape)

        clip = np.clip(clip, 0, 255)
        clip = clip / 255

        if self.dtype[0] == 'f':
            clip = torch.from_numpy(clip).float()
            wave = torch.from_numpy(wave).float()
        else:
            clip = torch.from_numpy(clip).double()
            wave = torch.from_numpy(wave).double()

        return clip, wave, HR, live, subj
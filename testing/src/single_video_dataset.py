import os
import torch
import torch.nn.functional as F
import numpy as np


class Dataset():
    def __init__(self, video, fpc, step):
        self.fpc = fpc
        self.step = step
        self.original_video_len = video.shape[2]
        self.video = self.prepare_video(video)
        self.start_idcs = self.prepare_idcs(self.video)
        self.video_len = self.start_idcs[-1] + self.fpc


    def prepare_video(self, video):
        ''' Pad the end such that we can predict the original length.
        '''
        B,C,T,H,W = video.shape
        if T % self.step != 0:
            ## Repeat last frame at least <step> times for padding
            pad = video[:,:,[-1]].repeat(1,1,self.step,1,1)
            ## Video is now longer, but overlap adding will include entire original
            video = torch.cat((video, pad), dim=2)
        return video


    def prepare_idcs(self, video):
        B,C,T,H,W = video.shape
        end = T - self.fpc
        starts = np.arange(0, end, self.step)
        return starts


    def __len__(self):
        return len(self.idcs)


    def __getitem__(self, idx):
        start = self.start_idcs[idx]
        end = start + self.fpc
        return self.video[:,:,start:end]



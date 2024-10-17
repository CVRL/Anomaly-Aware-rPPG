import os
import torch
import torch.nn.functional as F
from scipy.interpolate import CubicSpline
import numpy as np


class Dataset():
    def __init__(self, debug=False, return_video=True, return_waveform=False):
        self.return_video = return_video
        self.return_waveform = return_waveform

        ## TODO: set this to the root of the data
        self.root = ...

        self.input_root = os.path.join(self.root, 'input_data')
        self.video_paths, self.output_paths = self.get_paths()
        if debug:
            self.video_paths = self.video_paths[:3]
            self.output_paths = self.output_paths[:3]

    def get_paths(self):
        vid_paths = [os.path.join(self.input_root, f) for f in sorted(os.listdir(self.input_root))]
        out_paths = [f for f in sorted(os.listdir(self.input_root))]
        return vid_paths, out_paths

    def interpolate_clip(self, clip, length):
        ''' Obtain a face mask for a frame.

        Args:
            clip (float np.array) [B,C,T,H,W]: cropped/downscaled video clip of face
            length: Interpolated time dimension (e.g. input of [B,C,10,H,W] with length=20 would double the fps)

        Returns:
            np.array: Interpolated video array.
        '''
        clip = torch.from_numpy(clip)
        clip = F.interpolate(clip, (length, 64, 64), mode='trilinear', align_corners=False)
        return clip

    def cubic_spline(self, trace, target_len):
        orig_x = np.linspace(0, 100, len(trace))
        cubic = CubicSpline(orig_x, trace)
        upsamp_x = np.linspace(0, 100, target_len)
        return cubic(upsamp_x)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        output_path = self.output_paths[idx]
        label = 1

        if self.return_video or self.return_waveform:
            data = np.load(video_path)
            fps = data['fps']

            video = data['video'] # every video is 10 seconds (expected shape [T,H,W,C])
            video = video[:,:,:,[2,1,0]] ## convert to RGB
            video = np.transpose(video, (3,0,1,2))[np.newaxis,:]
            video = video.astype(np.float64) / 255

            c = 90.0 / fps
            vid_len = video.shape[2]
            target_len = round(c*vid_len)
            video = self.interpolate_clip(video, target_len) ## upsample to 90 fps
            video = video.float()

            if self.return_waveform:
                wave = data['wave']
                label = self.cubic_spline(wave, target_len)
        else:
            video = None


        return video, label, output_path


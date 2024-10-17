import os
import torch
import torch.nn.functional as F
import numpy as np


class Dataset():
    def __init__(self, debug=False, return_video=True):
        self.return_video = return_video
        ## TODO: set this to the root of the data
        self.root = ...
        self.input_root = os.path.join(self.root, 'input_data')
        self.video_paths, self.output_paths = self.get_paths()
        if debug:
            self.video_paths = self.video_paths[:3]
            self.output_paths = self.output_paths[:3]

    def get_paths(self):
        video_paths = []
        out_paths = []
        locs = sorted(os.listdir(self.input_root))
        for loc in locs:
            loc_dir = os.path.join(self.input_root, loc)
            dates = sorted(os.listdir(loc_dir))
            for date in dates:
                date_dir = os.path.join(loc_dir, date)
                sessions = sorted(os.listdir(date_dir))
                for session in sessions:
                    video_path = os.path.join(date_dir, session, 'image_00', 'data', 'video.npz')
                    out_path = os.path.join(loc, date, session, 'image_00', 'data', 'video.npz')
                    video_paths.append(video_path)
                    out_paths.append(out_path)
        return video_paths, out_paths

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

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        output_path = self.output_paths[idx]
        label = 0

        if self.return_video:
            data = np.load(video_path)
            video = data['video'] # every video is 10 seconds (expected shape [T,H,W,C])
            video = video[:,:,:,[2,1,0]] ## convert to RGB
            video = np.transpose(video, (3,0,1,2))[np.newaxis,:]
            video = video.astype(np.float64) / 255

            fps = 10
            c = 90.0 / fps
            vid_len = video.shape[2]
            video = self.interpolate_clip(video, round(c*vid_len)) ## upsample to 90 fps
            video = video.float()
        else:
            video = None

        return video, label, output_path


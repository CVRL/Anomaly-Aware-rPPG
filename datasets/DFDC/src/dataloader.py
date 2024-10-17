import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd


class Dataset():
    def __init__(self, debug=False, return_video=True, types='both'):
        if types not in ['attack','real','both']:
            print('types parameter must be in [attack,real,both].')
            sys.exit(-1)
        self.return_video = return_video
        self.types = types
        ## TODO: set this to the root of the data
        self.root = ...
        self.df = self.get_df()
        self.input_root = os.path.join(self.root, 'input_data', 'test')
        self.video_paths, self.output_paths, self.labels = self.get_meta()
        if debug:
            self.video_paths = self.video_paths[:3]
            self.output_paths = self.output_paths[:3]
            self.labels = self.labels[:3]

    def get_meta(self):
        filenames = self.df.filename.tolist()
        vid_paths = [os.path.join(self.input_root, f[:-3]+'npz') for f in filenames]
        out_paths = [os.path.join('test', f[:-3]+'npz') for f in filenames]
        labels = [self.df[self.df.filename == f[:-3]+'mp4']['label'].item() for f in filenames]
        return vid_paths, out_paths, labels

    def get_df(self):
        ## TODO: set this to the path to the labels
        df_path = ...
        df = pd.read_csv(df_path)
        df['label'] = 1 - df['label'] #DFDC 1=fake, but we want 1=real
        if self.types == 'attack':
            df = df[df.label == 0]
        elif self.types == 'real':
            df = df[df.label == 1]
        return df

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
        label = self.labels[idx]

        if self.return_video:
            data = np.load(video_path)
            video = data['video'] # every video is 10 seconds (expected shape [T,H,W,C])
            video = video[:,:,:,[2,1,0]] ## convert to RGB
            video = np.transpose(video, (3,0,1,2))[np.newaxis,:]
            video = video.astype(np.float64) / 255
            video = self.interpolate_clip(video, 900) ## upsample to 90 fps
            video = video.float()
        else:
            video = None

        return video, label, output_path


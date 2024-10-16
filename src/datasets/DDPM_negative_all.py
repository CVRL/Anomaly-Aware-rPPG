import numpy as np
from datasets.DDPM import DDPM

class DDPMNegativeAll(DDPM):
    def __init__(self, split, arg_obj, root='.'):
        super().__init__(split, arg_obj)


    def get_name(self):
        print('DDPM Negative All')


    def create_negative_sample(self, clip, wave, HR):
        noise_type = np.random.randint(0,3)
        if noise_type == 0: #Shuffle
            shuffle_idcs = np.random.permutation(self.frames_per_clip)
            clip = clip[shuffle_idcs] # shuffle the frame order
            HR = HR[shuffle_idcs] # shuffle order for wave
        elif noise_type == 1: #Normal
            frame_idx = np.random.randint(self.frames_per_clip)
            clip[:] = clip[frame_idx] # constant frame throughout clip
            noise = np.random.normal(0, self.noise_width, clip.shape)
            clip = clip + noise
            HR = np.random.uniform(0, 240, self.frames_per_clip) # random HR (not used)
        else: #Uniform
            ## Negative sample via single frame plus dynamic noise
            frame_idx = np.random.randint(self.frames_per_clip)
            clip[:] = clip[frame_idx] # constant frame throughout clip
            noise = np.random.uniform(-self.noise_width, self.noise_width, clip.shape)
            clip = clip + noise
            HR = np.random.uniform(0, 240, self.frames_per_clip) # random HR (not used)
        wave = np.zeros(self.frames_per_clip)
        return clip, wave, HR

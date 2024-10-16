import numpy as np
from datasets.DDPM import DDPM

np.random.seed(1)

class DDPMNegativeNormal(DDPM):
    def __init__(self, split, arg_obj, root='.'):
        super().__init__(split, arg_obj)


    def get_name(self):
        print('DDPM Negative Normal')


    def create_negative_sample(self, clip, wave, HR):
        ## Negative sample via single frame plus dynamic noise
        frame_idx = np.random.randint(self.frames_per_clip)
        clip[:] = clip[frame_idx] # constant frame throughout clip
        noise = np.random.normal(0, self.noise_width, clip.shape)
        clip = clip + noise
        wave = np.zeros(self.frames_per_clip)
        HR = np.random.uniform(0, 240, self.frames_per_clip) # random HR (not used)
        return clip, wave, HR


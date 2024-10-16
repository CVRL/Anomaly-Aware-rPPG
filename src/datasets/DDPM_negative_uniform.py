import numpy as np
from datasets.DDPM import DDPM

np.random.seed(1)

class DDPMNegativeUniform(DDPM):
    def __init__(self, split, arg_obj):
        ## std deviation of uniform distribution is (b-a)**2 / 12
        super().__init__(split, arg_obj)


    def get_name(self):
        print('DDPM Negative Uniform')


    def create_negative_sample(self, clip, wave, HR):
        ## Negative sample via single frame plus dynamic noise
        frame_idx = np.random.randint(self.frames_per_clip)
        clip[:] = clip[frame_idx] # constant frame throughout clip
        noise = np.random.uniform(-self.noise_width, self.noise_width, clip.shape)
        clip = clip + noise
        wave = np.zeros(self.frames_per_clip)
        HR = np.random.uniform(0, 240, self.frames_per_clip) # random HR (not used)
        return clip, wave, HR


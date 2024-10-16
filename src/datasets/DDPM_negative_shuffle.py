import numpy as np
from datasets.DDPM import DDPM

np.random.seed(1)

class DDPMNegativeShuffle(DDPM):
    def __init__(self, split, arg_obj, root='.'):
        super().__init__(split, arg_obj)


    def get_name(self):
        print('DDPM Negative Shuffle')


    def create_negative_sample(self, clip, wave, HR):
        ## Negative sample via shuffling frame order in a clip
        shuffle_idcs = np.random.permutation(self.frames_per_clip)
        clip = clip[shuffle_idcs] # shuffle the frame order
        wave = np.zeros(self.frames_per_clip)
        HR = HR[shuffle_idcs] # shuffle order for wave
        return clip, wave, HR


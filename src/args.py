import argparse
import sys

regions = ['face', 'cheek', 'forehead', 'delaunay', 'all']


def get_input():
    parser = argparse.ArgumentParser(description='This script loads or trains\
                                     the CNN.')

    ########################## Training ############################
    parser.add_argument('--debug',
                        default=0,
                        help='Whether to load debug dataset or not. [0]')
    parser.add_argument('--continue_training',
                        default=0,
                        help='Whether to continue training from last model. [0]')

    parser.add_argument('--model_type',
                        default='rpnet',
                        help='Model to use (rpnet). [rpnet]')

    parser.add_argument('--nfft',
                        default=5400,
                        help='Width of fft to increase resolution of frequency spectrum. 90*60=[5400]')
    parser.add_argument('--negative_prob',
                        default=0.5,
                        help='Probability of using a negative sample. [0.5]')
    parser.add_argument('--noise_width',
                        default=3,
                        help='Width of the dynamic normal or uniform noise. [3]')

    parser.add_argument('--dimensions',
                        default='txy',
                        help='Dimensions to allow the attack in [t,x,y]. [txy]')
    parser.add_argument('--loss',
                        default='np',
                        help='Type of loss to use (np, mse, l1, freq). [np]')

    parser.add_argument('--num_workers',
                        default=6,
                        help='Number of threads dedicated to preprocessing and feeding data. [6]')
    parser.add_argument('--load_path',
                        default=None,
                        help='If load_path is not None, then it will load and start training model.')
    parser.add_argument('--dataset',
                        default='mspec',
                        help='Dataset to train model with (mspec, nir, mask, syn_mask). [mspec]')
    parser.add_argument('--rerun_seeds',
                        default='../config/rerun_seeds.json',
                        help='Seed config file.')
    parser.add_argument('--K',
                        default=1,
                        type=int,
                        help='Fold to use when using K folds. [1]')
    parser.add_argument('--lmrk_type',
                        default='openface',
                        help='Which face landmarking to use (openface, Bulat). [openface]')
    parser.add_argument('--tk',
                        default=5,
                        help='Temporal kernel width [5]')
    parser.add_argument('--augmentation',
                        default='fig',
                        help='Augmentation during training. f=flipping, i=illumination \
                              changes, g=gaussian noise. [fig]')
    parser.add_argument('--val_mask_db',
                        default=0,
                        help='Whether to validate model with masked database. [0]')
    parser.add_argument('--mod_mask_db',
                        default=0,
                        help='Whether the model used masked database in training. [0]')
    parser.add_argument('--val_db',
                        default='mspec',
                        help='Whether to validate model with masked database. [0]')
    parser.add_argument('--mod_db',
                        default='mspec',
                        help='Whether the model used masked database in training. [0]')
    parser.add_argument('--channels',
                        default='RGB',
                        help='Input channels (any combo of {R,G,B,N}. [RGB]')
    parser.add_argument('--dropout',
                        default=0.2,
                        help='Dropout used in model. [0.2]')
    parser.add_argument('--batch_size',
                        default=8,
                        help='Batch size for training. [8]')
    parser.add_argument('--frame_width',
                        default=64,
                        help='Width of input frames. [64]')
    parser.add_argument('--frame_height',
                        default=64,
                        help='Height of input frames. [64]')
    parser.add_argument('--fpc',
                        default=136,
                        help='Frames per clip input to the model. [135]')
    parser.add_argument('--step',
                        default=68,
                        help='Step between clips when training. [90]')
    parser.add_argument('--skip',
                        default=1,
                        help='Skip between individual frames in the video. [1]')
    parser.add_argument('--dtype',
                        default='f',
                        help='Data type used for model and data (f, d). [f]')
    parser.add_argument('--detrend',
                        default=0,
                        help='Whether or not to use detrending of the labeled signal [0]')
    parser.add_argument('--lmbda',
                        default=400,
                        help='Lambda value for detrending [500].')
    parser.add_argument('--lr',
                        default=0.0001,
                        help='Learning rate. [0.0001]')
    parser.add_argument('--start_epoch',
                        default=0,
                        help='Starting epoch. [0]')
    parser.add_argument('--end_epoch',
                        default=20,
                        help='Ending epoch. [20]')
    parser.add_argument('--shuffle',
                        default=1,
                        help='Shuffling of training data. [1]')

    ################### Analyzing output and validation ########################
    parser.add_argument('--split',
                        default=0,
                        help='Split of the data (0=train, 1=val, 2=test). [0]')
    parser.add_argument('--fps',
                        default=90,
                        help='The framerate of the video [60]')
    parser.add_argument('--hanning',
                        default=1,
                        help='Whether to use hanning window while overlap-adding. [1]')
    parser.add_argument('--low_hz',
                        default=0.666666666,
                        help='Lower cutoff frequency for FIR filter.')
    parser.add_argument('--high_hz',
                        default=3,
                        help='Upper cutoff frequency for FIR filter.')
    parser.add_argument('--upsamp_hz',
                        default=-1,
                        help='Upsampling frequency if using cubic spline.')
    parser.add_argument('--nfft_scalar',
                        default=2,
                        help='Scalar for nfft in periodogram computation [2]')
    parser.add_argument('--maf_width',
                        default=-1,
                        help='Window width for moving average filter. Not used if \
                        less than 0. Should be 30 for 90 fps if proportional. [20]')
    parser.add_argument('--hamming_width',
                        default=-1,
                        help='Window width for moving average filter. Not used if \
                        less than 0 [-1]')
    parser.add_argument('--butterworth_order',
                        default=-1,
                        help='All in the name. Not used if less than 0 [5]')
    parser.add_argument('--window_size',
                        default=30,
                        help='Size of each window in seconds for FFT. [30]')
    parser.add_argument('--stride',
                        default=1,
                        help='Size of each stride (in seconds) for computing HR.')
    parser.add_argument('--periodogram_window',
                        default='hamming',
                        help='Window to use during DFT in periodogram {boxcar, hamming} [hamming]')
    arg_obj = parser.parse_args()

    return arg_obj



def print_args(args):
    print('')
    for arg in sorted(vars(args)):
        val = getattr(args, arg)
        if val is not None:
            print('{0:<21} {1:<}'.format(arg, val))
        else:
            print('{0:<21} None'.format(arg))
    print('')

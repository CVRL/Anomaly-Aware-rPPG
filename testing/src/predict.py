import os
import torch
import numpy as np
import argparse
import dataset_loader
import single_video_dataset as svd
import postprocess
import sys

from extract_features import estimate_hr_framestride

sys.path.append('../../src/')
import utils.model_selector as model_utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(args):
    print()
    testing_dataset = args.testing_dataset
    loss_type = args.loss_type
    debug = bool(int(args.debug))
    K = args.K

    fpc, step, binary = get_extra_params(loss_type)

    ## Constant pulse rate parameters
    width = 10
    fps = 90
    min_hz = 2.0 / 3.0
    max_hz = 4.0
    nfft = 5400
    maf_width = 271 #For hilbert transform

    best_model_root = '../../best_models'
    model, tag = model_utils.load_best_model(args, best_model_root, device)
    model.eval()

    print('Testing on dataset:', testing_dataset)
    loader = dataset_loader.load(testing_dataset, debug=debug, split='test')
    iter_length = len(loader)
    print('Number of samples:', iter_length)

    output_root = os.path.join(loader.root, 'output_data', K, tag)
    if not os.path.isdir(output_root):
        os.makedirs(output_root)
    print('Saving output data to:', output_root)

    i = 0
    for video, label, output_path in loader:
        output_head, output_tail = os.path.split(output_path)
        output_dir = os.path.join(output_root, output_head)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, output_tail)
        print(video.shape, output_path)

        video_dataset = svd.Dataset(video, fpc, step)
        pred_nonorm, pred_normed = postprocess.infer_over_video(model, video_dataset, device, binary)

        if testing_dataset in ['CDDPM', 'DDPM', 'PURE', 'UBFC_rPPG']:
            pr_nonorm = estimate_hr_framestride(pred_nonorm, fps=fps, width=width, stride=1, min_hz=min_hz, max_hz=max_hz, nfft=nfft)
            pr_normed = estimate_hr_framestride(pred_normed, fps=fps, width=width, stride=1, min_hz=min_hz, max_hz=max_hz, nfft=nfft)
            np.savez_compressed(output_path, pred_nonorm=pred_nonorm, pred_normed=pred_normed, pr_nonorm=pr_nonorm, pr_normed=pr_normed, label=label)
        else:
            np.savez_compressed(output_path, pred_nonorm=pred_nonorm, pred_normed=pred_normed, label=label)

        i += 1

    print(f'Finished after {i} samples')
    print(20*'-')


def get_extra_params(loss_type):
    binary = False
    if loss_type == 'bce':
        binary = True
        fpc = 136
        step = 68

    else: #loss_type in ['np', 'specentropy', 'specflatness', 'deviation', 'deviationmargin', 'mse']:
        binary = False
        fpc = 270
        step = 135

    return fpc, step, binary


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--testing_dataset',
                        help='Dataset to process.')
    parser.add_argument('--K',
                        help='Which model initialization to use [1-5].')
    parser.add_argument('--model_type',
                        help='Type of model for inference.')
    parser.add_argument('--loss_type',
                        help='Loss used to train the model.')
    parser.add_argument('--dataset',
                        help='Training dataset used by model.')
    parser.add_argument('--noise_width',
                        default=3.0,
                        help='Width of noise functions for negative samples.')
    parser.add_argument('--nfft',
                        default=5400,
                        help='Resolution of fft during spectral regularization.')
    parser.add_argument('--negative_prob',
                        default=0.5,
                        help='Probability of encountering a negative sample during training.')
    parser.add_argument('--tk',
                        default=5,
                        help='Temporal kernel width.')
    parser.add_argument('--channels',
                        default='rgb',
                        help='Color channels.')
    parser.add_argument('--dropout',
                        default=0.0,
                        help='Dropout for model.')
    parser.add_argument('--dtype',
                        default='f',
                        help='Model dtype.')
    parser.add_argument('--debug',
                        default=False,
                        help='Are we in debugging mode to decrease dataset size?')
    args = parser.parse_args()
    main(args)
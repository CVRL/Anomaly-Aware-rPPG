import numpy as np
import argparse
import os
from preprocessing_utils import get_spatial_averages
import CHROM
import POS


def main(args):

    video_root = args.input
    lmrks_root = args.lmrks
    output_root = args.output
    wave_root = args.waves

    os.makedirs(output_root, exist_ok=True)

    sessions = ls(video_root)

    for session in sessions:
        ## TODO: update the video path
        video_path = join(video_root, session, 'video.mp4')
        lmrks_path = join(lmrks_root, f'{session}.npz')
        output_path = join(output_root, f'{session}.npz')

        ## TODO: load the wave data here if it is available
        if wave_root is not None:
            wave_path = join(wave_root, f'{session}.csv')
            wave = np.loadtxt(wave_path)
        else:
            wave = None

        preproc_data = np.load(lmrks_path)
        fps = preproc_data['fps']
        lmrks = preproc_data['lmrks']

        signals, bboxes, successful = get_spatial_averages(video_path, lmrks)
        chrom_sig = CHROM.process_CHROM(signals, fps)
        pos_sig = POS.process_POS(signals, fps)
        if successful:
            np.savez_compressed(output_path, signals=signals, bboxes=bboxes, chrom=chrom_sig, pos=pos_sig, video_path=video_path, lmrks_path=lmrks_path, wave=wave, fps=fps)
        else:
            print(f'Skipping {session} because landmarks could not be detected.')


def ls(x='.'):
    return sorted(os.listdir(x))


def join(*x):
    return os.path.join(*x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input',
                        help='Path to the video directory.')
    parser.add_argument('lmrks',
                        help='Path to the landmarks directory.')
    parser.add_argument('output',
                        help='Path to the baselines output directory with CHROM and POS signals.')
    parser.add_argument('--waves',
                        default=None,
                        help='(Optional) Path to the folder containing the ground truth waves.')
    args = parser.parse_args()

    main(args)


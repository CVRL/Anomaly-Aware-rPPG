import numpy as np
import argparse
import os
import sys
import preprocessing_utils as utils


def ls(x='.'):
    return sorted(os.listdir(x))

def join(*x):
    return os.path.join(*x)


def main(args):

    input_root = args.input
    output_root = args.output
    fps = args.fps

    if not os.path.isdir(output_root):
        os.makedirs(output_root)

    sessions = ls(input_root)

    for session in sessions:
        ## TODO: update the video path
        video_path = join(input_root, session, 'vid.avi')
        ## TODO: update the input ground truth path (if PPG wave is present)
        gt_path = join(input_root, session, f'ground_truth.txt')
        wave, HR, wave_t = np.loadtxt(gt_path)

        output_path = join(output_root, f'{session}.npz')
        lmrks = utils.mediapipe_landmark_video(video_path)
        output_video, successful = utils.make_video_array(video_path, lmrks)
        if successful:
            print('video shape: ', output_video.shape)
            print('lmrks shape: ', lmrks.shape)
            print('waves shape: ', wave.shape)
            print(output_path)
            np.savez_compressed(output_path, video=output_video, wave=wave, video_path=video_path, lmrks=lmrks, fps=fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input',
                        help='Path to the dataset directory.')
    parser.add_argument('output',
                        help='Path to the preprocessed output dataset directory with cropped faces and landmarks.')
    parser.add_argument('fps',
                        type=float,
                        help='Frame rate of the input videos.')
    args = parser.parse_args()
    main(args)


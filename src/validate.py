import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import args
import os
import sys
from tqdm import tqdm
from copy import deepcopy
from scipy.stats import pearsonr

from datasets.utils import get_dataset
from losses import select_loss
from utils.model_selector import select_model
from SignalProc import SignalProcessor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def main():

    arg_obj = args.get_input()

    load_path = arg_obj.load_path
    model_type = arg_obj.model_type.lower() # default _rec_
    split = int(arg_obj.split)
    val_db = arg_obj.val_db
    step = int(arg_obj.step)
    fpc = int(arg_obj.fpc)
    loss_type = arg_obj.loss.lower()
    negative_prob = float(arg_obj.negative_prob)

    use_hanning = 1
    frame_rate = 90
    window_size = 30

    arg_obj.tk = 5
    arg_obj.dataset = val_db
    arg_obj.fps = frame_rate
    arg_obj.fpc = fpc
    arg_obj.step = step

    args.print_args(arg_obj)

    if not os.path.exists(load_path):
        print(f'Could not find path to model: {load_path}. Exiting.')
        sys.exit(-1)

    print('using load_path:', load_path)
    print('using model_type:', model_type)

    hrcnn = select_model(arg_obj, device)
    hrcnn = hrcnn.to(device)
    checkpoint = torch.load(load_path)
    hrcnn.load_state_dict(checkpoint['model_state_dict'])
    hrcnn.eval()

    criterion = select_loss(loss_type)

    arg_obj.window_size = window_size
    arg_obj.periodogram_window = 'hamming'
    signal_processor = SignalProcessor(arg_obj)
    signal_processor.lmbda = 0
    signal_processor.sampling_hz = frame_rate

    split_str = ['train','val','test'][split]
    data_set = get_dataset(split_str, arg_obj)
    loader = torch.utils.data.DataLoader(data_set, batch_size=1,
                                         num_workers=0, shuffle=False)
    print(f'Validating over {split_str} set with length {len(loader)}.')

    subjs = []
    waves = []
    HRs = []
    wave_preds = []
    HR_preds = []
    all_losses = 0
    loader_iterator = iter(loader)
    iter_length = len(loader)
    print('')
    print('*** Validating Model ***')
    print('Total iters: ', iter_length)
    pbar = tqdm(total=iter_length)
    for i in range(iter_length):
        try:
            data = next(loader_iterator)
        except StopIteration:
            loader_iterator = iter(loader)
            data = next(loader_iterator)

        frames, wave, HR, live, subj = (data[0].to(device), data[1].to(device), data[2], data[3].to(device), data[4])

        with torch.set_grad_enabled(False):
            outputs = hrcnn(frames)
            loss = criterion(outputs, wave)
            all_losses += loss.item()
            wave_pred_copy = deepcopy(outputs.cpu().numpy())
            wave_preds.append(wave_pred_copy)
            subj_copy = deepcopy(subj.cpu().numpy())
            subjs.append(subj_copy)
            HR_copy = deepcopy(HR.cpu().numpy())
            HRs.append(HR_copy)
            wave_copy = deepcopy(wave.cpu().numpy())
            waves.append(wave_copy)
            del subj
            del HR
            del loss
            del wave
            del outputs

        pbar.update(1)

    pbar.close()
    del loader_iterator
    ave_loss = all_losses / iter_length
    print('Loss: %.6f' % (ave_loss))
    print('************************')
    print('')

    wave_pred = np.vstack(wave_preds)
    wave = np.vstack(waves)
    HRs = np.vstack(HRs)
    subjs = np.hstack(subjs)
    print('HR: ', HRs.shape)
    print('Waves: ', wave.shape, wave_pred.shape)
    print('Task: ', subjs.shape)

    wave_preds, wave_arrs, HR_arrs = partition_by_subject(wave_pred, wave, HRs, subjs)
    print('After partitioning: ', wave_preds.shape, wave_arrs.shape, HR_arrs.shape)
    pred_arrs = overlap_add(wave_preds, fpc, step, use_hanning=use_hanning)
    wave_arrs, HR_arrs = flatten_ground_truth(wave_arrs, HR_arrs)
    print('After overlap_add: ', pred_arrs.shape, wave_arrs.shape, HR_arrs.shape)

    ### Save the outputs for each model
    model_name = load_path.split('/')[-1]
    output_tag = os.path.splitext(model_name)[0]

    gt_waves, gt_HRs, _ = load_signals(split)

    ## Clip gt to the same length as preds
    for i in range(len(pred_arrs)):
        pred_arr = pred_arrs[i]
        gt_wave = gt_waves[i]
        gt_HR = gt_HRs[i]
        if len(gt_wave) >= len(pred_arr):
            gt_waves[i] = gt_wave[:len(pred_arr)]
        else:
            pred_arrs[i] = pred_arr[:len(gt_wave)]
        if len(gt_HR) > len(pred_arr):
            gt_HRs[i] = gt_HR[:len(pred_arr)]

    HR_preds = predict_HRs(pred_arrs, signal_processor, smoothing=True)
    #gt_HRs = predict_HRs(gt_waves, signal_processor, smoothing=True)

    output_filename = f'{output_tag}_eval{val_db}_evalnegprob{negative_prob:.2f}_{split}.npz'
    np.savez_compressed(os.path.join('../outputs', output_filename),
        oadd_preds=pred_arrs,
        wave_preds=wave_preds,
        HR_preds=HR_preds,
        gt_wave=gt_waves,
        gt_HR=gt_HRs,
        subjs=subjs
    )

    flat_wave_preds = np.hstack((pred_arrs))
    flat_HR_preds = np.hstack((HR_preds))
    flat_gt_waves = np.hstack((gt_waves))
    flat_gt_HRs = np.hstack((gt_HRs))

    flat_gt_waves = flat_gt_waves[:len(flat_wave_preds)]
    flat_gt_HRs = flat_gt_HRs[:len(flat_wave_preds)]

    ME_HR = np.mean(flat_gt_HRs - flat_HR_preds)
    MAE_HR = np.mean(np.abs(flat_gt_HRs - flat_HR_preds))
    RMSE_HR = np.sqrt(np.mean(np.square(flat_gt_HRs - flat_HR_preds)))
    r_HR, p_HR = pearsonr(flat_HR_preds, flat_gt_HRs)
    r_wave, p_wave = pearsonr(flat_wave_preds, flat_gt_waves)

    print()
    print('ME_HR, MAE_HR, RMSE_HR, r_HR, r_wave')
    print(ME_HR, MAE_HR, RMSE_HR, r_HR, r_wave)
    print()
    print(output_filename)
    print('------------------------------------------------')



def partition_by_subject(wave_pred, wave, HRs, subjs):
    pred_arrs = []
    wave_arrs = []
    HR_arrs = []
    unique_subj = np.unique(subjs)
    for i, subj in enumerate(unique_subj):
        subj_idcs = np.where(subjs == subj)[0]
        subj_preds = wave_pred[subj_idcs]
        subj_wave = wave[subj_idcs]
        subj_HRs = HRs[subj_idcs]
        pred_arrs.append(subj_preds)
        wave_arrs.append(subj_wave)
        HR_arrs.append(subj_HRs)
    pred_arrs = np.array(pred_arrs, dtype=object)
    wave_arrs = np.array(wave_arrs, dtype=object)
    HR_arrs = np.array(HR_arrs, dtype=object)
    return pred_arrs, wave_arrs, HR_arrs


def flatten_ground_truth(wave_arrs, HR_arrs):
    flat_wave_arrs = []
    flat_HR_arrs = []
    for i in range(len(wave_arrs)):
        wave_arr = np.hstack(wave_arrs[i])
        HR_arr = np.hstack(HR_arrs[i])
        flat_wave_arrs.append(wave_arr)
        flat_HR_arrs.append(HR_arr)
    flat_wave_arrs = np.array(flat_wave_arrs, dtype=object)
    flat_HR_arrs = np.array(flat_HR_arrs, dtype=object)
    return flat_wave_arrs, flat_HR_arrs


def overlap_add(pred_arrs, fpc, step, use_hanning=True):
    oadd_arrs = []
    hanning_window = np.hanning(fpc)
    for i, subj_pred in enumerate(pred_arrs):
        plen = (len(subj_pred)*step) + (fpc-step)
        oadd_arr = np.zeros(plen)
        for j, subj_win in enumerate(subj_pred):
            start = j*step
            end = start+fpc
            if use_hanning:
                oadd_arr[start:end] = oadd_arr[start:end] + (hanning_window * ((subj_win - np.mean(subj_win)) / np.std(subj_win)))
            else:
                oadd_arr[start:end] = oadd_arr[start:end] + ((subj_win - np.mean(subj_win)) / np.std(subj_win))
        oadd_arr = (oadd_arr - np.mean(oadd_arr)) / np.std(oadd_arr)
        oadd_arrs.append(oadd_arr)
    if len(oadd_arrs) > 1:
        oadd_arrs = np.array(oadd_arrs, dtype=object)
    else:
        oadd_arrs = np.array(oadd_arrs, dtype=float)
    return oadd_arrs


def overlap_add_no_norm(pred_arrs, fpc, step, use_hanning=True):
    oadd_arrs = []
    hanning_window = np.hanning(fpc)
    for i, subj_pred in enumerate(pred_arrs):
        plen = (len(subj_pred)*step) + (fpc-step)
        oadd_arr = np.zeros(plen)
        for j, subj_win in enumerate(subj_pred):
            start = j*step
            end = start+fpc
            if use_hanning:
                oadd_arr[start:end] = oadd_arr[start:end] + (hanning_window * subj_win)
            else:
                oadd_arr[start:end] = oadd_arr[start:end] + subj_win
        oadd_arrs.append(oadd_arr)
    if len(oadd_arrs) > 1:
        oadd_arrs = np.array(oadd_arrs, dtype=object)
    else:
        oadd_arrs = np.array(oadd_arrs, dtype=float)
    return oadd_arrs


def predict_HRs(pred_waves, signal_processor, smoothing=False, maf_width=451):
    HRs = []
    for wave in pred_waves:
        HR = signal_processor.sliding_bpm(wave, stride=1, method='standardize')
        if smoothing:
            HR = np.convolve(np.pad(HR, maf_width//2, 'edge'), np.ones((maf_width))/maf_width, mode='valid')
        HRs.append(HR)
    HRs = np.array(HRs, dtype=object)
    return HRs


def load_signals(split):
    ## TODO: replace this with your metadata path to the DDPM dataset
    meta_path = 'datasets/Meta.csv'

    meta = pd.read_csv(meta_path)
    split_str = ['train', 'val', 'test'][split]
    subjects = meta[meta['Set'] == split_str]['Session ID'].to_numpy()

    ## TODO: add your root directory of the DDPM dataset
    root = None

    waves = []
    HRs = []
    for subject in subjects:
        wave_path = os.path.join(root, subject, 'wave.npy')
        HR_path = os.path.join(root, subject, 'HR.npy')
        wave = np.load(wave_path)
        HR = np.load(HR_path)
        waves.append(wave)
        HRs.append(HR)
    waves = np.array(waves)
    HRs = np.array(HRs)
    last_vid_idcs = None
    return waves, HRs, last_vid_idcs


if __name__ == '__main__':
    main()


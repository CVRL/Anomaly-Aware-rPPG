import numpy as np
import torch
import torch.nn as nn
import os

def infer_over_video(model, video_dataset, device, binary=False):
    wave_normed = np.zeros(video_dataset.video_len)
    wave_nonorm = np.zeros(video_dataset.video_len)

    fpc = video_dataset.fpc
    step = video_dataset.step
    hanning_window = np.hanning(fpc)

    if binary:
        sigmoid = nn.Sigmoid()

    for i, clip in enumerate(video_dataset):

        ## Get predictions for this clip
        clip_pred = model(clip.to(device))
        if binary:
            clip_pred = sigmoid(clip_pred)
        clip_pred = clip_pred.cpu().detach().numpy()

        ## Overlap add for both an unnormalized and normalized version
        start = i*step
        end = start+fpc

        wave_nonorm[start:end] = wave_nonorm[start:end] + (hanning_window * clip_pred)

        norm = np.std(clip_pred)
        if np.isclose(norm, 0):
            norm = 1e-6
        wave_normed[start:end] = wave_normed[start:end] + (hanning_window * ((clip_pred - np.mean(clip_pred)) / norm))

    ## Cut last padded portion to original length
    wave_normed = wave_normed[:video_dataset.original_video_len]
    wave_nonorm = wave_nonorm[:video_dataset.original_video_len]

    ## Normalize the final result of the normed wave
    norm = np.std(wave_normed)
    if np.isclose(norm, 0):
        norm = 1e-6
    wave_normed = (wave_normed - np.mean(wave_normed)) / norm

    return wave_nonorm, wave_normed


def get_output_tag(model_type, dataset, loss_type, negative_prob, noise_width, nfft, epoch=None):
    if epoch is None:
        tag = 'mod%s_db%s_%s_negprob%.2f_sigma%.2f_nfft%d' % (model_type, dataset, loss_type, negative_prob, noise_width, nfft)
    else:
        tag = 'mod%s_db%s_%s_negprob%.2f_sigma%.2f_nfft%d_e%d' % (model_type, dataset, loss_type, negative_prob, noise_width, nfft, epoch)
    return tag


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

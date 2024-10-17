import os, sys
import numpy as np
from natsort import natsorted
import scipy.signal
from pyampd import ampd


def moving_average_filter(sig, maf_width):
    sig = np.convolve(sig, np.ones(maf_width), 'same') / maf_width
    return sig


def center_peaks(pred, peaks, w=10):
    true_peaks = peaks.copy()
    for i, peak in enumerate(peaks):
        start = peak-w
        end = peak+w
        if start < 0:
            start = 0
        if end > len(pred):
            end = len(pred)
        true_peaks[i] = start + np.argmax(pred[start:end])
    return true_peaks


def interpolate_intervals(intervals, wave, mode='linear'):
    t = np.cumsum(intervals)
    t -= t[0]
    f_interpol = scipy.interpolate.interp1d(t, intervals, mode)
    t_interpol = np.arange(t[0], t[-1], 1/90)
    intervals = f_interpol(t_interpol)
    t_interpol, intervals = pad_edge(intervals, t_interpol, wave)
    return t_interpol, intervals


def interpolate_bpms(bpms, intervals, mode='linear'):
    t = np.cumsum(intervals)
    t -= t[0]
    f_interpol = scipy.interpolate.interp1d(t, bpms, mode)
    t_interpol = np.arange(t[0], t[-1], 1/90)
    bpms = f_interpol(t_interpol)
    return t_interpol, bpms


def interpolate_moments(intervals, wave, mode='linear'):
    t = np.linspace(0, len(wave)/90, len(intervals))
    f_interpol = scipy.interpolate.interp1d(t, intervals, mode)
    t_interpol = np.linspace(0, len(wave)/90, len(wave))
    intervals = f_interpol(t_interpol)
    t_interpol, intervals = pad_edge(intervals, t_interpol, wave)
    return t_interpol, intervals


def pad_edge(bpms, t, HR):
    diff = len(HR) - len(bpms)
    pad = int(diff/2)
    t = np.hstack((np.arange(t[0]-pad/90, t[0], 1/90), t, np.arange(t[-1], t[-1]+pad/90, 1/90)))
    bpms = np.pad(bpms, pad, mode='edge')
    if len(bpms) < len(HR):
        bpms = np.hstack((bpms, bpms[-1]))
    if len(t) < len(HR):
        t = np.hstack((t, t[-1]+(1/90)))
    elif len(t) > len(HR):
        t = t[:-1]
    t -= t[0]
    return t, bpms


def windowed_func(sig, func, n, stride=1):
    '''
    Applies a function over windows of the signal.
    Args:
        sig (np.array): signal
        func (function): function to be applied to every window
        n (int): window width
        stride (int): stride of window (default=1)
    Returns:
        sig (np.array): signal after applying function over windows of it.
    '''
    if n % 2 == 0:
        print('Use odd length window')
        sys.exit(-1)

    pad_len = int((n-1)/2)
    pad_sig = np.pad(sig, pad_len, 'edge')
    out_sig = np.zeros_like(sig, dtype=np.float64)
    for i in range(pad_len, len(pad_sig)-pad_len):
        out_sig[i-pad_len] = func(pad_sig[i-pad_len:i+pad_len+1])
    return out_sig


def norm_std(sig):
    sig = (sig - np.mean(sig)) / np.std(sig)
    return sig


def bpm_from_peaks(intervals, n=45):
    '''
    peaks: must be in s
    '''
    intervals = windowed_func(intervals, np.mean, n)
    bpms = 60 * (1/intervals)
    return bpms


def calc_RMSSD(intervals):
    N = len(intervals)
    return np.sqrt((1/(N-1))*np.sum(np.diff(intervals)**2))


def find_peaks_scipy(sig):
    #peaks, _ = scipy.signal.find_peaks(sig, distance=30, prominence=, height=, width=)
    peaks, _ = scipy.signal.find_peaks(sig, distance=30, prominence=0.8)
    return peaks


def find_peaks_ampd(sig, mode='original', scale=270, window=270):
    if mode == 'original':
        peaks = ampd.find_peaks_original(sig, scale=scale)
    elif mode == 'adaptive':
        peaks = ampd.find_peaks_adaptive(sig, window=window)
    else:
        peaks  = ampd.find_peaks(sig, scale=scale)
    return peaks


def peak_refinement(sig, peaks, low_inter=.333333, high_inter=1.333333):
    inters = np.diff(peaks)/90
    print(np.mean(inters))
    low_inter_idcs = np.where(inters < low_inter)[0]
    high_inter_idcs = np.where(inters > high_inter)[0]
    print(low_inter_idcs)
    print(high_inter_idcs)
    window_len = 30
    argmaxes = np.zeros_like(sig)
    for i in range(0, len(sig)-window_len):
        amax = np.argmax(sig[i:i+window_len])
        argmaxes[i+amax] += 1
    return argmaxes


def short_term_autocorrelation(x, win_size=120, t=1):
    for i in range(0, len(x) - win_size):
        x[i:i+win_size]
    return np.corrcoef(np.array([x[:-t], x[t:]]))


def extract_template(sig):
    return sig[16738:16785]


def ricker_template(width=12):
    #template = scipy.signal.triang(M=45)
    #template = scipy.signal.ricker(100, 15)
    return scipy.signal.ricker(100, width)


def find_peaks_template(sig, freqs=[1,1.25,1.5,1.75,2,2.25,2.5]):
    template = np.load('template.npy')
    for freq in freqs:
        template_f = modulate_freq(template, freq)


def modulate_freq(template, freq):
    base_freq = 60


def convolve_template(sig, template):
    filtered = scipy.signal.filtfilt(template, np.array([1]), sig)
    return filtered


def peak_intervals(peaks):
    return np.diff(peaks)


def interval_diffs(intervals):
    return np.abs(np.diff(intervals))


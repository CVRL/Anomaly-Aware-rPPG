import os
import numpy as np
import argparse
import scipy.signal as signal
import dataset_loader
import temporal_feature_extraction as tfeats

def makedir(d):
    if not os.path.isdir(d):
        try:
            os.makedirs(d)
        except:
            pass


def main(args):
    print()
    task_id = int(args.task_id) - 1
    task_count = int(args.task_count)
    testing_dataset = args.testing_dataset
    debug = bool(int(args.debug))
    K = args.K

    ## Constant parameters
    width = 10
    fps = 90
    min_hz = 2.0 / 3.0
    max_hz = 4.0
    nfft = 5400
    maf_width = 271 #For hilbert transform

    print('Testing on dataset:', testing_dataset)

    loader = dataset_loader.load(testing_dataset, debug=debug, return_video=False, split='val')
    #loader = dataset_loader.load(testing_dataset, debug=debug, return_video=False, split='test')
    print('Loader root:', loader.root)

    iter_length = len(loader)
    print('Number of samples:', iter_length)

    model_tags = [
            'modrpnet_dbmspech5_np_negprob0.00_sigma3.00_nfft5400',
            'modrpnet_dbnegallh5_specentropy_negprob0.50_sigma3.00_nfft5400',
            'modrpnet_dbnegallh5_deviation_negprob0.50_sigma3.00_nfft5400',
            'modrpnet_dbnegallh5_specflatness_negprob0.50_sigma3.00_nfft5400',
            'chrom',
            'pos'
    ]
    print('model_tags:', model_tags)

    for model_tag in model_tags:
        print()
        if model_tag in ['chrom', 'pos']:
            input_dir = os.path.join(loader.root, 'WACV_round1', 'output_data', model_tag)
            output_dir = os.path.join(loader.root, 'baselines_features', model_tag)
        else:
            input_dir = os.path.join(loader.root, 'output_data', K, model_tag)
            output_dir = os.path.join(loader.root, 'features', K, model_tag)

        print(f'Reading predictions from directory {input_dir}')
        makedir(output_dir)
        print(f'Saving output data to directory {output_dir}')

        i = 0
        for _, label, input_tag in loader:
            if (i % task_count) != task_id:
                i += 1
                continue

            input_path = os.path.join(input_dir, input_tag)

            output_head, output_tail = os.path.split(input_tag)
            output_head = os.path.join(output_dir, output_head)
            makedir(output_head)
            output_path = os.path.join(output_head, output_tail)

            print(f'Reading from {input_path}')
            print(f'Writing to {output_path}')
            assert(input_path != output_path)

            preds = np.load(input_path)
            pred_nonorm = preds['pred_nonorm']
            pred_normed = preds['pred_normed']
            print('wave:', pred_nonorm.shape, pred_normed.shape)

            ## Smoothed Hilbert Envelope does not require padding
            window_width = width * fps
            start = window_width // 2
            end = start - 1
            envelope_nonorm = estimate_smoothed_envelope(pred_nonorm, maf_width)[start:-end]
            envelope_normed = estimate_smoothed_envelope(pred_normed, maf_width)[start:-end]
            print('envelope:', envelope_nonorm.shape, envelope_normed.shape)

            ## Moving SNR
            snr_nonorm = estimate_snr_framestride(pred_nonorm, fps=fps, width=width, stride=1, min_hz=min_hz, max_hz=max_hz, nfft=nfft)
            snr_normed = estimate_snr_framestride(pred_normed, fps=fps, width=width, stride=1, min_hz=min_hz, max_hz=max_hz, nfft=nfft)
            print('snr:', snr_nonorm.shape, snr_normed.shape, np.isnan(snr_nonorm).sum(), np.isinf(snr_nonorm).sum())

            ## Root Mean Squared Amplitude
            rms_nonorm = estimate_rms_framestride(pred_nonorm, fps=fps, width=width, stride=1)
            rms_normed = estimate_rms_framestride(pred_normed, fps=fps, width=width, stride=1)
            print('rms:', rms_nonorm.shape, rms_normed.shape)

            ## Calculate waveform peaks of upside-down signal (easier)
            try:
                peaks_nonorm = tfeats.find_peaks_ampd(-pred_nonorm, mode='adaptive')
                peaks_normed = tfeats.find_peaks_ampd(-pred_normed, mode='adaptive')
                mean_i_nonorm, std_i_nonorm, mean_idiffs_nonorm, std_idiffs_nonorm, rmssd_nonorm = estimate_interval_statistics(pred_nonorm, peaks_nonorm, fps=fps, width=width, stride=1)
                mean_i_normed, std_i_normed, mean_idiffs_normed, std_idiffs_normed, rmssd_normed = estimate_interval_statistics(pred_normed, peaks_normed, fps=fps, width=width, stride=1)
                print('intervals:', mean_i_nonorm.shape, std_i_nonorm.shape)
                print('interval diffs:', mean_idiffs_nonorm.shape, std_idiffs_nonorm.shape)
                print('RMSSD:', rmssd_nonorm.shape, rmssd_normed.shape)
            except ValueError: #raised if peaks could not be found (e.g. constant prediction)
                N = len(rms_nonorm)
                mean_i_nonorm      = np.zeros(N)
                mean_i_normed      = np.zeros(N)
                std_i_nonorm       = np.zeros(N)
                std_i_normed       = np.zeros(N)
                mean_idiffs_nonorm = np.zeros(N)
                mean_idiffs_normed = np.zeros(N)
                std_idiffs_nonorm  = np.zeros(N)
                std_idiffs_normed  = np.zeros(N)
                rmssd_nonorm       = np.zeros(N)
                rmssd_normed       = np.zeros(N)

            np.savez_compressed(output_path,
                    snr_nonorm=snr_nonorm,
                    snr_normed=snr_normed,
                    envelope_nonorm=envelope_nonorm,
                    envelope_normed=envelope_normed,
                    rms_nonorm=rms_nonorm,
                    rms_normed=rms_normed,
                    mean_i_nonorm=mean_i_nonorm,
                    mean_i_normed=mean_i_normed,
                    std_i_nonorm=std_i_nonorm,
                    std_i_normed=std_i_normed,
                    mean_idiffs_nonorm=mean_idiffs_nonorm,
                    mean_idiffs_normed=mean_idiffs_normed,
                    std_idiffs_nonorm=std_idiffs_nonorm,
                    std_idiffs_normed=std_idiffs_normed,
                    rmssd_nonorm=rmssd_nonorm,
                    rmssd_normed=rmssd_normed,
                    label=label,
                    width=width)

            i += 1

    print(f'Finished after {i} samples')
    print(20*'-')
    return



def smooth(wave, maf_width):
    return np.convolve(np.pad(wave, maf_width//2, 'edge'), np.ones((maf_width))/maf_width, mode='valid')


def estimate_smoothed_envelope(sig, maf_width):
    analytic = signal.hilbert(sig)
    envelope = np.abs(analytic)
    envelope = smooth(envelope, maf_width)
    return envelope


def estimate_interval_statistics(sig, peaks, fps=90, width=20, stride=1):
    N = len(sig)
    window_width = int(fps*width)
    means = []
    stds = []
    diff_means = []
    diff_stds = []
    rmssds = []
    ## Brute force (!)
    for i in range(0, N-window_width+1, stride):
        start_idx = i
        end_idx = i + window_width
        window_peak_idcs = np.logical_and(peaks >= start_idx, peaks < end_idx)
        window_peaks = peaks[window_peak_idcs]
        if len(window_peaks) > 1:
            intervals = np.diff(window_peaks) / fps
            mean = np.mean(intervals)
            std = np.std(intervals)
            if len(intervals) > 1:
                intervals_diff = np.diff(intervals)
                diff_mean = np.mean(intervals_diff)
                diff_std = np.std(intervals_diff)
                rmssd = tfeats.calc_RMSSD(intervals)
            else:
                diff_mean = 0
                diff_std = 0
                rmssd = 0
        else:
            mean = 0
            std = 0
            diff_mean = 0
            diff_std = 0
            rmssd = 0
        means.append(mean)
        stds.append(std)
        diff_means.append(diff_mean)
        diff_stds.append(diff_std)
        rmssds.append(rmssd)
    means = np.array(means)
    stds = np.array(stds)
    diff_means = np.array(diff_means)
    diff_stds = np.array(diff_stds)
    rmssds = np.array(rmssds)
    return means, stds, diff_means, diff_stds, rmssds


def pad_signal(sig, fps=90, width=20):
    N = len(sig)
    window_width = int(fps*width)
    left_width = window_width // 2
    right_width = window_width // 2
    if window_width % 2 == 1:
      left_width += 1
    pad_sig = np.pad(sig, left_width, mode='edge')
    return pad_sig


def estimate_rms_framestride(sig, fps=90, width=20, stride=1):
    N = len(sig)
    window_width = int(fps*width)
    rmss = []
    for i in range(0, N-window_width+1, stride):
      sig_window = sig[i:(i+window_width)]
      ac_window = sig_window - np.mean(sig_window)
      rms = np.sqrt(np.mean(np.square(ac_window)))
      rmss.append(rms)
    rmss = np.array(rmss)
    return rmss


def snr_eval(sig, fps=90, harm_width=6, min_hz=0.66667, max_hz=4, nfft=1024):
    '''
    harm_width is in bpm.
    '''
    f, amp_spect = signal.periodogram(sig, fs=fps, window='hamming', nfft=nfft, return_onesided=True, scaling='spectrum')

    ## If a near-constant signal, we can consider a flat spectrum for stability
    if np.isclose(np.std(sig), 0):
        amp_spect = np.ones_like(amp_spect)

    amp_spect = amp_spect**2
    band_idcs = np.where((f > min_hz) & (f < max_hz))[0]
    band_spect = amp_spect[band_idcs]
    band_spect = band_spect / np.sum(band_spect)
    band_f = f[band_idcs]

    ## estimate spectral peak
    hr_freq = band_f[np.argmax(band_spect)]

    ## set bounds for first harmonic
    first_low_freq = hr_freq - (harm_width/60)
    first_high_freq = hr_freq + (harm_width/60)
    first_harm_idcs = (band_f > first_low_freq) & (band_f < first_high_freq)
    first_harm_power = np.sum(band_spect[first_harm_idcs])

    second_low_freq = 2*hr_freq - (harm_width/60)
    second_high_freq = 2*hr_freq + (harm_width/60)
    second_harm_idcs = (band_f > second_low_freq) & (band_f < second_high_freq)
    second_harm_power = np.sum(band_spect[second_harm_idcs])

    sig_power = first_harm_power + second_harm_power
    noise_power = np.sum(band_spect) - sig_power
    snr = 10 * np.log10(sig_power / noise_power)

    return snr, band_f, band_spect


def estimate_snr_framestride(sig, fps=90, width=20, stride=1, min_hz=0.66667, max_hz=4, nfft=1024):
    N = len(sig)
    window_width = int(fps*width)
    snrs = []
    for i in range(0, N-window_width+1, stride):
      sig_window = sig[i:(i+window_width)]
      ac_window = sig_window - np.mean(sig_window)
      f, amp_spect = signal.periodogram(sig, fs=fps, window='hamming', nfft=nfft, return_onesided=True, scaling='spectrum')
      snr, band_f, band_spect = snr_eval(ac_window, fps=fps, harm_width=6, min_hz=min_hz, max_hz=max_hz, nfft=nfft)
      snrs.append(snr)
    snrs = np.array(snrs)
    return snrs


def estimate_hr_framestride(sig, fps=90, width=20, stride=1, min_hz=0.66667, max_hz=4, nfft=1024):
    '''
            Returns: Heart rate prediction with the same size as
                     the input signal.
    '''
    n = sig.shape[0]
    window_size = int(width*fps)
    window_count = ((n - window_size) / stride) + 1
    start_idcs = (stride * np.arange(window_count)).astype(int)
    end_idcs = start_idcs + window_size

    HRs = []
    for s_idx in start_idcs:
        e_idx = s_idx + window_size
        sig_window = sig[s_idx:e_idx]
        sig_window = standardize(sig_window)
        HR = estimate_bpm(sig_window, fps, min_hz, max_hz, nfft)
        HRs.append(HR)

    HRs = resize_to_input(HRs, n, stride, window_size)
    assert(HRs.shape[0] == sig.shape[0])
    return HRs


def resize_to_input(HRs, n, stride, window_size):
    HRs = np.asarray(HRs)
    HRs = np.repeat(HRs, stride)
    diff = n - HRs.shape[0]
    pad = int(diff / 2)
    first_win = np.repeat(HRs[0], pad)
    last_win = np.repeat(HRs[-1], pad + (diff % 2 == 1))
    HRs = np.hstack((first_win, HRs, last_win))
    return HRs


def standardize(trace):
    trace = (trace - np.mean(trace, axis=0)) / np.std(trace, axis=0)
    return trace


def estimate_bpm(sig, fps, low_hz, high_hz, nfft):
    window = signal.get_window('hamming', len(sig))
    freq, density = signal.periodogram(sig, window=window, fs=fps, nfft=nfft)
    idcs = np.where((freq >= low_hz) & (freq <= high_hz))[0]
    freq = freq[idcs]
    density = density[idcs]
    pulse_freq = freq[np.argmax(density)]
    HR = pulse_freq * 60
    return HR


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--K',
                        help='Which model initialization to use [1-5].')
    parser.add_argument('--task_id',
                        help='ID for the task array index.',
                        default=0,
                        type=int)
    parser.add_argument('--task_count',
                        help='Number of tasks in array.',
                        default=1,
                        type=int)
    parser.add_argument('--testing_dataset',
                        help='Dataset to process.')
    parser.add_argument('--debug',
                        default=False,
                        help='Are we in debugging mode to decrease dataset size?')
    args = parser.parse_args()
    main(args)

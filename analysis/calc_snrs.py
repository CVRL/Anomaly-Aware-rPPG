import numpy as np
import scipy.signal as signal
import os
import sys

def main():
    number = int(sys.argv[1]) - 1
    root = 'outputs'
    width = 30
    file = sorted(os.listdir(root))[number]
    print(file)

    outdir = f'snrs/snrs_{width}'
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    outfile = os.path.join(outdir, os.path.splitext(file)[0] + '.npy')
    print(outfile)
    if os.path.exists(outfile):
        print('File already exists. Exiting.')
        return

    splits = file.split('_')
    train_db = splits[1][2:]
    loss_type = splits[2][4:]
    eval_db = splits[-2][4:]

    pos_flag = eval_db == 'ddpm'

    file_path = os.path.join(root, file)
    data = np.load(file_path, allow_pickle=True)
    pred_waves = data['oadd_preds']
    gt_waves = data['gt_wave']
    snrs = []
    for i in range(len(pred_waves)):
        pred_wave = pred_waves[i]
        t = np.arange(len(pred_wave))/90
        print('wave:', pred_wave.shape)
        snr = estimate_snr_framestride(pred_wave, fps=90, width=width, stride=1, min_hz=0.66667, max_hz=4, nfft=5400)
        print('snr:', snr.shape)
        snrs.append(snr)

    np.save(outfile, np.array(snrs))


def snr_eval(sig, fps=90, harm_width=6, min_hz=0.66667, max_hz=4, nfft=1024):
    '''
    harm_width is in bpm.
    '''
    f, amp_spect = signal.periodogram(sig, fs=fps, window='hann', nfft=nfft, return_onesided=True, scaling='spectrum')
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
    left_width = window_width // 2
    right_width = window_width // 2
    if window_width % 2 == 1:
      left_width += 1
    pad_sig = np.pad(sig, left_width)
    snrs = []
    for i in range(left_width, left_width+N, stride):
      sig_window = pad_sig[(i-left_width):(i+right_width)]
      snr, band_f, band_spect = snr_eval(sig_window, fps=fps, harm_width=6, min_hz=min_hz, max_hz=max_hz, nfft=nfft)
      #print(band_f*60)
      snrs.append(snr)
    snrs = np.array(snrs)
    return snrs


if __name__ == '__main__':
    main()

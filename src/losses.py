import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
import numpy as np
import sys


def select_loss(loss_type):
    if loss_type == 'mse':
        criterion = mse
    elif loss_type == 'np':
        criterion = negpearson
    elif loss_type == 'specentropy':
        criterion = spectral_entropy
    elif loss_type == 'specflatness':
        criterion = spectral_flatness
    elif loss_type == 'npentropy':
        criterion = negpearson_entropy
    elif loss_type == 'npflatness':
        criterion = negpearson_flatness
    elif loss_type == 'bce':
        criterion = nn.BCELoss()
    elif loss_type == 'deviation':
        criterion = deviation
    elif loss_type == 'deviationmargin':
        criterion = deviation_margin
    else:
        print('Could not find loss function specified.')
        sys.exit(-1)
    return criterion


def mse(x, y, y_live=None, nfft=None):
    error = torch.mean(torch.square(x - y))
    return error


def negpearson(x, y, y_live=None, nfft=None, epsilon=1e-6):
    if len(x.shape) < 2:
        x = torch.reshape(x, (1,-1))
    mean_x = torch.mean(x, 1)
    mean_y = torch.mean(y, 1)
    xm = x.sub(mean_x[:, None])
    ym = y.sub(mean_y[:, None])
    r_num = torch.einsum('ij,ij->i', xm, ym)
    r_den = torch.norm(xm, 2, dim=1) * torch.norm(ym, 2, dim=1) + epsilon
    r_vals = r_num / r_den
    r_val = torch.mean(r_vals)
    return 1 - r_val


def negpearson_entropy(x, y, y_live=None, nfft=544, fps=90):
    negpea = negpearson(x, y)
    se = torch_spectral_entropy(x, nfft, fps)
    return negpea + se


def negpearson_flatness(x, y, y_live=None, nfft=544, fps=90):
    negpea = negpearson(x, y)
    sf = torch_spectral_flatness(x, nfft, fps)
    return negpea + sf


def deviation(x, y, y_live, nfft=None):
    return torch.mean(y_live*negpearson(x, y) + (1-y_live)*torch_std(x))


def deviation_margin(x, y, y_live, nfft=None):
    return torch.mean(y_live*(negpearson(x, y) + torch.square(1-torch_std(x))) + (1-y_live)*torch_std(x))


def spectral_entropy(x, y, y_live, nfft=544, fps=90):
    return torch.mean(y_live*negpearson(x, y) + (1-y_live)*torch_spectral_entropy(x, nfft, fps))


def spectral_flatness(x, y, y_live, nfft=544, fps=90):
    return torch.mean(y_live*negpearson(x, y) + (1-y_live)*torch_spectral_flatness(x, nfft, fps))


def torch_std(x):
    if len(x.shape) < 2:
        x = torch.reshape(x, (1,-1))
    return torch.std(x, dim=1)


def torch_power_spectral_density(x, nfft=544, fps=90, low_hz=0.66, high_hz=4):
    centered = x - torch.mean(x, keepdim=True, dim=1)
    psd = torch.abs(fft.rfft(centered, n=nfft, dim=1))**2
    N = psd.shape[1]
    freqs = np.fft.rfftfreq(2*N-1, 1/fps)
    freq_idcs = np.logical_and(freqs >= low_hz, freqs <= high_hz)
    freqs = freqs[freq_idcs]
    psd = psd[:,freq_idcs]
    psd = psd / torch.sum(psd, keepdim=True, dim=1) ## treat as probabilities
    return freqs, psd


def np_power_spectral_density(x, nfft=544, fps=90, low_hz=0.66, high_hz=4):
    centered = x - np.mean(x, keepdims=True, axis=1)
    psd = np.abs(np.fft.rfft(x, n=nfft, axis=1))**2
    N = psd.shape[1]
    freqs = np.fft.rfftfreq(2*N-1, 1/fps)
    freq_idcs = np.logical_and(freqs >= low_hz, freqs <= high_hz)
    freqs = freqs[freq_idcs]
    psd = psd[:,freq_idcs]
    psd = psd / np.sum(psd, keepdims=True, axis=1) ## treat as probabilities
    return freqs, psd


def torch_spectral_entropy(x, nfft=544, fps=90, epsilon=0.000001):
    freqs, psd = torch_power_spectral_density(x, nfft, fps)
    N = psd.shape[1]
    H = - torch.sum(psd*torch.log2(psd+epsilon), dim=1) ## entropy
    H_n =  H / np.log2(N) ## normalized entropy
    batch_H_n = torch.mean(H_n)
    return 1 - batch_H_n


def torch_spectral_flatness(x, nfft=544, fps=90, epsilon=0.000001):
    freqs, psd = torch_power_spectral_density(x, nfft, fps)
    N = psd.shape[1]
    norm = 1 / N
    geo = torch.exp(norm * torch.sum(torch.log(psd+epsilon), dim=1))
    arith = torch.mean(psd, dim=1)
    sf = geo / arith
    batch_sf = torch.mean(sf)
    return 1 - batch_sf


def np_spectral_entropy(x, nfft=544, fps=90, epsilon=0.000001):
    freqs, psd = np_power_spectral_density(x, nfft, fps)
    N = psd.shape[1]
    H = - np.sum(psd*np.log2(psd+epsilon), axis=1) ## entropy
    H_n =  H / np.log2(N) ## normalized entropy
    batch_H_n = np.mean(H_n)
    return 1 - batch_H_n


def np_spectral_flatness(x, nfft=544, fps=90, epsilon=0.000001):
    freqs, psd = np_power_spectral_density(x, nfft, fps)
    N = psd.shape[1]
    norm = 1 / N
    geo = np.exp(norm * np.sum(np.log(psd+epsilon), axis=1))
    arith = np.mean(psd, axis=1)
    sf = geo / arith
    batch_sf = np.mean(sf)
    return 1 - batch_sf


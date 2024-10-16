from scipy import signal
from scipy import fft
from scipy.signal import firwin
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning
from scipy.interpolate import CubicSpline
import sys

import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

class SignalProcessor():
        def __init__(self, arg_obj):
                # self.video_length = float(arg_obj.video_length)
                self.sampling_hz = float(arg_obj.fps)
                self.nfft_scalar = int(arg_obj.nfft_scalar)
                self.maf_width = int(arg_obj.maf_width)         # Moving average filter points
                self.hamming_width = int(arg_obj.hamming_width)
                self.butterworth_order = int(arg_obj.butterworth_order)
                # lmbda = (1, 2, 4, 10, 20, 50, 300) gives lower cutoff frequency of:
                #       sampling_frequency*(0.189, 0.132, 0.093, 0.059, 0.041, 0.025, 0.011)
                self.lmbda = int(arg_obj.lmbda)
                self.periodogram_window = arg_obj.periodogram_window
                # Window size is given in seconds, so we multiply by sampling rate
                self.window_size = int(int(arg_obj.window_size) * self.sampling_hz)
                self.stride = int(int(arg_obj.stride) * self.sampling_hz)
                self.low_hz = float(arg_obj.low_hz)
                self.high_hz = float(arg_obj.high_hz)

                print('Window size: ', self.window_size)
                print('Stride size: ', self.stride)

                self.Nyquist = self.sampling_hz / 2
                self.Nyquist_low = self.low_hz / self.Nyquist
                self.Nyquist_high = self.high_hz / self.Nyquist
                self.numtaps = 129



        def sliding_bpm(self, sig, window_size=None, stride=None, processed=False, method=None):
                '''
                        Returns: Heart rate prediction with the same size as
                                 the input signal.
                '''
                if window_size is None:
                        window_size = self.window_size

                if stride is None:
                        stride = self.stride

                n = sig.shape[0]
                window_count = ((n - window_size) / stride) + 1
                start_idcs = (stride * np.arange(window_count)).astype(int)
                end_idcs = start_idcs + window_size

                HRs = []
                for s_idx in start_idcs:
                        e_idx = s_idx + window_size
                        sig_window = sig[s_idx:e_idx]
                        if not processed:
                            if method == 'standardize':
                                sig_window = self.standardize(sig_window)
                            else:
                                sig_window = sig_window
                        HR = self.estimate_bpm(sig_window)
                        HRs.append(HR)

                HRs = self.resize_to_input(HRs, n, stride, window_size)
                assert(HRs.shape[0] == sig.shape[0])
                return HRs


        def resize_to_input(self, HRs, n, stride, window_size):
                HRs = np.asarray(HRs)
                HRs = np.repeat(HRs, stride)
                diff = n - HRs.shape[0]
                pad = int(diff / 2)
                first_win = np.repeat(HRs[0], pad)
                last_win = np.repeat(HRs[-1], pad + (diff % 2 == 1))
                HRs = np.hstack((first_win, HRs, last_win))
                return HRs


        def standardize(self, trace):
            trace = (trace - np.mean(trace, axis=0)) / np.std(trace, axis=0)
            return trace


        def clean_trace(self, trace):
            dirty = np.logical_or(np.isnan(trace), np.isinf(trace))
            dirty_idcs = np.where((dirty == True).all(axis=1))[0]
            clean_idcs = np.where((dirty == False).all(axis=1))[0]
            for dirty_idx in dirty_idcs:
                i = dirty_idx + 1
                try:
                    while dirty[i].all() == True:
                        i+=1
                    trace[dirty_idx] = trace[i]
                except:
                    trace[dirty_idx] = trace[clean_idcs[-1]]
            return trace


        ## An Advanced Detrending Method With Application to HRV Analysis - Tarvainen, 2002
        def detrend(self, trace):
                T = trace.shape[0]
                I = np.eye(T, dtype=int)
                filt = np.array([[1, -2, 1]]).T.dot(np.ones((1, T), dtype=int))
                D2 = sps.spdiags(filt, range(3), T-2, T)
                trace = (I - np.linalg.inv(I + self.lmbda**2 * D2.T.dot(D2))).dot(trace)
                trace = np.asarray(trace).reshape(T, -1)
                return trace



        def moving_average_filter(self, trace):
                trace = np.convolve(trace, np.ones(self.maf_width), 'same') / self.maf_width
                return trace



        def hamming_filter(self, trace):
                window = np.hamming(self.hamming_width)
                trace = np.convolve(trace, window, 'same')
                return trace


        def poh_hamming_filter(self, trace):
                b = firwin(self.numtaps, [self.Nyquist_low, self.Nyquist_high], window='hamming',
                           nyq=self.Nyquist, scale=False)
                bandpassed = signal.filtfilt(b, np.array([1]), trace)
                return bandpassed


        def butterworth_filter(self, trace):
                b, a = signal.butter(self.butterworth_order, [self.Nyquist_low, self.Nyquist_high],
                        'band', analog='False')
                trace = signal.lfilter(b, a, trace)
                return trace


        def bob_butterworth_filter(self, trace):
            b = firwin(self.numtaps, [self.Nyquist_low, self.Nyquist_high], pass_zero=False)
            bandpassed = signal.filtfilt(b, np.array([1]), trace)
            return bandpassed


        def cvpr_fir(self, trace):
            b = firwin(self.numtaps, [self.Nyquist_low, self.Nyquist_high], window='hamming', pass_zero='bandpass')
            bandpassed = signal.filtfilt(b, np.array([1]), trace)
            return bandpassed


        def estimate_bpm(self, sig):
                window = signal.get_window(self.periodogram_window, len(sig))
                freq, density = signal.periodogram(sig, window=window, fs=self.sampling_hz,
                        nfft=self.nfft_scalar*len(sig))

                idcs = np.where((freq >= self.low_hz) & (freq <= self.high_hz))[0]
                freq = freq[idcs]
                density = density[idcs]
                pulse_freq = freq[np.argmax(density)]

                HR = pulse_freq * 60

                return HR



import numpy as np
import math


def process_POS(signals, framerate=90):
    '''
        Runs the Plane-Orthogonal-to-Skin (POS) algorithm from a spatially-averaged BGR trace.
        For more details see: Wang et al. Algorithmic Principles of Remote-PPG. 2017.

        Arguments
            signals: [T,3] np.array where each time point is an [B,G,R] vector.
            framerate: sampling rate of video which was spatially averaged.
        Returns
            H: [T] np.array of the estimated waveform.
    '''

    #Convert to RGB
    signals = signals[:,[2,1,0]]

    #Calculating l
    l = int(framerate * 1.6)
    H = np.zeros(signals.shape[0])

    for t in range(0, (signals.shape[0]-l)):
        #t = 0
        # Step 1: Spatial averaging
        C = signals[t:t+l-1,:].T

        #Step 2 : Temporal normalization
        mean_color = np.mean(C, axis=1)
        diag_mean_color = np.diag(mean_color)
        try:
            diag_mean_color_inv = np.linalg.inv(diag_mean_color)
            Cn = np.matmul(diag_mean_color_inv,C)
        except: #usually all zeros
            Cn = C

        #Step 3: 
        projection_matrix = np.array([[0,1,-1],[-2,1,1]])
        S = np.matmul(projection_matrix,Cn)

        #Step 4: 2D signal to 1D signal
        S_std_numer = np.std(S[0,:])
        S_std_denom = np.std(S[1,:])
        if math.isclose(S_std_denom, 0):
            P_std = 1e-5

        std = np.array([1,S_std_numer/S_std_denom])
        P = np.matmul(std,S)

        #Step 5: Overlap-Adding
        P_std = np.std(P)
        if math.isclose(P_std, 0):
            P_std = 1e-5
        add_seg = (P-np.mean(P))/P_std
        H[t:t+l-1] = H[t:t+l-1] + add_seg

    return H



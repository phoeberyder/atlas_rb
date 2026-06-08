import numpy as np
from scipy.signal import find_peaks

def peakfindergeneral(x, y):
    '''
    Inputs:
        x: 1d array of lomb scargle frequencies
        y: 1d array of power of each frequency
        
    Outputs:
        peak_freqs
        peak_periods'''
    peaks = find_peaks(np.power, height=np.max(y)*0.1)
    peak_freqs = x[peaks[0]]
    print("Peak frequencies (Hz):", peak_freqs)
    peak_periods = 1/peak_freqs
    print("Corresponding periods (s):", peak_periods)
    return peak_freqs, peak_periods
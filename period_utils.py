import numpy as np
from scipy.signal import find_peaks

def peakfindergeneral(x, y, threshold):
    '''
    Inputs:
        x: 1d array of lomb scargle frequencies
        y: 1d array of power of each frequency
        threshold: factor of max power to check for peaks above
        
    Outputs:
        peak_freqs
        peak_periods'''
    peaks = find_peaks(y, height=np.max(y)*threshold)
    peak_freqs = x[peaks[0]]
    print("Peak frequencies (Hz):", peak_freqs)
    peak_periods = 1/peak_freqs
    print("Corresponding periods (s):", peak_periods)
    return peak_freqs, peak_periods
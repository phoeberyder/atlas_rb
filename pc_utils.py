import numpy as np

def rect(x):
    return np.where(np.abs(x)<=0.5, 1, 0)

def signal_samp_to_dB(power, signal):
    noise = np.median(power)
    snr = signal/noise
    signal_dB = 20*np.log10(snr)
    return signal_dB
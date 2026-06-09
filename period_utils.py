import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

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

def gaussian(x, A, mu, sigma):
    beamwidth = 0.106
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def single_gaussian_fitter(f, p, init_mu, init_sigma, target_name):
    
    # power_range = np.sum(data[:, start_range:end_range], axis=1)
    popt, pcov = curve_fit(gaussian, f, p, p0=[max(p), init_mu, init_sigma])
    A, mu, sigma = popt

    errors = np.sqrt(np.diag(pcov))
    mu_err = errors[1]
    # sigma_err = errors[2]

    x_fit = f
    y_fit = gaussian(f, *popt)
    print('Peak offset = ', mu, 'Hz')
    print('Error = ±', mu_err, 'Hz')
    print('Width = ', sigma, 'Hz')
    print('percentage error', sigma/f[np.argmax(p)]*100, '%')
    plt.plot(f, p, label = 'observation')
    plt.plot(x_fit, y_fit, label="Gaussian fit", color ='r')
    plt.xlim(0.05, 0.075)
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    plt.title(target_name)
    plt.legend()
    plt.show()
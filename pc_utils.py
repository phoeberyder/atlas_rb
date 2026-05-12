import numpy as np
from unpack_vdif import unpacksamps, sortframes, readframes, readheader
from schedule_utils import range_finder_general
from skyfield.api import load
import matplotlib.pyplot as plt
from scipy.signal.windows import blackman, blackmanharris, boxcar, kaiser

def rect(x):
    return np.where(np.abs(x)<=0.5, 1, 0)

def signal_samp_to_dB(power, signal):
    noise = np.median(power)
    snr = signal/noise
    signal_dB = 20*np.log10(snr)
    return signal_dB

def open_vdif(infilename, channel_pol=0):
    with open(infilename) as infile:
        header = readheader(infile)
        framedata, seconds, framenums, threads = readframes(infile, header)
    threaddata = sortframes(framedata, seconds, framenums, threads)
    pola = unpacksamps(threaddata[channel_pol,:], header['nbits'], header['dtype'])
    return pola, header

def iq_conversion(pola):
     # Convert to IQ    
    phase_4_samples = np.array([ 1+0j, 0-1j, -1+0j, 0+1j ])
    phase_all_samples = np.tile(phase_4_samples, pola.size//4+1)
    iq_samples = pola * phase_all_samples[:pola.size]
    return iq_samples

def pc_and_spectrogram(target, height, points, overlap_factor, iq_samples, samp_rate, pri, alpha, tle, telescope, Tp, freq, window_function):
    '''
    Inputs:
    height: CPI size in pulses
    points: number of samples per pulse
    overlap_factor: how much the CPIs should overlap (e.g. 2 means 50% overlap)
    iq_samples: the raw IQ samples from the VDIF file
    samp_rate: sampling rate of the data
    pri: pulse repetition interval
    alpha: chirp rate (bandwidth/pulse duration)
    tle: TLE data for the target satellite
    telescope: which telescope is being used (for range rate calculation)
    Tp: pulse duration
    freq: carrier frequency
    
    Returns:
    rcm_map: 2D array of pulse-compressed power values for each strip
    spectrogram: 2D array of micro-Doppler spectra for each strip
    peak_history: list of indices of the peaks in each strip'''
    c = 299792458
    ts = load.timescale()
    freqs = np.fft.fftfreq(points, d=1/samp_rate) # Used for Fourier Shift RCM correction
    startoffset = int(samp_rate * 100)

    cpi_jump_samples = (height * points) // overlap_factor
    number_of_strips = int((len(iq_samples) - startoffset - (height * points)) // cpi_jump_samples)

    rcm_map = np.zeros((points, number_of_strips))
    spectrogram = np.zeros((height, number_of_strips))
    peak_history = []

    print(f'Total strips to process: {number_of_strips}')


    for n in range(number_of_strips):
        print(f"Processing strip {n+1} of {number_of_strips}", end='\r')
        
        start_idx = startoffset + (n * cpi_jump_samples)
        end_idx = start_idx + (height * points)
        cpi_data = iq_samples[start_idx:end_idx].reshape((height, points))
        
        # Update TLE range rate for the start of this CPI
        s_offset = (start_idx / samp_rate)
        if target == 'intelsat':
            t_tle = ts.utc(2025, 2, 5, 13, 45, s_offset)
        elif target == 'atlas':
            t_tle = ts.utc(2026, 2, 18, 14, 50, s_offset)
        range_rate = range_finder_general(tle, t_tle, telescope)[1]
        tau_dot = range_rate / c
        
        # Create template: baseband * chirp * envelope
        t_pulse = np.linspace(0, pri, points, endpoint=False)
        template = np.exp(1j * np.pi * alpha * t_pulse**2 * (1 - tau_dot)**2) * rect(t_pulse / (Tp * (1 + tau_dot)))
        template_fft = np.fft.fft(template)

        cdat_pc = np.zeros((height, points), dtype=complex)
        
        # Pulse Compression
        for i in range(height):
            start_sec = i * pri
            pulse_fft = np.fft.fft(cpi_data[i])
            
            # Calculating RCM Correction
            delta_tau = 2 * (range_rate * start_sec) / c     # change in range rate since beginning of CPI, converted to time delay
            rcm_shift = np.exp(1j * 2 * np.pi * freqs * delta_tau)          #applying RCM correction in the frequency domain using linear phase shift corresponding to the change in range delay over the CPI duration
            
            # Pulse Compression and RCM shift
            compressed_pulse = np.fft.ifft(pulse_fft * np.conj(template_fft) * rcm_shift)
            
            # Doppler Phase Correction
            f_d = -2 * range_rate * freq / c
            bulk_phase = np.exp(-1j * 2 * np.pi * f_d * start_sec)
            
            # applying doppler correction
            cdat_pc[i] = compressed_pulse * bulk_phase
        
        # Average power
        cpi_power = np.mean(np.abs(cdat_pc)**2, axis=0)
        rcm_map[:, n] = cpi_power
        
        # Extract the stable range-cut for the spectrogram
        peak_idx = np.argmax(cpi_power)
        peak_history.append(peak_idx)
        range_cut = cdat_pc[:, peak_idx] 
        
        # Micro-Doppler (Slow-time FFT)
        if window_function == 'hanning':
            window = np.hanning(height)
        elif window_function == 'hamming':
            window = np.hamming(height)
        elif window_function == 'blackman':
            window = np.blackman(height)
        elif window_function == 'boxcar':  
            window = boxcar(height)
        elif window_function == 'kaiser':
            window = kaiser(height, beta=14)
        elif window_function == 'blackmanharris':
            window = blackmanharris(height)
        else:
            print("Window function sounds made up to me. Using Hanning window by default.")
            window = np.hanning(height)

        doppler_spectrum = np.fft.fftshift(np.fft.fft(range_cut * window))
        spectrogram[:, n] = np.abs(doppler_spectrum)**2
        return rcm_map, spectrogram, peak_history, number_of_strips
    
def plotter(to_be_plotted, title: str, xlabel: str, ylabel: str, extent, target_name: str, height: int, overlap_factor: float, channel: int, window_function: str):
    '''
    Inputs:
    to_be_plotted: 2D array of values to be plotted (e.g. RCM map or spectrogram)
    title: title of the plot (used for saving the file)
    xlabel: label for x-axis
    ylabel: label for y-axis
    extent: extent of the plot (used for spectrogram to set axes in physical units)
    target_name: name of the target satellite (used for saving the file)
    height: CPI size (used for saving the file)
    overlap_factor: how much the CPIs overlapped (used for saving the file)
    channel: the channel number (used for saving the file)
    window_function: the window function used (used for saving the file)
    Output:
    A saved plot of the input 2D array with appropriate labels and title.
    '''
    plt.figure(figsize=(10, 6))
    plt.imshow(10 * np.log10(to_be_plotted + 1e-12), aspect='auto', origin='lower', extent=extent, vmax=0, vmin=-25)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(label='Power (dB)')
    plt.tight_layout()
    plt.savefig('./'+title+ '_' + target_name + '_' + str(height) + '_'+str(height//overlap_factor)+'channel_'+str(channel)+'_'+str(window_function)+'.png')
    plt.close()
    return
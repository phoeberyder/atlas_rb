import numpy as np
from schedule_utils import range_finder_general
from skyfield.api import load, EarthSatellite
from pc_utils import rect
from pc_utils import open_vdif, iq_conversion
from scipy.signal.windows import tukey, blackmanharris, boxcar, kaiser

cpi = 128
channel = 1
infilename = '/share/nas2/pryder/SET_Observations_Test_1/Wednesday/vdifs/TSSat_20250205_lo1_1295MHz_intelsat33e.vdifc'
window_function = 'tukey'
zero_pad = 128

# inputting TLE data for intelsat 33e
tle_line_1 = '1 61995U 16053M   25035.62796428  .00000095  00000-0  00000-0 0  9998'
tle_line_2 = '2 61995   0.2642  88.1708 0053675 268.4745   3.2920  1.01033442  1106'
target_name = 'intelsat'
ts = load.timescale()
intelsat_tle = EarthSatellite(tle_line_1, tle_line_2, target_name, ts)
tle = intelsat_tle

# input parameters       
bw = 8e6        
Tp = 800e-6    
pri = 19.7e-3  #0.0197
freq = 1295e6
c = 299792458
samp_rate = 16e6
alpha = bw/Tp   
height =cpi # CPI Size
points = int(samp_rate * pri)    #315,200

# Open VDIF
pola, header = open_vdif(infilename, channel) #length is 4,277,760,000
print('VDIF opened.')

# Convert to IQ
iq_samples = iq_conversion(pola)
print('Converted into IQ samples.')

cpi_jump_samples = (height * points)         #40,345,600
number_of_strips = 106
target = 'intelsat'
telescope = 'lovell'

ts = load.timescale()
freqs = np.fft.fftfreq(points, d=1/samp_rate) # Used for Fourier Shift RCM correction
startoffset = 0#int(samp_rate * 100)
number_of_samples_in_whole_dataset = len(iq_samples) - startoffset
    

rcm_map = np.zeros((points, number_of_strips))

peak_history = []

print(f'Total strips to process: {number_of_strips}')
window_size = 128+zero_pad
spectrogram = np.zeros((window_size, number_of_strips))

for n in range(number_of_strips):
    print(f"Processing strip {n+1} of {number_of_strips}", end='\r')
    
    start_idx = startoffset + (n * cpi_jump_samples)
    end_idx = start_idx + (height * points)
    cpi_data = iq_samples[start_idx:end_idx].reshape((height, points))
    
    # Update TLE range rate for the start of this CPI
    s_offset = (start_idx / samp_rate)
    if target == 'intelsat':
        t_tle = ts.utc(2025, 2, 5, 14, 0, s_offset+1)
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
    
    lsv, S, rsv = np.linalg.svd(cdat_pc, full_matrices=False)
    S_cleaned = S.copy()
    number_dop_com = 2
    S_cleaned[:number_dop_com] = 0.0
    cleaned_cdat_pc = np.dot(lsv * S_cleaned, rsv)
    # Average power
    cpi_power = np.mean(np.abs(cleaned_cdat_pc)**2, axis=0)
    rcm_map[:, n] = cpi_power
    
    # Extract the stable range-cut for the spectrogram
    peak_idx = np.argmax(cpi_power)
    peak_history.append(peak_idx)
    # adding more consistency in range peak
    # _ = 128075, a = 192082, b = 27303, c = 179346
    expected_peak = 179346
    peak_offset = expected_peak- peak_idx
    # print(peak_offset)
    if np.abs(peak_offset)<10:
        # print('\n accepted')
        # accepted +=1
        range_cut = cleaned_cdat_pc[:, peak_idx] 
    else:
        # print('\n rejected')
        # rejected +=1
        range_cut = cleaned_cdat_pc[:, expected_peak] 

    #adding zero-padding
    # range_cut_padded = np.zeros(range_cut.shape[0]+zero_pad, dtype=complex)
    range_cut_padded = np.pad(range_cut, (zero_pad//2, zero_pad//2), mode='constant')

    
    print('window_size:', window_size)
    
    # Micro-Doppler (Slow-time FFT)
    if window_function == 'hanning':
        window = np.hanning(window_size)
    elif window_function == 'hamming':
        window = np.hamming(window_size)
    elif window_function == 'blackman':
        window = np.blackman(window_size)
    elif window_function == 'boxcar':  
        window = boxcar(window_size)
    elif window_function == 'kaiser':
        window = kaiser(window_size, beta=14)
    elif window_function == 'blackmanharris':
        window = blackmanharris(window_size)
    elif window_function == 'tukey':
        window = tukey(window_size, alpha=0.5)
    else:
        print("Window function sounds made up to me. Using Hanning window by default.")
        window = np.hanning(window_size)

    # print('window size:', len(window))
    doppler_spectrum = np.fft.fftshift(np.fft.fft(range_cut_padded * window))
    
    spectrogram[:, n] = np.abs(doppler_spectrum)**2

np.save('./spectrogram_c_intelsat_128cpi_1sampoverlap_tukey_128pad_106_strips_pol1.npy', spectrogram)
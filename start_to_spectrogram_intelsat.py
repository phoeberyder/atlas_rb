from unpack_vdif import unpacksamps, sortframes, readframes, readheader
import numpy as np
import matplotlib.pyplot as plt
from schedule_utils import range_finder_general
from skyfield.api import load, EarthSatellite
from pc_utils import rect

# inputting TLE data for intelsat 33e
tle_line_1 = '1 61995U 16053M   25035.62796428  .00000095  00000-0  00000-0 0  9998'
tle_line_2 = '2 61995   0.2642  88.1708 0053675 268.4745   3.2920  1.01033442  1106'
ts = load.timescale()
intelsat_tle = EarthSatellite(tle_line_1, tle_line_2, 'intelsat 33e', ts)

# input parameters
f1 = 0        
bw = 8e6        
Tp = 800e-6    
pri = 20.506e-3 
freq = 1295e6
c = 299792458
samp_rate = 16e6
alpha = bw/Tp   
height = 128 # CPI Size
startoffset = int(samp_rate * 100)
infilename = '/share/nas2/pryder/SET_Observations_Test_1/Wednesday/vdifs/TSSat_20250205_lo1_1295MHz_intelsat33e.vdif'
channel_pol = 0

# Open VDIF
with open(infilename) as infile:
    header = readheader(infile)
    framedata, seconds, framenums, threads = readframes(infile, header)
threaddata = sortframes(framedata, seconds, framenums, threads)
pola = unpacksamps(threaddata[channel_pol,:], header['nbits'], header['dtype'])
print('VDIF opened.')

# Convert to IQ
phase_4_samples = np.array([ 1+0j, 0-1j, -1+0j, 0+1j ])
phase_all_samples = np.tile(phase_4_samples, pola.size//4+1)
iq_samples = pola * phase_all_samples[:pola.size]
print('Converted into IQ samples.')

points = int(samp_rate * pri)
freqs = np.fft.fftfreq(points, d=1/samp_rate) # Used for Fourier Shift RCM correction

# Overlapping Windows for a smoother Spectrogram
overlap_factor = 2
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
    t_tle = ts.utc(2025, 2, 5, 13, 45, s_offset)
    range_rate = range_finder_general(intelsat_tle, t_tle, 'lovell')[1]
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
    window = np.hanning(height)
    doppler_spectrum = np.fft.fftshift(np.fft.fft(range_cut * window))
    spectrogram[:, n] = np.abs(doppler_spectrum)**2

print('\nProcessing Complete.')

# --- Plotting ---
# Crop the RCM map so Matplotlib doesn't alias the track to oblivion
min_peak = max(0, np.min(peak_history) - 200)
max_peak = min(points, np.max(peak_history) + 200)
cropped_rcm = rcm_map[min_peak:max_peak, :]

plt.figure(figsize=(10, 6))
plt.imshow(10 * np.log10(cropped_rcm + 1e-12), aspect='auto', origin='lower', extent=[0, number_of_strips, min_peak, max_peak])
plt.title("Range-Time Migration")
plt.xlabel("CPI Index")
plt.ylabel("Range Bin")
plt.colorbar(label='Power (dB)')
plt.tight_layout()
plt.savefig('./rcm_intelsat.png')

max_velocity = c / (4 * pri * (freq))

plt.figure(figsize=(10, 6))
plt.imshow(10 * np.log10(spectrogram + 1e-12), aspect='auto', origin='lower', extent=[0, spectrogram.shape[1], -max_velocity, max_velocity])
plt.title("Micro-Doppler Spectrogram")
plt.xlabel("CPI Index")
plt.ylabel("Doppler Velocity (m/s)")
plt.colorbar(label='Power (dB)')
plt.tight_layout()
plt.savefig('./spectrogram_intelsat.png')

np.save('./processed_data_rcm_intelsat.npy', cropped_rcm)
np.save('./processed_data_spectrogram_intelsat.npy', spectrogram)


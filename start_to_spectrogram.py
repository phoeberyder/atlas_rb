from unpack_vdif import unpacksamps, sortframes, readframes, readheader
import numpy as np
import matplotlib.pyplot as plt
from schedule_utils import range_finder_general
from skyfield.api import load, EarthSatellite
from pc_utils import rect

# --- Input Parameters ---
tle_line_1 = '1 40731U 15033B   26046.96407114  .00000009  00000-0  00000+0 0  9995'
tle_line_2 = '2 40731  54.6696 288.5525 0237286   9.2718 351.2385  1.90866878 73834'
ts = load.timescale()
atlas_tle = EarthSatellite(tle_line_1, tle_line_2, 'atlas', ts)

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
infilename = '/share/nas2/pryder/realtime_test_1/vdifs/SD20003_20260218_mk2_1295MHz_atlasrb.vdif'

# --- Open and Unpack VDIF ---
with open(infilename) as infile:
    header = readheader(infile)
    framedata, seconds, framenums, threads = readframes(infile, header)
threaddata = sortframes(framedata, seconds, framenums, threads)
pola = unpacksamps(threaddata[0,:], header['nbits'], header['dtype'])
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

# --- Processing Loop ---
for n in range(number_of_strips):
    print(f"Processing strip {n+1} of {number_of_strips}", end='\r')
    
    start_idx = startoffset + (n * cpi_jump_samples)
    end_idx = start_idx + (height * points)
    cpi_data = iq_samples[start_idx:end_idx].reshape((height, points))
    
    # Update TLE range rate for the start of this CPI
    s_offset = (start_idx / samp_rate)
    t_tle = ts.utc(2026, 2, 18, 14, 50, s_offset)
    range_rate = range_finder_general(atlas_tle, t_tle, 'mark')[1]
    tau_dot = range_rate / c
    
    # Create Coherent Template
    t_pulse = np.linspace(0, pri, points, endpoint=False)
    template = np.exp(1j * np.pi * alpha * t_pulse**2 * (1 - tau_dot)**2) * rect(t_pulse / (Tp * (1 + tau_dot)))
    template_fft = np.fft.fft(template)

    cdat_pc = np.zeros((height, points), dtype=complex)
    
    # Pulse Compression with Intra-CPI Corrections
    for i in range(height):
        t_i = i * pri
        pulse_fft = np.fft.fft(cpi_data[i])
        
        # 1. RCM Correction (Fourier Shift Theorem)
        # Shift the pulse back by the distance the target moved since the start of the CPI
        delta_tau = 2 * (range_rate * t_i) / c
        rcm_shift = np.exp(1j * 2 * np.pi * freqs * delta_tau)
        
        # Apply matched filter and RCM shift simultaneously
        compressed_pulse = np.fft.ifft(pulse_fft * np.conj(template_fft) * rcm_shift)
        
        # 2. Bulk Doppler Phase Correction
        # Remove the massive phase rotation caused by orbital velocity to center target at 0 Hz
        f_d = -2 * range_rate * freq / c
        bulk_phase = np.exp(-1j * 2 * np.pi * f_d * t_i)
        
        cdat_pc[i] = compressed_pulse * bulk_phase
    
    # Average power across the aligned pulses
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
plt.title("Figure 4: Range-Time Migration")
plt.xlabel("CPI Index")
plt.ylabel("Range Bin")
plt.colorbar(label='Power (dB)')
plt.tight_layout()
plt.savefig('./figure4_rcm.png')

plt.figure(figsize=(10, 6))
plt.imshow(10 * np.log10(spectrogram + 1e-12), aspect='auto', origin='lower')
plt.title("Figure 5: Micro-Doppler Spectrogram")
plt.xlabel("CPI Index")
plt.ylabel("Doppler Bin")
plt.colorbar(label='Power (dB)')
plt.tight_layout()
plt.savefig('./figure5_spectrogram.png')

np.save('./processed_data_rcm.npy', rcm_map)
np.save('./processed_data_spectrogram.npy', spectrogram)


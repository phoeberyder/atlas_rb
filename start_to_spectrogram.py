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
height = 128 # CPI Size (Number of pulses)
startoffset = int(samp_rate * 100)
infilename = '/share/nas2/pryder/realtime_test_1/vdifs/SD20003_20260218_mk2_1295MHz_atlasrb.vdif'

# --- Open and Unpack VDIF ---
with open(infilename) as infile:
    header = readheader(infile)
    framedata, seconds, framenums, threads = readframes(infile, header)
threaddata = sortframes(framedata, seconds, framenums, threads)
pola = unpacksamps(threaddata[0,:], header['nbits'], header['dtype'])

# Convert to IQ
phase_4_samples = np.array([ 1+0j, 0-1j, -1+0j, 0+1j ])
phase_all_samples = np.tile(phase_4_samples, pola.size//4+1)
iq_samples = pola * phase_all_samples[:pola.size]

points = int(samp_rate * pri)
# Calculate how many CPIs we can fit
number_of_strips = int((len(iq_samples) - startoffset) // (height * points))

# Initialize storage
# Fig 4: Range-Time Map (Power)
rcm_map = np.zeros((points, number_of_strips)) 
# Fig 5: Spectrogram (Power)
spectrogram = np.zeros((height, number_of_strips))

# --- Processing Loop ---
for n in range(number_of_strips):
    print(f"Processing strip {n+1} of {number_of_strips}")
    
    # 1. Extract CPI data
    start_idx = startoffset + (n * height * points)
    end_idx = start_idx + (height * points)
    cpi_data = iq_samples[start_idx:end_idx].reshape((height, points))
    
    # 2. Update TLE range rate for this specific time
    s_offset = (n * height * pri)
    t_tle = ts.utc(2026, 2, 18, 14, 50, s_offset)
    range_rate = range_finder_general(atlas_tle, t_tle, 'mark')[1]
    tau_dot = range_rate / c
    
    # 3. Create Coherent Template
    t_pulse = np.linspace(0, pri, points, endpoint=False)
    template = np.exp(1j * np.pi * alpha * t_pulse**2 * (1 - tau_dot)**2) * \
               rect(t_pulse / (Tp * (1 + tau_dot)))
    template_fft = np.fft.fft(template)

    # 4. Pulse Compression (Stay Complex!)
    cdat_pc = np.zeros((height, points), dtype=complex)
    for i in range(height):
        pulse_fft = np.fft.fft(cpi_data[i])
        # Matched filter in freq domain
        cdat_pc[i] = np.fft.ifft(pulse_fft * np.conj(template_fft))
    
    # 5. Figure 4 Logic: Range-Time Migration
    # Average power across the pulses in this CPI to find the target range
    cpi_power = np.mean(np.abs(cdat_pc)**2, axis=0)
    rcm_map[:, n] = cpi_power
    
    # 6. Figure 5 Logic: Micro-Doppler (Slow-time FFT)
    # Find the target peak in range to extract the Doppler "cut"
    peak_idx = np.argmax(cpi_power)
    # Extract the complex range-bin across all pulses in the CPI
    range_cut = cdat_pc[:, peak_idx] 
    
    # Apply window and FFT across the pulses (Slow-time)
    window = np.hanning(height)
    doppler_spectrum = np.fft.fftshift(np.fft.fft(range_cut * window))
    spectrogram[:, n] = np.abs(doppler_spectrum)**2

# --- Plotting ---
plt.figure(figsize=(10, 6))
plt.imshow(10 * np.log10(rcm_map + 1e-12), aspect='auto', origin='lower')
plt.title("Figure 4: Range-Time Migration")
plt.xlabel("CPI Index (Time)")
plt.ylabel("Range Bin")
plt.colorbar(label='dB')
plt.savefig('./figure4_rcm.png')

plt.figure(figsize=(10, 6))
plt.imshow(10 * np.log10(spectrogram + 1e-12), aspect='auto', origin='lower')
plt.title("Figure 5: Micro-Doppler Spectrogram")
plt.xlabel("CPI Index (Time)")
plt.ylabel("Doppler Bin (Velocity)")
plt.colorbar(label='dB')
plt.savefig('./figure5_spectrogram.png')

plt.show()
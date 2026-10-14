import os
file_path = os.path.realpath(__file__)
print("Current file path:", file_path)

import numpy as np
import matplotlib.pyplot as plt
from schedule_utils import range_finder_general
from skyfield.api import load, EarthSatellite, wgs84
from pc_utils import rect, open_vdif, iq_conversion, pc_and_spectrogram, plotter
from scipy.ndimage import shift

tle_line_1 = '1 61995U 16053M   25035.62796428  .00000095  00000-0  00000-0 0  9998'
tle_line_2 = '2 61995   0.2642  88.1708 0053675 268.4745   3.2920  1.01033442  1106'
target_name = 'intelsat'
ts = load.timescale()
intelsat_tle = EarthSatellite(tle_line_1, tle_line_2, target_name, ts)

# input parameters 
cpi = 128      
bw = 8e6        
Tp = 800e-6    
pri = 19.7e-3
freq = 1295e6
c = 299792458
samp_rate = 16e6
alpha = bw/Tp   
height =cpi # CPI Size
points = int(samp_rate * pri)
infilename = '/share/nas2/pryder/SET_Observations_Test_1/Wednesday/vdifs/TSSat_20250205_lo1_1295MHz_intelsat33e.vdifa' #second vdif
channel = 0
start_strip = 2400

# Open VDIF
pola, header = open_vdif(infilename, channel)
print('VDIF opened.')

# Convert to IQ
iq_samples = iq_conversion(pola)
print('Converted into IQ samples.')

# what's being recorded
peak_index = []
peak_power = []
average_noise = []
tle_range_bin = []

max_velocity = c / (4 * 19.7e-3* (freq))

ts = load.timescale()
freqs = np.fft.fftfreq(points, d=1/samp_rate) # Used for Fourier Shift RCM correction
startoffset = 0 #int(samp_rate * 100)

overlap = points
rough_period = 15.6
time_of_one_cpi = pri*cpi #in seconds
points_of_one_period = rough_period/time_of_one_cpi * points
number_of_strips = 800 # roughly number of strips in one period
print(number_of_strips)
spectrogram = np.zeros((height*2, number_of_strips))
strip_time = np.linspace(0, number_of_strips*overlap/samp_rate, number_of_strips)

#adding an initial idea of where the signal is going to be based on the tle
bin_spacing_m = c/(samp_rate)
R_tle = np.zeros(height)
target = intelsat_tle
for i in range(height):
        start_idx = start_strip+(i*overlap)
        s_offset = (start_idx / samp_rate) + 1.658 + (i * pri)
        t = ts.utc(2025, 2, 5, 13, 50, s_offset)
        sat_pos = target.at(t).position.km

        # Transmitter and receiver locations
        mhr = wgs84.latlon(42.6, 288.5)
        rx = wgs84.latlon(53.2365, -2.3087)
        tx_pos = mhr.at(t).position.km
        rx_pos = rx.at(t).position.km

        difference_rx = (target - rx)
        difference_tx = (target - mhr)

        topocentric_rx = (difference_rx.at(t))
        topocentric_tx = (difference_tx.at(t))

        range_rx_m = topocentric_rx.distance().m
        range_tx_m = topocentric_tx.distance().m
        range_m = range_rx_m + range_tx_m
        R_tle[i] = range_m

for n in range(number_of_strips):
    print(n)
    start_idx = (n*overlap) + start_strip
    end_idx = start_idx + (height * points)
    cpi_data = iq_samples[start_idx:end_idx].reshape((height, points))

    # Update TLE range rate for the start of this CPI
    s_offset = (start_idx / samp_rate) + 1.658 #offset from 50 from start of file
    t = ts.utc(2025, 2, 5, 13, 50, s_offset)
    sat_pos = target.at(t).position.km

    # Transmitter and receiver locations
    mhr = wgs84.latlon(42.6, 288.5)
    rx = wgs84.latlon(53.2365, -2.3087)
    tx_pos = mhr.at(t).position.km
    rx_pos = rx.at(t).position.km

    difference_rx = (target - rx)
    difference_tx = (target - mhr)

    topocentric_rx = (difference_rx.at(t))
    topocentric_tx = (difference_tx.at(t))

    range_rx_m = topocentric_rx.distance().m
    range_tx_m = topocentric_tx.distance().m
    range_m = range_rx_m + range_tx_m
    tau = range_m/c
    range_bin = int(np.floor(range_m/(2*bin_spacing_m)))
    tle_range_bin.append(range_bin)
    delay_samples = tau * samp_rate
    print('approx range bin: ', range_bin)

    p = np.polyfit(np.linspace(0, pri, height), R_tle, 2)
    R_fit = np.polyval(p, np.linspace(0, pri, height))
    dR = R_fit - R_fit[0]
    d_bins = dR / (2*bin_spacing_m)

    _, _, the_range, _, _, range_rate = topocentric_rx.frame_latlon_and_rates(rx)
    _, _, the_range_tx, _, _, range_rate_tx = topocentric_tx.frame_latlon_and_rates(mhr)
    range_rate = range_rate_tx.m_per_s/2 + range_rate.m_per_s/2
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
        # rowshift = range_rate*np.arange(height)
        
        # # Calculating RCM Correction
        # delta_tau = 2 * (range_rate * start_sec) / c     # change in range rate since beginning of CPI, converted to time delay
        # rcm_shift = np.exp(1j * 2 * np.pi * freqs * delta_tau)          #applying RCM correction in the frequency domain using linear phase shift corresponding to the change in range delay over the CPI duration
        
        # # Pulse Compression and RCM shift
        compressed_pulse = np.fft.ifft(pulse_fft * np.conj(template_fft))
        
        # # Doppler Phase Correction
        # f_d = -2 * range_rate * freq / c
        # bulk_phase = np.exp(-1j * 2 * np.pi * f_d * start_sec)
        cdat_pc[i] = shift(compressed_pulse, -d_bins[i], mode='constant', order=1)
    
    # cpi_power = np.mean(np.abs(cdat_pc)**2, axis=0)
    power = (np.abs(cdat_pc))**2
    doppler_trimmed_power = power[42:86, :] #removing anything below -1 m/s and above 1 m/s
    incoho_sum = np.sum(doppler_trimmed_power, axis=0)
    peak_idx = np.argmax(incoho_sum)
    peak_index.append(peak_idx)
    peak_power.append(incoho_sum[peak_idx])
    average_noise.append(np.median(incoho_sum))

# plt.plot(strip_time, peak_power, label='Peak Power', color='blue')
# plt.plot(strip_time, average_noise, label='Average Noise', color='orange')
# plt.xlabel('Time (s)')
# plt.ylabel('Power')
# plt.title('Peak Power and Average Noise Over Time')
# plt.legend()
# plt.savefig('/share/nas2/pryder/tumbling_git/Range rate compensation/peak_power_average_noise_over_time.png')
# plt.clf()

# plt.plot(strip_time, peak_index, label='Peak Index', color='green')
# plt.xlabel('Time (s)')
# plt.ylabel('Peak Index')
# plt.legend()
# plt.savefig('/share/nas2/pryder/tumbling_git/Range rate compensation/peak_index_over_time.png')
# plt.clf()


# plt.plot(strip_time, tle_range_bin, label='TLE Range Bin', color='red')
# plt.xlabel('Time (s)')
# plt.ylabel('TLE Range Bin')
# plt.legend()
# plt.savefig('/share/nas2/pryder/tumbling_git/Range rate compensation/tle_range_bin_over_time.png')



np.save('/share/nas2/pryder/tumbling_git/Range rate compensation/peak_index_over_time_'+str(start_strip)+'.npy', peak_index)
np.save('/share/nas2/pryder/tumbling_git/Range rate compensation/peak_power_over_time_'+str(start_strip)+'.npy', peak_power)
np.save('/share/nas2/pryder/tumbling_git/Range rate compensation/average_noise_over_time_'+str(start_strip)+'.npy', average_noise)

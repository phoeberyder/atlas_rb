import numpy as np
import matplotlib.pyplot as plt
from skyfield.api import load, EarthSatellite
from pc_utils import range_finder_general, rect
from scipy.constants import c
from datetime import datetime, timedelta
import scipy.signal as signal

# input parameters       
bw = 8e6        
Tp = 800e-6    
pri = 19.7e-3
freq = 1295e6
c = 299792458
samp_rate = 16e6
alpha = bw/Tp   
height = 128 # CPI Size
points = int(samp_rate * pri)

tle_line_1 = '1 61995U 16053M   25035.62796428  .00000095  00000-0  00000-0 0  9998'
tle_line_2 = '2 61995   0.2642  88.1708 0053675 268.4745   3.2920  1.01033442  1106'
target_name = 'intelsat'
ts = load.timescale()
intelsat_tle = EarthSatellite(tle_line_1, tle_line_2, target_name, ts)
tle_list = [target_name, tle_line_1, tle_line_2]



iq_samples = np.load('/share/nas2/pryder/tumbling_git/full_obs_iq.npy')
ts = load.timescale()
freqs = np.fft.fftfreq(points, d=1/samp_rate) # Used for Fourier Shift RCM correction
startoffset = 0 #int(samp_rate * 10)
overlap_factor = 'min'
rejected = 0
accepted = 0

if overlap_factor == 'max':
    cpi_jump_samples = 1
    number_of_strips = 200
    overlap_factor = 1

elif overlap_factor == 'min':
    cpi_jump_samples = height * points
    number_of_strips = int((len(iq_samples) - startoffset - (height * points)) // cpi_jump_samples)

else:
    cpi_jump_samples = (height * points) // overlap_factor
    number_of_strips = int((len(iq_samples) - startoffset - (height * points)) // cpi_jump_samples)

rcm_map = np.zeros((points, number_of_strips))
spectrogram = np.zeros((height, number_of_strips))

peak_history = []

print(f'Total strips to process: {number_of_strips}')


for n in range(number_of_strips):
    # n=n+15
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

    # power = np.abs(cdat_pc)**2
    # incoho_sum = np.sum(power, axis=0)
    # plt.plot(incoho_sum)
    # plt.xlabel('Range (km)')
    # plt.ylabel('Incoherent Sum Power')
    # plt.savefig(f'./plots/incoherent_sum_strip_{n+1}.png')
    # plt.show()

    # plt.imshow(power[:, 127000:129000], aspect='auto',vmin=0, vmax=5000, cmap='plasma', extent=[127000, 129000, 0, height*pri])
    # plt.colorbar(label='Power')
    # plt.xlabel('Range (km)')
    # plt.ylabel('Time (s)')
    # plt.savefig(f'./plots/rti_{n+1}.png')
    # plt.show()
    # cdat_pc_trim = cdat_pc[:, 127075:129075]
    power = np.abs(cdat_pc)**2
    incoho_sum = np.sum(power, axis=0)
    peak = np.argmax(incoho_sum)


    # using single value decomposition (svd) theorem to split cdat_pc matrix into three components
    # left singular values -> slow time profiles (orthogonal matrix)
    # singular values - > strength of each component (diagonal matrix)
    # right singular values -> fast time profiles (conjugate transpose of orthogonal matrix)

    lsv, S, rsv = np.linalg.svd(cdat_pc, full_matrices=False)
    S_cleaned = S.copy()
    number_dop_com = 2
    S_cleaned[:number_dop_com] = 0.0
    cleaned_cdat_pc = np.dot(lsv * S_cleaned, rsv)

    win_fast = signal.get_window('hann', points, fftbins=False)
    win_slow = signal.get_window('boxcar', height, fftbins=False)
    win_fast /= np.mean(win_fast)
    win_slow /= np.mean(win_slow)
    windowed_cdat_pc = cleaned_cdat_pc * win_slow[:, np.newaxis]



    unshifted_range_doppler = np.fft.fft(windowed_cdat_pc, axis=0)
    range_doppler = np.fft.fftshift(unshifted_range_doppler, axes=0)

    # max_velocity = c / (4 * pri * (freq))

    # lowest_value = -max_velocity

    # number_of_points = range_doppler.shape[0]
    # velocity_axis = np.linspace(lowest_value, max_velocity, number_of_points)
    # range_axis = np.linspace(0, pri * c / 2, cdat_pc.shape[1])
    # range_axis = range_axis[127075:129075]
    # plt.figure()
    # plt.pcolormesh(range_axis, velocity_axis, np.abs(range_doppler ** 2), shading="auto", vmax=3e6)
    # plt.colorbar()
    # plt.xlabel("Range (m)")
    # plt.ylabel("Velocity (m/s)")
    # plt.title('Intelsat 33e Debris from observations on 5/2/25 with Lovell')
    # # plt.xlim([range_axis[127075], range_axis[129075]])
    # plt.savefig(f'./plots/range_doppler_{n+1}.png')
    # plt.show()
    # peak_offset = 128075- peak
    # # print(peak_offset)
    # if np.abs(peak_offset)<10:
    #     # print('\n accepted')
    #     accepted +=1
    #     spectrogram[:, n] = np.abs(range_doppler[:, 1000-peak_offset])**2
    # else:
    #     # print('\n rejected')
    #     rejected +=1
    #     spectrogram[:, n] = np.abs(range_doppler[:, 1000])**2

    spectrogram[:, n] = np.abs(range_doppler[:, peak])**2

    
np.save('/share/nas2/pryder/tumbling_git/most_recent_spectrogram.npy', spectrogram)

# print('Number of accepted = ', accepted)
# print('Number of rejected = ', rejected)
# print("acceptance rate = ", accepted/number_of_strips*100)

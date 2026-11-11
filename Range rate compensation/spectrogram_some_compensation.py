import numpy as np
from pc_utils import rect, open_vdif, iq_conversion

# input parameters 
cpi = 256    
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

# Open VDIF
pola, header = open_vdif(infilename, channel)
print('VDIF opened.')

# Convert to IQ
iq_samples = iq_conversion(pola)
print('Converted into IQ samples.')

overlap = points
time_of_one_cpi = pri*cpi #in seconds
number_of_strips = 10 # = 118.2/6 (20) seconds
print(number_of_strips)
# spectrogram = np.zeros((height*2, number_of_strips))

# what's being recorded
peak_index = []
peak_power = []
average_noise = []

slope = 0.06113453070838821
intercept = 192079.8402591338
shifts = (slope * np.arange(number_of_strips)) + intercept
spectrogram = np.zeros((height*2, number_of_strips))



for n in range(number_of_strips):
    start_idx = (n*overlap)
    end_idx = start_idx + (height * points)
    cpi_data = iq_samples[start_idx:end_idx].reshape((height, points))
    num_pulses, num_bins = cpi_data.shape
    freqs = np.fft.fftfreq(num_bins)
    tau_dot = 0.0

    # Create template: baseband * chirp * envelope
    t_pulse = np.linspace(0, pri, points, endpoint=False)
    template = np.exp(1j * np.pi * alpha * t_pulse**2 * (1 - tau_dot)**2) * rect(t_pulse / (Tp * (1 + tau_dot)))
    template_fft = np.fft.fft(template)

    cdat_pc = np.zeros((height, points), dtype=complex)
    
    

    # Pulse Compression
    for i in range(height):
        pulse_fft = np.fft.fft(cpi_data[i])
        
        # # Pulse Compression and RCM shift
        # phase_ramp = np.exp(-2j*np.pi*shift*freqs)
        shifted_pulse = pulse_fft #* phase_ramp
        compressed_pulse = np.fft.ifft(shifted_pulse * np.conj(template_fft))
        cdat_pc[i] = compressed_pulse

    
    
    
    peak_idx = int(shifts[n])
    peak_index.append(peak_idx)
   

    range_cut = cdat_pc[:, peak_idx] 
    zero_pad = height
    range_cut_padded = np.pad(range_cut, (zero_pad//2, zero_pad//2), mode='constant')

    window_size = range_cut_padded.shape[0]
    window = np.hanning(window_size)
    doppler_spectrum = np.fft.fftshift(np.fft.fft(range_cut_padded * window))

    spectrogram[:, n] = np.abs(doppler_spectrum)**2

np.save('./points_overlap_spectrogram_picking_range_bin_256cpi_short.npy', spectrogram)
    
    
np.save('/share/nas2/pryder/tumbling_git/Range rate compensation/peak_index_over_time_picking_range_bin_256cpi_short.npy', peak_index)
# np.save('/share/nas2/pryder/tumbling_git/Range rate compensation/peak_power_over_time_fractional_compensation_256cpi_short.npy', peak_power)
# np.save('/share/nas2/pryder/tumbling_git/Range rate compensation/average_noise_over_time_fractional_compensation_256cpi.npy', average_noise)


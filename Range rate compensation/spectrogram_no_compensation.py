import numpy as np
from pc_utils import rect, open_vdif, iq_conversion

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

# Open VDIF
pola, header = open_vdif(infilename, channel)
print('VDIF opened.')

# Convert to IQ
iq_samples = iq_conversion(pola)
print('Converted into IQ samples.')

overlap = points
time_of_one_cpi = pri*cpi #in seconds
number_of_strips = 6000 # = 118.2 seconds
print(number_of_strips)
spectrogram = np.zeros((height*2, number_of_strips))

# what's being recorded
peak_index = []
peak_power = []
average_noise = []


for n in range(number_of_strips):
    start_idx = (n*overlap)
    end_idx = start_idx + (height * points)
    cpi_data = iq_samples[start_idx:end_idx].reshape((height, points))
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
        compressed_pulse = np.fft.ifft(pulse_fft * np.conj(template_fft))
        cdat_pc[i] = compressed_pulse
    
    power = (np.abs(cdat_pc))**2
    incoho_sum = np.sum(power, axis=0)
    peak_idx = np.argmax(incoho_sum)
    peak_index.append(peak_idx)
    peak_power.append(incoho_sum[peak_idx])
    average_noise.append(np.median(incoho_sum))
    
    
np.save('/share/nas2/pryder/tumbling_git/Range rate compensation/peak_index_over_time_no_compensation.npy', peak_index)
np.save('/share/nas2/pryder/tumbling_git/Range rate compensation/peak_power_over_time_no_compensation.npy', peak_power)
np.save('/share/nas2/pryder/tumbling_git/Range rate compensation/average_noise_over_time_no_compensation.npy', average_noise)


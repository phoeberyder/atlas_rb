import numpy as np
import matplotlib.pyplot as plt
from pc_utils import rect, open_vdif, iq_conversion

# input parameters 
# cpi = 128      
bw = 8e6        
Tp = 800e-6    
pri = 19.7e-3
freq = 1295e6
c = 299792458
samp_rate = 16e6
alpha = bw/Tp   
# height =cpi # CPI Size
points = int(samp_rate * pri)
infilename = '/share/nas2/pryder/SET_Observations_Test_1/Wednesday/vdifs/TSSat_20250205_lo1_1295MHz_intelsat33e.vdifa' #second vdif
channel = 0
max_velocity = c / (4 * 19.7e-3* (freq))

# Open VDIF
pola, header = open_vdif(infilename, channel)
print('VDIF opened.')

# Convert to IQ
iq_samples = iq_conversion(pola)
print('Converted into IQ samples.')

cpi_list = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
cpi_peak = []
for cpi in cpi_list:
    print(cpi)
    height = cpi
    start_idx = points*2791
    end_idx = start_idx + (height * points)
    cpi_data = iq_samples[start_idx:end_idx].reshape((height, points))
    tau_dot = 0.0
    # Create template: baseband * chirp * envelope
    t_pulse = np.linspace(0, pri, points, endpoint=False)
    template = np.exp(1j * np.pi * alpha * t_pulse**2 * (1 - tau_dot)**2) * rect(t_pulse / (Tp * (1 + tau_dot)))
    template_fft = np.fft.fft(template)

    cdat_pc = np.zeros((height, points), dtype=complex)

    for i in range(height):
        # print(i)
        start_sec = i * pri
        pulse_fft = np.fft.fft(cpi_data[i])
        compressed_pulse = np.fft.ifft(pulse_fft * np.conj(template_fft))
        cdat_pc[i] = compressed_pulse
    power = (np.abs(cdat_pc))**2
    incoho_sum = np.sum(power, axis=0)
    mean_noise = np.mean(incoho_sum[0:100])
    cpi_peak.append(incoho_sum[192083]-mean_noise)

np.save('/share/nas2/pryder/tumbling_git/Range rate compensation/cpi_snr.npy', cpi_peak)



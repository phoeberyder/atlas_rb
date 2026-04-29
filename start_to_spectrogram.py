from unpack_vdif import unpacksamps, sortframes, readframes, readheader
import numpy as np
import matplotlib.pyplot as plt
from schedule_utils import range_finder_general
from skyfield.api import load, EarthSatellite
from pc_utils import rect

# getting original signal and spectrum

#inputting tle
tle_line_1 = '1 40731U 15033B   26046.96407114  .00000009  00000-0  00000+0 0  9995'
tle_line_2 = '2 40731  54.6696 288.5525 0237286   9.2718 351.2385  1.90866878 73834'
ts = load.timescale()
atlas_tle = EarthSatellite(tle_line_1, tle_line_2, 'atlas', ts)

#input paramters
f1 = int(0)        
bw = int(8e6)        
Tp = 800e-6    
pri = 20.506e-3 
freq = 1295e6
c= 299792458
samp_rate = 16e6
alpha = bw/Tp   #chirp rate
N = 100
infilename = '/share/nas2/pryder/realtime_test_1/vdifs/SD20003_20260218_mk2_1295MHz_atlasrb.vdif'

#opens vdif
with open(infilename) as infile:
    header = readheader(infile)
    framedata, seconds, framenums, threads = readframes(infile, header)
threaddata = sortframes(framedata, seconds, framenums, threads)
pola = unpacksamps(threaddata[0,:], header['nbits'], header['dtype'])
print('VDIF opened.')

# Convert to IQ. Add a phase that rotates every 4 samples
phase_4_samples = np.array([ 1+0j, 0-1j, -1+0j, 0+1j ])
phase_all_samples = np.tile(phase_4_samples, pola.size//4+1)  # +1 in case not divisible by 4
iq_samples_not_down = pola*phase_all_samples[:pola.size]
print('Converted into IQ samples.')

# reshaping into pulse repetition intervals
points = int(samp_rate*pri) #number of samples per pulse
height = 128 #coherent processing interval
reduced_length = height*points #length for cpi that includes an integer number of pulses
startoffset = int(samp_rate*100)

cpi_jump_samples = int(reduced_length/4)

# if doing successive cpis, this is the maximum number of strips you can get in your spectrogram
# now edited so it will be universal max (i think) - but for small increments this will be very large - don't run 10 million!
n_max_successive = int(np.floor((len(pola) - startoffset - 1- reduced_length)/reduced_length))
n_max_window = int(np.floor((len(pola)-startoffset-reduced_length)/cpi_jump_samples))

# if doing one sample difference, then n_max is very large 

number_of_strips_in_spectrogram = n_max_window

spectrogram = np.zeros((height, number_of_strips_in_spectrogram))
rcm = np.zeros((320, number_of_strips_in_spectrogram))

#getting range rate for template pulse
ts = load.timescale()
t_tle = ts.utc(2026, 2, 18, 14, 50)
range_rate_from_tle = range_finder_general(atlas_tle, t_tle, 'mark')[1]
tau_dot = range_rate_from_tle/c #normalised range rate
k = np.arange(points)

#making template pulse
t1=np.linspace(0,pri,points,endpoint=False)
baseband_offset_term = np.exp(-2j*np.pi*f1*t1*tau_dot)
chirp_term= np.exp(1j*np.pi*alpha*t1**2*(1-tau_dot)**2) 
envelope_term = rect((t1)/(Tp*(1+tau_dot)))
template = baseband_offset_term*chirp_term*envelope_term
print('Template pulse made.')

peak_range_history = []

for n in range(number_of_strips_in_spectrogram):
    print("strip ", n, " of ", number_of_strips_in_spectrogram)
    cdat_pc = np.zeros((height, points), dtype=complex)
    start_index = cpi_jump_samples*n + startoffset
    end_index = cpi_jump_samples*(n+1)+ reduced_length + startoffset - cpi_jump_samples
    print('which covers samples ', start_index, " to ", end_index)
    pola_reduced = iq_samples_not_down[start_index:end_index]
    dat = pola_reduced.reshape((height, points))

    #doing pulse compression
    for i in range(height):
        start_sec=i*pri
        end_sec=(i+1)*pri
        t=np.linspace(start_sec,end_sec,points,endpoint=False)
        received_pulse=dat[i]
        phase_ramp=np.exp(2j*np.pi*freq*(t+tau_dot+(Tp/2))) #doppler phase correction
        signal1_fft = np.fft.fft(received_pulse)
        signal2_fft = np.fft.fft(template)
        correlation2 = np.fft.ifft(signal1_fft * signal2_fft.conjugate()*phase_ramp)
        cdat_pc[i,:]=correlation2
    print('Pulse compression complete')

    power = (np.abs(cdat_pc))**2
    spectrum = np.sum(power, axis=0)

    freqs = np.fft.fftshift(np.fft.fftfreq(points, (1/samp_rate)))

    lower = 142615
    upper = 142775

    power_detection = power[:, lower:upper].T

    power_detection_padded = np.zeros((power_detection.shape[0]+160, power_detection.shape[1]))
    for i in range(power_detection.shape[1]):
        power_detection_padded[:, i] = np.pad(power_detection[:, i], (160//2, 160//2), mode='constant')

    window = np.hanning(power_detection_padded.shape[0])
    power_detection_p_and_w = np.zeros_like(power_detection_padded)
    for i in range(power_detection_padded.shape[1]):
        power_detection_p_and_w[:, i] = power_detection_padded[:, i] * window

    unshifted_range_doppler = np.fft.fft(power_detection_p_and_w.T, axis=0)
    range_doppler_map = np.fft.fftshift(unshifted_range_doppler, axes=0)

    peaks = np.sum(range_doppler_map, axis = 0)
    peak = np.argmax(peaks)
    strip = range_doppler_map[:, peak:peak+1]
    print('Peak index: ', peak)
    peak_range_history.append(peak)
    # print(spectrogram[:, n:n+1])
    spectrogram[:, n:n+1] = np.abs(strip)**2
    peaks_r = np.sum(range_doppler_map, axis = 1)
    peak_r = np.argmax(peaks_r)
    strip_r = range_doppler_map[peak_r:peak_r+1, :]
    rcm[:, n:n+1] = np.abs(strip_r.T)**2

np.save('./spectrogram.npy', spectrogram)
np.save('./rcm.npy', rcm)

plt.imshow(spectrogram[:, :], origin='lower')
plt.xlabel('Time samples')
plt.ylabel ('range rate samples')
plt.savefig('./spectrogram.png')

plt.imshow(rcm[:, :], origin='lower')
plt.xlabel('Time samples')
plt.ylabel ('range samples')
plt.savefig('./rcm.png')
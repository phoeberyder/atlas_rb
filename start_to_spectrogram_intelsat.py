import numpy as np
import matplotlib.pyplot as plt
from schedule_utils import range_finder_general
from skyfield.api import load, EarthSatellite
from pc_utils import rect, open_vdif, iq_conversion, pc_and_spectrogram, plotter

def start_to_spectrogram_intelsat(infilename, cpi, overlap_factor, telescope, channel, window_function):
    # inputting TLE data for intelsat 33e
    tle_line_1 = '1 61995U 16053M   25035.62796428  .00000095  00000-0  00000-0 0  9998'
    tle_line_2 = '2 61995   0.2642  88.1708 0053675 268.4745   3.2920  1.01033442  1106'
    target_name = 'intelsat33e'
    ts = load.timescale()
    intelsat_tle = EarthSatellite(tle_line_1, tle_line_2, target_name, ts)

    # input parameters       
    bw = 8e6        
    Tp = 800e-6    
    pri = 19.7e-3
    freq = 1295e6
    c = 299792458
    samp_rate = 16e6
    alpha = bw/Tp   
    height =cpi # CPI Size
    points = int(samp_rate * pri)

    # Open VDIF
    pola, header = open_vdif(infilename, channel)
    print('VDIF opened.')

    # Convert to IQ
    iq_samples = iq_conversion(pola)
    print('Converted into IQ samples.')

    rcm_map, spectrogram, peak_history, number_of_strips = pc_and_spectrogram(height, points, overlap_factor, iq_samples, samp_rate, pri, alpha, intelsat_tle, telescope, Tp, freq, window_function)
    print('\nProcessing Complete.')

    # --- Plotting ---
    # Crop the RCM map so Matplotlib doesn't alias the track to oblivion
    min_peak = max(0, np.min(peak_history) - 200)
    max_peak = min(points, np.max(peak_history) + 200)
    cropped_rcm = rcm_map[min_peak:max_peak, :]

    plotter(cropped_rcm, "Range-Time Migration", "CPI Index", "Range Bin", [0, number_of_strips, min_peak, max_peak], target_name, height, overlap_factor, channel, window_function)

    max_velocity = c / (4 * pri * (freq))

    plotter(spectrogram, "Micro-Doppler Spectrogram", "CPI Index", "Doppler Velocity (m/s)", [0, spectrogram.shape[1], -max_velocity, max_velocity], target_name, height, overlap_factor, channel, window_function)

    np.save('./rcm_' + target_name + '_' + str(height) + '_'+str(height//overlap_factor)+'.npy', cropped_rcm)
    np.save('./spectrogram_' + target_name + '_' + str(height) + '_'+str(height//overlap_factor)+'.npy', spectrogram)

    return

def start_to_spectrogram_atlas(infilename, cpi, overlap_factor, telescope, channel, window_function):
    # inputting TLE data for atlas
    tle_line_1 = '1 40731U 15033B   26046.96407114  .00000009  00000-0  00000+0 0  9995'
    tle_line_2 = '2 40731  54.6696 288.5525 0237286   9.2718 351.2385  1.90866878 73834'
    target_name = 'atlas'
    ts = load.timescale()
    atlas_tle = EarthSatellite(tle_line_1, tle_line_2, target_name, ts)

    # input parameters
    f1 = 0        
    bw = 8e6        
    Tp = 800e-6    
    pri = 20.506e-3 
    freq = 1295e6
    c = 299792458
    samp_rate = 16e6
    alpha = bw/Tp   
    height =cpi # CPI Size
    startoffset = int(samp_rate * 100)
    points = int(samp_rate * pri)

    # Open VDIF
    pola, header = open_vdif(infilename, channel)
    print('VDIF opened.')

    # Convert to IQ
    iq_samples = iq_conversion(pola)
    print('Converted into IQ samples.')

    rcm_map, spectrogram, peak_history, number_of_strips = pc_and_spectrogram(height, points, overlap_factor, iq_samples, samp_rate, pri, alpha, atlas_tle, telescope, Tp, freq, window_function)
    print('\nProcessing Complete.')

    # --- Plotting ---
    # Crop the RCM map so Matplotlib doesn't alias the track to oblivion
    min_peak = max(0, np.min(peak_history) - 200)
    max_peak = min(points, np.max(peak_history) + 200)
    cropped_rcm = rcm_map[min_peak:max_peak, :]

    # plotter(cropped_rcm, "Range-Time Migration", "CPI Index", "Range Bin", [0, number_of_strips, min_peak, max_peak], target_name, height, overlap_factor, channel, window_function)

    max_velocity = c / (4 * pri * (freq))

    plotter(spectrogram, "Spectrogram", "CPI Index", "Doppler Velocity (m/s)", [0, spectrogram.shape[1], -max_velocity, max_velocity], target_name, height, overlap_factor, channel, window_function)

    np.save('./rcm_' + target_name + '_' + str(height) + '_'+str(height//overlap_factor)+'channel_'+str(channel)+str(window_function)+'.npy', cropped_rcm)
    np.save('./spectrogram_' + target_name + '_' + str(height) + '_'+str(height//overlap_factor)+'channel_'+str(channel)+str(window_function)+'.npy', spectrogram)
    return
  



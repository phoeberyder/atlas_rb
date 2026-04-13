import numpy as np
from scipy import constants as c
from skyfield.api import EarthSatellite, wgs84, load
from datetime import datetime
import csv
import matplotlib.pyplot as plt
import math
import pytelpoint.transform as pt
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
import astropy.units as u
from astropy.time import Time
import pandas as pd

def tle_collator(filename):
    '''
    Takes csv file of tles and makes them into list of tles
    
    Input:
        filename: path to csv file
    
    Output:
        list of tles
    '''

    with load.open(filename, mode='r') as f:
        data = list(csv.DictReader(f))
    ts = load.timescale()
    sat_list = [EarthSatellite.from_omm(ts, fields) for fields in data]
    return sat_list

def beam_plotter(centre_el, centre_az, bm, wiggle, az_list, el_list, name_list):
    '''
    
    Inputs:
        centre_el: elevation of beam centre in degrees
        centre_az: azimuth of beam centre in degrees
        bm: beamwidth in degrees
        wiggle: area around beam you also want to check in degrees
        az_list: list of azimuths
        el_list: list of elevations
        name_list: list of sat names
    
    Output:
       Plot of beam and surrounding area '''
    min_el = centre_el - bm
    max_el = centre_el + bm
    min_az = centre_az - bm
    max_az = centre_az + bm
    el_in_beam = []
    az_in_beam = []
    name_in_beam = []

    for i in range(len(az_list)):
        # print(az_list[i])
        if (min_az-wiggle <= az_list[i] <= max_az+wiggle) and (min_el-wiggle <= el_list[i] <= max_el+wiggle):
            el_in_beam.append(el_list[i])
            az_in_beam.append(az_list[i])
            name_in_beam.append(name_list[i])

    fig, ax = plt.subplots()
    ax.scatter(az_in_beam, el_in_beam)
    for i, label in enumerate(name_in_beam):
        ax.text(az_in_beam[i], el_in_beam[i], label, fontsize=9, ha='right', va='bottom')
    cir = plt.Circle((centre_az, centre_el), bm, color='r',fill=False)
    ax.set_aspect('equal', adjustable='datalim')
    ax.add_patch(cir)
    ax.set_xlabel('Azimuth (deg)')
    ax.set_ylabel('Elevation (deg)')
    plt.show()
    return

def manual_tle_input(tle_line_1, tle_line_2, name, ts):
    '''Asks for line 1, then line 2 and then a name (line 3) and outputs object as EarthSatellite'''
    tle_line_1 = input('Line 1')
    tle_line_2 = input('Line 2')
    name = input('Name')
    satellite = EarthSatellite(tle_line_1, tle_line_2, str(name), ts)
    return satellite

def apply_pointing_corrections(azimuth, elevation, time):
     #Lovell pointing corrections
    IA = -650.0          #Azimuth index value (i.e. zeropoint)
    IE = -4474.0         #Elevation index value
    HASA2 = -47.0        
    HACA2 = 17.0
    HESE = 170.0
    HESA = 20.0
    HECA = 56.0
    NPAE = 38.0          #Az/El non-perpendicularity. In an alt-az mount, if the azimuth and elevation axes are not exactly at right angles, horizontal shifts occur that are proportional to sin(el).
    CA = -56.0           #Left-Right collimation error. In an alt-az mount, this collimation error is the non-perpendicularity between the nominated pointing direction and the elevation axis. It produces a left-right shift on the sky that is constant for all elevations.
    AW = 34.0            #East-West misalignment of azimuth angle
    TF = -1061.0         #Tube flexure term proportional to cos(el)
    TX = -15.0           #Tube flexure term proportional to cot(el)

    lovell_dish = EarthLocation(lat= 53.2366112016117 * u.deg, lon=-2.3084296589314497 * u.deg)
    aa = AltAz(location=lovell_dish, obstime=time)
    input_coord = SkyCoord(azimuth*u.deg, elevation*u.deg, frame=aa)

    new_coords = pt.azel_model(input_coord, IA, IE, 0, AW, CA, NPAE, TF, TX)

    temp_el = new_coords.alt
    temp_az = new_coords.az

    pytel_corr_el = temp_el.to_value()
    pytel_corr_az = temp_az.to_value()

    correction_el = elevation - pytel_corr_el
    correction_az = azimuth - pytel_corr_az

    rev_pytel_corr_el = elevation + correction_el
    rev_pytel_corr_az = azimuth + correction_az

    del_az_1 = HASA2*math.sin(2*-rev_pytel_corr_az*((2*math.pi)/360))
    del_az_2 = HACA2*math.cos(2*-rev_pytel_corr_az*((2*math.pi)/360))

    del_el_1 = HESE*math.sin(-rev_pytel_corr_el*((2*math.pi)/360))
    del_el_2 = HECA*math.cos(-rev_pytel_corr_az*((2*math.pi)/360))

    del_az = (del_az_1 + del_az_2)/3600
    del_el = (del_el_1 + del_el_2)/3600

    harm_el = rev_pytel_corr_el + del_el
    harm_az = rev_pytel_corr_az + del_az

    return harm_az, harm_el


def range_finder_general(target, t, rx):
    sat_pos = target.at(t).position.km

    # Transmitter and receiver locations
    mhr = wgs84.latlon(42.6, 288.5)
    if rx == 'lovell':
        rx = wgs84.latlon(53.2365, -2.3087)

    if rx == 'cm':
        rx = wgs84.latlon(52.1669, 0.0372)
    
    if rx=='mark':
        rx=wgs84.latlon(53.2339, -2.3038)


    tx_pos = mhr.at(t).position.km
    rx_pos = rx.at(t).position.km

    # Distances
    d_tx_sat = np.linalg.norm(sat_pos - tx_pos)
    d_sat_rx = np.linalg.norm(sat_pos - rx_pos)
    d_tx_rx = np.linalg.norm(tx_pos - rx_pos)

    difference_rx = (target - rx)
    difference_tx = (target - mhr)

    topocentric_rx = (difference_rx.at(t))
    topocentric_tx = (difference_tx.at(t))

    _, _, the_range, _, _, range_rate = topocentric_rx.frame_latlon_and_rates(rx)
    _, _, the_range_tx, _, _, range_rate_tx = topocentric_tx.frame_latlon_and_rates(mhr)
    rr = range_rate_tx.m_per_s/2 + range_rate.m_per_s/2


    # Bistatic range
    bistatic_range = d_tx_sat + d_sat_rx - d_tx_rx
    return bistatic_range, rr
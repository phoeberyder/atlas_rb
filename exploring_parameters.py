from start_to_spectrogram_intelsat import start_to_spectrogram_atlas_function, start_to_spectrogram_intelsat

window_functions_list = ['hamming', 'blackman', 'boxcar', 'kaiser', 'blackmanharris']

for window_function in window_functions_list:
    print(f"Processing Intelsat with {window_function} window...")
    start_to_spectrogram_intelsat('/share/nas2/pryder/SET_Observations_Test_1/Wednesday/vdifs/TSSat_20250205_lo1_1295MHz_intelsat33e.vdif', cpi=128, overlap_factor=2, telescope='lovell', channel=0, window_function=window_function)
    start_to_spectrogram_intelsat('/share/nas2/pryder/SET_Observations_Test_1/Wednesday/vdifs/TSSat_20250205_lo1_1295MHz_intelsat33e.vdif', cpi=128, overlap_factor=2, telescope='lovell', channel=1, window_function=window_function)

    print(f"Processing Atlas with {window_function} window...")
    start_to_spectrogram_atlas_function('/share/nas2/pryder/realtime_test_1/vdifs/SD20003_20260218_mk2_1295MHz_atlasrb.vdif', cpi=128, overlap_factor=2, telescope='mark', channel=0, window_function=window_function)
    start_to_spectrogram_atlas_function('/share/nas2/pryder/realtime_test_1/vdifs/SD20003_20260218_mk2_1295MHz_atlasrb.vdif', cpi=128, overlap_factor=2, telescope='mark', channel=1, window_function=window_function)


# cpi_list = [32, 64, 128, 256, 512]

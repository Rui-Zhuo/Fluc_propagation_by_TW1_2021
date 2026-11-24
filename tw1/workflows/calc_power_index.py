"""calc power index. Use main() or python run_analysis.py."""

def main(case=None, start=None, end=None, save=None, plot=None, file1=None, file2=None, show=True):
    """Run locally; times are UTC HH:MM[:SS] or YYYYDDDHHMMSS."""
    from project_paths import CODE_DIR, EXTERNAL_UTILS_DIR
    import matplotlib.pyplot as plt
    import pyleoclim as pyleo
    import numpy as np
    from matplotlib.colors import LogNorm
    import sys

    sys.path.append(str(EXTERNAL_UTILS_DIR))
    from python_utils_JSHEPT import get_plot_WaveletAnalysis_of_var_vect

    sys.path.append(str(CODE_DIR))
    from utils_signal_analyse import convert_to_second_of_day, convert_to_HHMM, \
        eliminate_outliers, interpolate, detrend, log_linear_fit

    ## import data
    i_case = 3 if case is None else int(case)
    save_or_not = 0 if save is None else int(save)
    from tw1.cases import get_case
    config = get_case('calc_power_index', i_case, start=start, end=end, file1=file1, file2=file2)
    file_dir = config.input_directory
    save_dir = config.output_directory(create=bool(save_or_not))
    file1_name = config.parameters['file1_name']
    file2_name = config.parameters['file2_name']
    fs = config.parameters['fs']
    time_beg = config.parameters['time_beg']
    time_end = config.parameters['time_end']
    time_lb = config.parameters['time_lb']
    time_ub = config.parameters['time_ub']
    # convert subplot time interval
    str_beg = str(time_lb)[-6:-2]
    str_end = str(time_ub)[-6:-2]
    sod_beg = convert_to_second_of_day(time_beg)[0]
    sod_end = convert_to_second_of_day(time_end)[0]
    sod_lb = convert_to_second_of_day(time_lb)[0]
    sod_ub = convert_to_second_of_day(time_ub)[0]
    
    # Time formats and station polarity are declared per filename in cases.json.
    from tw1.io import read_series
    file1_path = file_dir + file1_name
    time1, sod1, freq1 = read_series(file1_path,
        time_format=config.parameters['time_formats'][file1_name],
        sign=config.parameters['frequency_signs'].get(file1_name, 1))
    file2_path = file_dir + file2_name
    time2, sod2, freq2 = read_series(file2_path,
        time_format=config.parameters['time_formats'][file2_name],
        sign=config.parameters['frequency_signs'].get(file2_name, 1))

    from tw1.time import TimeWindow
    window = TimeWindow(time_beg, time_end)

    ## select time interval
    ind1_sub = np.where(window.mask(sod1))
    ind2_sub = np.where(window.mask(sod2))
    # extract corresponding segment
    sod1_sub = sod1[ind1_sub]
    sod2_sub = sod2[ind2_sub]
    freq1_sub = freq1[ind1_sub]
    freq2_sub = freq2[ind2_sub]

    ## frequency series preprocess
    # step 1: eliminate outliers
    freq1_out, sod1_out = eliminate_outliers(freq1_sub, sod1_sub, 10)
    freq2_out, sod2_out = eliminate_outliers(freq2_sub, sod2_sub, 10)
    # step 2: detrend for frequency sequence
    freq1_fit, freq1_detrend = detrend(freq1_out, sod1_out, 3)
    freq1_fit, freq2_detrend = detrend(freq2_out, sod2_out, 3)
    # step 3: interpolation for frequency sequence
    freq1_interp = interpolate(freq1_detrend, sod1_out, sod1_sub)
    freq2_interp = interpolate(freq2_detrend, sod2_out, sod2_sub)

    ## construct pyleo.series         
    series1 = pyleo.Series(time=sod1_sub, value=freq1_interp, \
        time_name='Time', time_unit='s', value_name = 'Freq', value_unit='Hz', label=file1_name[0:2])
    series2 = pyleo.Series(time=sod2_sub, value=freq2_interp, \
        time_name='Time', time_unit='s', value_name = 'Freq', value_unit='Hz', label=file2_name[0:2])

    # wavelet transform
    time1_vect, period1_vect, WaveletObj_arr1, WaveletCoeff_arr1, sub_wave_arr1 = \
        get_plot_WaveletAnalysis_of_var_vect(sod1_sub, freq1_interp, period_range=np.array([2,500]), num_periods=100)
    WaveletCoeff_arr1 = np.transpose(WaveletCoeff_arr1)
    time2_vect, period2_vect, WaveletObj_arr2, WaveletCoeff_arr2, sub_wave_arr2 = \
        get_plot_WaveletAnalysis_of_var_vect(sod2_sub, freq2_interp, period_range=np.array([2,500]), num_periods=100)
    WaveletCoeff_arr2 = np.transpose(WaveletCoeff_arr2)

    timee1, periodd1 = np.meshgrid(time1_vect, period1_vect)
    timee2, periodd2 = np.meshgrid(time2_vect, period2_vect)

    # set xlabels as HHMM
    xposs = [0, 1/6, 2/6, 3/6, 4/6, 5/6, 1]
    xticks = [int(xpos*(sod_end-sod_beg)+sod_beg) for xpos in xposs]
    xlabels = [convert_to_HHMM(xtick) for xtick in xticks]

    # ## plot wavelet coefficient spectrum
    # plt.figure()

    # plt.subplot(2,1,1)
    # plt.pcolormesh(timee1, periodd1, WaveletCoeff_arr1.real)
    # plt.colorbar()
    # plt.xlim([sod_beg, sod_end])
    # plt.xticks(xticks, xlabels)
    # plt.yscale('log')
    # plt.ylabel('Scale[s]')

    # plt.subplot(2,1,2)
    # plt.pcolormesh(timee2, periodd2, WaveletCoeff_arr2.real)
    # plt.colorbar()
    # plt.xlim([sod_beg, sod_end])
    # plt.xticks(xticks, xlabels)
    # plt.yscale('log')
    # plt.ylabel('Scale[s]')
    # plt.close()

    ## calculate wavelet power spectrum
    psd1 = 2 * np.abs(WaveletCoeff_arr1)**2 / fs
    psd2 = 2 * np.abs(WaveletCoeff_arr2)**2 / fs
    if i_case == 2 and file1_name == 'YgYgchan3_1frephase2s.dat':
        psd1 = 2 * np.abs(WaveletCoeff_arr1)**2 / 0.5
    elif i_case == 2 and file1_name == 'YgYgchan3_1frephase2s.dat':
        psd2 = 2 * np.abs(WaveletCoeff_arr2)**2 / 0.5

    ## select time interval for calculating power index
    sod1_slc_ind = np.where((time1_vect > sod_lb) & (time1_vect < sod_ub))[0]
    sod2_slc_ind = np.where((time2_vect > sod_lb) & (time2_vect < sod_ub))[0]
    sod1_slc = time1_vect[sod1_slc_ind]
    sod2_slc = time2_vect[sod2_slc_ind]
    psd1_slc = psd1[:, sod1_slc_ind]
    psd2_slc = psd2[:, sod2_slc_ind]

    ## calculate power index
    freq1_vect = 1/period1_vect
    freq2_vect = 1/period2_vect
    psd1_mean = np.mean(psd1_slc, 1)
    psd2_mean = np.mean(psd2_slc, 1)

    psd_slope1, freq1_calc, psd1_fit = log_linear_fit(freq1_vect, psd1_mean, 0.01)
    psd_slope2, freq2_calc, psd2_fit = log_linear_fit(freq2_vect, psd2_mean, 0.01)

    ## plot figure
    plt.figure(figsize=(10, 6))
    plt.subplot2grid((2, 3), (0, 0), colspan=2)
    plt.pcolormesh(timee1, periodd1, psd1, norm=LogNorm())
    plt.clim(np.min(psd1)*((np.max(psd1)/np.min(psd1))**(1/4)), np.max(psd1))
    plt.axvline(sod_lb, color='black', linestyle='--')
    plt.axvline(sod_ub, color='black', linestyle='--')
    plt.colorbar()
    plt.xlim([sod_beg, sod_end])
    plt.xticks(xticks, xlabels)
    plt.yscale('log')
    plt.ylabel('Scale[s]')

    plt.subplot2grid((2, 3), (0, 2))
    plt.plot(freq1_vect, psd1_mean)
    plt.plot(freq1_calc, psd1_fit, label='alpha=' + str(-round(psd_slope1,2)))
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.ylabel(file1_name[0:2] + '-PSD[Hz2/Hz]')

    plt.subplot2grid((2, 3), (1, 0), colspan=2)
    plt.pcolormesh(timee2, periodd2, psd2, norm=LogNorm())
    plt.clim(np.min(psd2)*((np.max(psd2)/np.min(psd2))**(1/4)), np.max(psd2))
    plt.axvline(sod_lb, color='black', linestyle='--')
    plt.axvline(sod_ub, color='black', linestyle='--')
    plt.colorbar()
    plt.xlim([sod_beg, sod_end])
    plt.xticks(xticks, xlabels)
    plt.yscale('log')
    plt.xlabel('Time[s]')
    plt.ylabel('Scale[s]')

    plt.subplot2grid((2, 3), (1, 2))
    plt.plot(freq2_vect, psd2_mean)
    plt.plot(freq2_calc, psd2_fit, label='alpha=' + str(-round(psd_slope2,2)))
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.xlabel('Freq[Hz]')
    plt.ylabel(file2_name[0:2] + '-PSD[Hz2/Hz]')

    plt.suptitle('PSD on '+file_dir.rstrip('/').split('/')[-1])

    if save_or_not == 1:
        plt.savefig(save_dir + 'psd-' + str_beg + '-' + str_end + '.png')

    if show:

        plt.show()

    # db

    result = {'psd_slope1': psd_slope1, 'psd_slope2': psd_slope2}
    if not show:
        plt.close("all")
    return result

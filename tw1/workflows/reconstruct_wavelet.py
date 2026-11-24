"""reconstruct wavelet. Use main() or python run_analysis.py."""

def main(case=None, start=None, end=None, save=None, plot=None, file1=None, file2=None, show=True):
    """Run locally; times are UTC HH:MM[:SS] or YYYYDDDHHMMSS."""
    from project_paths import CODE_DIR, EXTERNAL_UTILS_DIR
    import matplotlib.pyplot as plt
    import pyleoclim as pyleo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import sys

    sys.path.append(str(EXTERNAL_UTILS_DIR))
    from python_utils_JSHEPT import get_plot_WaveletAnalysis_of_var_vect

    sys.path.append(str(CODE_DIR))
    from utils_signal_analyse import convert_to_second_of_day, convert_to_HHMM, eliminate_outliers, interpolate, detrend

    ## Selecting station pair and time interval
    i_case = 9 if case is None else int(case)
    period_lb, period_ub = 100, 300
    save_or_not = 1 if save is None else int(save)
    ########## Case 01-19 are 2021 Conjunction ##########
    from tw1.cases import get_case
    config = get_case('reconstruct_wavelet', i_case, start=start, end=end, file1=file1, file2=file2)
    file_dir = config.input_directory
    save_dir = config.output_directory(create=bool(save_or_not))
    file1_name = config.parameters['file1_name']
    file2_name = config.parameters['file2_name']
    time_beg = config.parameters['time_beg']
    time_end = config.parameters['time_end']
    
    # convert subplot time interval
    str_beg = str(time_beg)[-6:-2]
    str_end = str(time_end)[-6:-2]
    sod_beg = convert_to_second_of_day(time_beg)[0]
    sod_end = convert_to_second_of_day(time_end)[0]
    
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
    freq1_out, sod1_out = eliminate_outliers(freq1_sub, sod1_sub, 10) #10
    freq2_out, sod2_out = eliminate_outliers(freq2_sub, sod2_sub, 10) #10
    # step 2: detrend for frequency sequence
    freq1_fit, freq1_detrend = detrend(freq1_out, sod1_out, 3) #3,5,7,9,11
    freq1_fit, freq2_detrend = detrend(freq2_out, sod2_out, 3) #3,5,7,9,11
    # step 3: interpolation for frequency sequence
    freq1_interp = interpolate(freq1_detrend, sod1_out, sod1_sub)
    freq2_interp = interpolate(freq2_detrend, sod2_out, sod2_sub)

    ## construct pyleo.series         
    series1 = pyleo.Series(time=sod1_sub, value=freq1_interp, \
        time_name='Time', time_unit='s', value_name = 'Freq', value_unit='Hz', label=file1_name[0:2])
    series2 = pyleo.Series(time=sod2_sub, value=freq2_interp, \
        time_name='Time', time_unit='s', value_name = 'Freq', value_unit='Hz', label=file2_name[0:2])

    ## lowpass filter cutoff at 0.1 Hz
    if sod1[1] - sod1[0] < 1: # PLL data, already filted
        series1 = series1.interp(step=0.5)
        series2 = series2.interp(step=0.5)
    elif sod1[1] - sod1[0] == 5:
        series1 = series1.interp().filter(cutoff_freq=0.05)
        series2 = series2.interp().filter(cutoff_freq=0.05)
    else:
        series1 = series1.interp().filter(cutoff_freq=0.1)
        series2 = series2.interp().filter(cutoff_freq=0.1)
    sod1_sub = series1.time
    sod2_sub = series2.time
    freq1_interp = series1.value
    freq2_interp = series2.value

    # wavelet transform
    time1_vect, period1_vect, WaveletObj_arr1, WaveletCoeff_arr1, sub_wave_arr1 = \
        get_plot_WaveletAnalysis_of_var_vect(sod1_sub, freq1_interp, period_range=np.array([2,800]), num_periods=100)
    WaveletCoeff_arr1 = np.transpose(WaveletCoeff_arr1)
    time2_vect, period2_vect, WaveletObj_arr2, WaveletCoeff_arr2, sub_wave_arr2 = \
        get_plot_WaveletAnalysis_of_var_vect(sod2_sub, freq2_interp, period_range=np.array([2,800]), num_periods=100)
    WaveletCoeff_arr2 = np.transpose(WaveletCoeff_arr2)

    timee1, periodd1 = np.meshgrid(time1_vect, period1_vect)
    timee2, periodd2 = np.meshgrid(time2_vect, period2_vect)

    ## extract wavelet component
    period1_slc_ind = np.where((period1_vect > period_lb) & (period1_vect < period_ub))[0]
    period2_slc_ind = np.where((period2_vect > period_lb) & (period2_vect < period_ub))[0]
    period1_slc = period1_vect[period1_slc_ind]
    period2_slc = period2_vect[period2_slc_ind]
    sub_wave1_slc = sub_wave_arr1[:,period1_slc_ind]
    sub_wave2_slc = sub_wave_arr2[:,period2_slc_ind]

    ## wavelet coherence analysis
    coh = series2.wavelet_coherence(series1, method='cwt')
    coh.wtc[coh.wtc>1] = np.nan
    scale_range = [np.log10(np.min(coh.scale)), np.log10(np.max(coh.scale))]

    ## plotly figure
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, shared_yaxes=True, vertical_spacing=0.02)

    # panel 1: time series
    fig.add_trace(go.Scatter(x=sod1_sub, y=freq1_interp, mode='lines', marker={'color':'blue'}, name=file1_name[0:2]), \
        row=1, col=1)
    fig.add_trace(go.Scatter(x=sod2_sub, y=freq2_interp, mode='lines', marker={'color':'red'}, name=file2_name[0:2]), \
        row=1, col=1)
    fig.update_yaxes(title='Freq [Hz]', row=1, col=1)

    # panel 2: coherence spectrum
    fig.add_trace(go.Contour(x=coh.time, y=coh.scale, z=np.flipud(np.rot90(coh.wtc)), \
        colorscale='magma', colorbar_title_text='Coherence', colorbar_y=0.5, colorbar_len=0.2), \
            row=2, col=1)
    fig.add_trace(go.Scatter(x=coh.time, y=coh.coi, mode='lines', line=dict(dash='dash'), name='coi'), \
        row=2, col=1)
    fig.add_hline(y=period_lb, line=dict(color="black", dash='dash'), row=2, col=1)
    fig.add_hline(y=period_ub, line=dict(color="black", dash='dash'), row=2, col=1)
    fig.update_yaxes(range=scale_range, type='log', title='Scale [s]', row=2, col=1)

    # panel 3: reconstructed series
    fig.add_trace(go.Scatter(x=sod1_sub, y=np.sum(sub_wave1_slc, axis=1), mode='lines', marker={'color':'blue'}, \
        name=file1_name[0:2]+'-recon'), row=3, col=1)
    fig.add_trace(go.Scatter(x=sod2_sub, y=np.sum(sub_wave2_slc, axis=1), mode='lines', marker={'color':'red'}, \
        name=file2_name[0:2]+'-recon'), row=3, col=1)
    fig.update_yaxes(title='Freq [Hz]', row=3, col=1)

    # figure layout
    xposs = [0, 1/6, 2/6, 3/6, 4/6, 5/6, 1]
    xticks = [int(xpos*(sod_end-sod_beg)+sod_beg) for xpos in xposs]
    xlabels = [convert_to_HHMM(xtick) for xtick in xticks]
    fig.update_xaxes(title_text='Time [HHMM]', tickvals=xticks, ticktext=xlabels, row=3, col=1)
    fig.update_layout(title={'text': 'Wavelet Reconstruction on '+file_dir.rstrip('/').split('/')[-1],'x': 0.5, 'y': 0.95})

    if save_or_not == 1:
        fig.write_html(save_dir + file1_name[0:2] + '-' + file2_name[0:2] + '-' + str_beg + '-' + str_end + '-Reconstruct.html')
    else:
        if show:
            fig.show()

    # db

    result = {'series1': series1, 'series2': series2, 'sub_wave1_slc': sub_wave1_slc, 'sub_wave2_slc': sub_wave2_slc}
    if not show:
        plt.close("all")
    return result

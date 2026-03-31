import matplotlib.pyplot as plt
import pandas as pd
import pyleoclim as pyleo
import numpy as np
from matplotlib.colors import SymLogNorm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import statistics
import matplotlib as mpl

import utils_signal_analyse
from utils_signal_analyse import convert_to_second_of_day, convert_to_HHMM, \
    eliminate_outliers, interpolate, detrend

## Selecting station pair and time interval
i_date = 1
save_or_not = 1
plot_option = 1 # 0 for plotly, 1 for plt
freq_or_phase = 0 # 0 for frequency, 1 for phase
########## 2021 Conjunction ##########
if i_date == 1: # 2021/10/23(296), 04:23-06:30, sh-km, sh-ur, km-ur
    file_dir = 'E:/Research/Data/Tianwen-1/m1a23x/high_resol/'
    save_dir = 'E:/Research/Work/tianwen_IPS/m1a23x/high_resol/'
    file_sh = 'TWshfreq_1s_t.dat' # 04:23-06:30, 1s, time has been formatted as 'sod'
    file_km = 'TWkmfreq_1s_t.dat' # 04:24-06:30, 1s, time has been formatted as 'sod'
    file_ur = 'TWurfreq_1s_t.dat' # 04:28-06:30, 1s, time has been formatted as 'sod'
    file_sh_high_resol = 'TWshfreq_short_t.dat' # 04:23-06:30, 0.025s, time has been formatted as 'sod'
    file_km_high_resol = 'TWkmfreq_short_t.dat' # 04:24-06:30, 0.025s, time has been formatted as 'sod'
    file_ur_high_resol = 'TWurfreq_short_t.dat' # 04:28-06:30, 0.025s, time has been formatted as 'sod'
    file1_name = file_sh_high_resol
    file2_name = file_ur_high_resol
    time_beg = 2021296054000
    time_end = 2021296055000
    
# convert subplot time interval
str_beg = str(time_beg)[-6:-2]
str_end = str(time_end)[-6:-2]
sod_beg = convert_to_second_of_day(time_beg)[0]
sod_end = convert_to_second_of_day(time_end)[0]
    
## read data
file1_path = file_dir + file1_name
file2_path = file_dir + file2_name
data1 = np.loadtxt(file1_path)
data2 = np.loadtxt(file2_path)

## extrcat data
time1 = data1[:, 0]
time2 = data2[:, 0]
sod1 = time1 # time has been formatted as 'sod'
sod2 = time2 # time has been formatted as 'sod'

if file1_name[-8:-6] == '1s':
    freq1 = data1[:,1]
    freq2 = data2[:,1]
elif file1_name[-11:-6] == 'short':
    freq1 = data1[:,2]
    freq2 = data2[:,2]
    phase1 = data1[:,3]
    phase2 = data2[:,3]

if freq_or_phase == 0:
    freq1, freq2 = freq1, freq2
elif freq_or_phase == 1:
    freq1, freq2 = phase1, phase2

## select time interval
ind1_sub = np.where((sod1 > sod_beg) & (sod1 < sod_end))
ind2_sub = np.where((sod2 > sod_beg) & (sod2 < sod_end))
# extract corresponding segment
sod1_sub = sod1[ind1_sub]
sod2_sub = sod2[ind2_sub]
freq1_sub = freq1[ind1_sub]
freq2_sub = freq2[ind2_sub]

## frequency series preprocess
# step 1: eliminate outliers
freq1_out, sod1_out = eliminate_outliers(freq1_sub, sod1_sub, 3) #10
freq2_out, sod2_out = eliminate_outliers(freq2_sub, sod2_sub, 3) #10
# step 2: detrend for frequency sequence
freq1_fit, freq1_detrend = detrend(freq1_out, sod1_out, 9) #3,5,7,9,11
freq2_fit, freq2_detrend = detrend(freq2_out, sod2_out, 9) #3,5,7,9,11
# step 3: interpolation for frequency sequence
freq1_interp = interpolate(freq1_detrend, sod1_out, sod1_sub)
freq2_interp = interpolate(freq2_detrend, sod2_out, sod2_sub)

# ## calculate standard deviation
std_dev1 = statistics.stdev(freq1_interp)
std_dev2 = statistics.stdev(freq2_interp)
print("std_dev_1=", std_dev1)
print("std_dev_2=", std_dev2)

## constuct test series
def test_arr(test_type, length, dt):
    time_arr = np.arange(1, length + 1)
    if test_type == 'random':
        random_seq = np.random.rand(length + dt)
        test_arr1 = random_seq[:length]
        test_arr2 = random_seq[dt:dt+length]
    elif test_type == 'sine':
        sine_time_arr = np.arange(1, length + dt + 1)
        sine_seq = np.sin(2 * np.pi * sine_time_arr / (length / 5))
        test_arr1 = sine_seq[:length]
        test_arr2 = sine_seq[dt:dt+length]
    return time_arr, test_arr1, test_arr2

time_arr, test_arr1, test_arr2 = test_arr('random', 1800, 10)
# sod1_sub, sod2_sub = time_arr, time_arr
# freq1_interp, freq2_interp = test_arr1, test_arr2

## construct pyleo.series
if freq_or_phase == 0:
    series1 = pyleo.Series(time=sod1_sub, value=freq1_interp, \
        time_name='Time', time_unit='s', value_name = 'Freq', value_unit='Hz', label=file1_name[2:4])
    series2 = pyleo.Series(time=sod2_sub, value=freq2_interp, \
        time_name='Time', time_unit='s', value_name = 'Freq', value_unit='Hz', label=file2_name[2:4])
elif freq_or_phase == 1:
    series1 = pyleo.Series(time=sod1_sub, value=freq1_interp, \
        time_name='Time', time_unit='s', value_name = 'Phase', value_unit='rad', label=file1_name[2:4])
    series2 = pyleo.Series(time=sod2_sub, value=freq2_interp, \
        time_name='Time', time_unit='s', value_name = 'Phase', value_unit='rad', label=file2_name[2:4])

## lowpass filter cutoff at 0.1 Hz
if sod1[1] - sod1[0] < 1: # PLL data, already filted
    series1 = series1.interp()
    series2 = series2.interp()
elif sod1[1] - sod1[0] == 5:
    series1 = series1.interp().filter(cutoff_freq=0.05)
    series2 = series2.interp().filter(cutoff_freq=0.05)
else:
    series1 = series1.interp().filter(cutoff_freq=0.1)
    series2 = series2.interp().filter(cutoff_freq=0.1)
freq1_interp = series1.value
freq2_interp = series2.value

# ## transform analysis
# # Fourier transform
# fft1 = series1.spectral(method='cwt')
# fft2 = series2.spectral(method='cwt')
# # wavelet transform
# cwt1 = series1.wavelet(method='cwt')
# cwt2 = series2.wavelet(method='cwt')

## wavelet coherence analysis
coh = series2.wavelet_coherence(series1, method='cwt')
coh.wtc[coh.wtc>1] = np.nan
time_lag = np.flipud(np.rot90(coh.phase/2/np.pi/coh.frequency[np.newaxis,:]))
# if i_date == 9:
#     distance = 1175.8 # Js-Bd
#     vel = distance / time_lag
scale_range = [np.log10(np.min(coh.scale)), np.log10(np.max(coh.scale))]

if plot_option == 0:
    ## plot coherence and time lag spectrum
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, shared_yaxes=True, vertical_spacing=0.02)

    # panel 1: time series
    fig.add_trace(go.Scatter(x=coh.time, y=freq1_interp, mode='lines', name=file1_name[2:4]), row=1, col=1)
    fig.add_trace(go.Scatter(x=coh.time, y=freq2_interp, mode='lines', name=file2_name[2:4]), row=1, col=1)
    if freq_or_phase == 0:
        fig.update_yaxes(title='Freq [Hz]', row=1, col=1)
    elif freq_or_phase == 1:
        fig.update_yaxes(title='Phase [rad]', row=1, col=1)

    # panel 2: complex spectrum , with time lag as data and coherence as background
    fig.add_trace(go.Contour(x=coh.time, y=coh.scale, z=np.flipud(np.rot90(coh.wtc)), \
        colorscale='magma', colorbar_title_text='Coherence', colorbar_y=0.5, colorbar_len=0.2), \
            row=2, col=1)
    fig.add_trace(go.Heatmap(x=coh.time, y=coh.scale, z=time_lag, \
        colorscale='RdBu', zmin=-np.max(np.abs(time_lag)), zmax=np.max(np.abs(time_lag)), \
            colorbar_title_text='Lag [s]', colorbar_y=0.1, colorbar_len=0.3), \
                row=2, col=1)
    # fig.add_trace(go.Heatmap(x=coh.time, y=coh.scale, z=vel, \
    #     colorscale='RdBu', zmin=-np.max(np.abs(vel)), zmax=np.max(np.abs(vel)), \
    #         colorbar_title_text='Velocity [km/s]', colorbar_y=0.1, colorbar_len=0.3), \
    #             row=2, col=1)
    # plot cone of influence
    fig.add_trace(go.Scatter(x=coh.time, y=coh.coi, mode='lines', line=dict(dash='dash'), name='COI'), \
        row=2, col=1)
    fig.update_yaxes(range=scale_range, type='log', title='Scale [s]', row=2, col=1)

    # panel 3: time lag spectrum
    fig.add_trace(go.Contour(x=coh.time, y=coh.scale, z=time_lag, \
        colorscale='RdBu', zmin=-np.max(np.abs(time_lag)), zmax=np.max(np.abs(time_lag)), \
            colorbar_title_text='Lag [s]', colorbar_y=0.1, colorbar_len=0.3), \
                row=3, col=1)
    # plot cone of influence
    fig.add_trace(go.Scatter(x=coh.time, y=coh.coi, mode='lines', line=dict(dash='dash'), name='COI'), \
        row=3, col=1)
    fig.update_yaxes(range=scale_range, type='log', title='Scale [s]', row=3, col=1)

    # # panel 4: velocity spectrum
    # fig.add_trace(go.Contour(x=coh.time, y=coh.scale, z=vel, \
    #     colorscale='RdBu', zmin=-np.max(np.abs(vel)), zmax=np.max(np.abs(vel)), \
    #         colorbar_title_text='Velocity [km/s]', colorbar_y=0.1, colorbar_len=0.3), \
    #             row=3, col=1)
    # # plot cone of influence
    # fig.add_trace(go.Scatter(x=coh.time, y=coh.coi, mode='lines', line=dict(dash='dash'), name='COI'), \
    #     row=3, col=1)
    # fig.update_yaxes(range=scale_range, type='log', title='Scale [s]', row=3, col=1)

    # figure layout
    xposs = [0, 1/6, 2/6, 3/6, 4/6, 5/6, 1]
    xticks = [int(xpos*(sod_end-sod_beg)+sod_beg) for xpos in xposs]
    xlabels = [convert_to_HHMM(xtick) for xtick in xticks]
    fig.update_xaxes(title_text='Time [HHMM]', tickvals=xticks, ticktext=xlabels, row=3, col=1)
    fig.update_layout(title={'text': file2_name[2:4] + ' relative to ' + file1_name[2:4] + ' on ' + file_dir[35:-1], \
        'x': 0.5, 'y': 0.95})

    if save_or_not == 1:
        fig.write_html(save_dir + file1_name[2:4] + '-' + file2_name[2:4] + '-' + str_beg + '-' + str_end + '-Summary.html')
        # fig.write_html(save_dir + file1_name[0:2] + '-' + file2_name[0:2] + '-' + str_beg + '-' + str_end + '-Velocity.html')
        # fig.write_html(save_dir + 'test.html')
    else:
        fig.show()

elif plot_option == 1:
    plt.figure(figsize=(16,8))
    
    # Panel 1: Time series
    plt.subplot(3, 1, 1)
    # Plotting wavelet coherency spectrum
    plt.plot(coh.time, freq1_interp, color='r', label=file1_name[2:4])
    plt.plot(coh.time, freq2_interp, color='b', label=file2_name[2:4])
    # Plotting horizontal reference line
    plt.axhline(y=0, color='k', linestyle=':')
    # Setting figure layout
    plt.ylabel('Freq (Hz)')
    plt.xlim([np.min(coh.time), np.max(coh.time)])
    plt.ylim([-0.4, 0.4])
    plt.xticks([])
    lg = plt.legend(loc='upper right')
    
    # Panel 2: Wavelet Coherency
    plt.subplot(3, 1, 2)
    # Plotting wavelet coherency spectrum
    X, Y = np.meshgrid(coh.time, coh.scale)
    contour_plot = plt.contourf(X, Y, np.flipud(np.rot90(coh.wtc)), cmap='jet', vmin=0, vmax=1)
    cbar1 = plt.colorbar()
    cbar1.set_label('WTC')
    # Plotting Cone of Influence
    plt.plot(coh.time, coh.coi, color='w', linestyle='--', label='COI')
    # Plotting horizontal reference line
    plt.axhline(y=100, color='c', linestyle=':')
    # Setting figure layout
    plt.yscale('log')
    plt.ylabel('Scale (s)')
    plt.ylim([np.min(coh.scale), np.max(coh.scale)])
    plt.xticks([])
    lg = plt.legend(loc='upper right')
    for text in lg.texts:
        text.set_color('w')

    # Panel 3: Time lag
    plt.subplot(3, 1, 3)
    # Plotting time lag spectrum
    X, Y = np.meshgrid(coh.time, coh.scale)
    contour_plot = plt.contourf(X, Y, time_lag, cmap='RdBu', vmin=-np.max(np.abs(time_lag)), vmax=np.max(np.abs(time_lag)))
    contour_line = plt.contour(X, Y, time_lag, colors='k', linewidths=0.3)
    cbar = plt.colorbar(contour_plot)
    cbar.set_label('Lag [s]')
    # Plotting Cone of Influence
    plt.plot(coh.time, coh.coi, color='k', linestyle='--', label='COI')
    # Plotting horizontal reference line
    plt.axhline(y = 100, color='c', linestyle=':')
    # Setting figure layout
    plt.yscale('log')
    plt.xlabel('Time [HH:MM]')
    plt.ylabel('Scale (s)')
    plt.ylim([np.min(coh.scale), np.max(coh.scale)])
    plt.legend(loc='upper right')
    
    # Setting X-labels
    xposs = [0, 1/6, 2/6, 3/6, 4/6, 5/6, 1]
    xticks = [int(xpos*(sod_end-sod_beg)+sod_beg) for xpos in xposs]
    xlabels = [convert_to_HHMM(xtick) for xtick in xticks]
    plt.xticks(xticks, labels=xlabels)

    # Setting subplot vertical gaps
    plt.subplots_adjust(hspace=0.2)

    plt.savefig(save_dir + file1_name[2:4] + '-' + file2_name[2:4] + '-' + str_beg + '-' + str_end + '-Summary.png')
    
    plt.close()
    

# db
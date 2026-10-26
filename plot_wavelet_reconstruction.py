import sys
import numpy as np
import pandas as pd
import pyleoclim as pyleo
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import SymLogNorm, LogNorm

sys.path.append(r'E:/Research/Program/[else]')
import python_utils_JSHEPT
from python_utils_JSHEPT import get_plot_WaveletAnalysis_of_var_vect

from utils_signal_analyse import convert_to_second_of_day, convert_to_HHMM, \
    eliminate_outliers, interpolate, detrend, log_linear_fit

## Selecting station pair and time interval
time_lb, time_ub = 35600, 35850
period_lb, period_ub = 80, 140
save_or_not = 1

# Case at 2021/10/12(285), 09:50-11:00, Ht-Wz, Ht-Ys, Wz-Ys
file_dir = 'E:/Research/Data/Tianwen-1/m1a12x/'
save_dir = 'E:/Research/Work/Fluc_propagation_by_TW1_2021/m1a12x/recon/'
file_Ht = 'HtHtchan3_1frephase1s.dat' # 09:50-11:00, 1s
file_Wz = 'WzWzchan3_1frephase1s.dat' # 09:50-14:00, 1s
file_Ys = 'YsYschan3_1frephase1s.dat' # 09:50-11:00, 1s
file1_name = file_Ht
file2_name = file_Wz
file3_name = file_Ys
time_beg = 2021285095000
time_end = 2021285102000
    
# convert subplot time interval
str_beg = str(time_beg)[-6:-2]
str_end = str(time_end)[-6:-2]
sod_beg = convert_to_second_of_day(time_beg)[0]
sod_end = convert_to_second_of_day(time_end)[0]
    
## read data
file1_path = file_dir + file1_name
file2_path = file_dir + file2_name
file3_path = file_dir + file3_name
data1 = np.loadtxt(file1_path)
data2 = np.loadtxt(file2_path)
data3 = np.loadtxt(file3_path)

## extrcat data
time1 = data1[:, 0]
time2 = data2[:, 0]
time3 = data3[:, 0]
sod1 = convert_to_second_of_day(time1)
sod2 = convert_to_second_of_day(time2)
sod3 = convert_to_second_of_day(time3)
freq1 = data1[:,1]
freq2 = data2[:,1]
freq3 = data3[:,1]

## select time interval
ind1_sub = np.where((sod1 > sod_beg) & (sod1 < sod_end))
ind2_sub = np.where((sod2 > sod_beg) & (sod2 < sod_end))
ind3_sub = np.where((sod3 > sod_beg) & (sod3 < sod_end))
# extract corresponding segment
sod1_sub = sod1[ind1_sub]
sod2_sub = sod2[ind2_sub]
sod3_sub = sod3[ind3_sub]
freq1_sub = freq1[ind1_sub]
freq2_sub = freq2[ind2_sub]
freq3_sub = freq3[ind3_sub]

## frequency series preprocess
# step 1: eliminate outliers
freq1_out, sod1_out = eliminate_outliers(freq1_sub, sod1_sub, 10) #10
freq2_out, sod2_out = eliminate_outliers(freq2_sub, sod2_sub, 10) #10
freq3_out, sod3_out = eliminate_outliers(freq3_sub, sod3_sub, 10) #10
# step 2: detrend for frequency sequence
freq1_fit, freq1_detrend = detrend(freq1_out, sod1_out, 3) #3,5,7,9,11
freq1_fit, freq2_detrend = detrend(freq2_out, sod2_out, 3) #3,5,7,9,11
freq1_fit, freq3_detrend = detrend(freq3_out, sod3_out, 3) #3,5,7,9,11
# step 3: interpolation for frequency sequence
freq1_interp = interpolate(freq1_detrend, sod1_out, sod1_sub)
freq2_interp = interpolate(freq2_detrend, sod2_out, sod2_sub)
freq3_interp = interpolate(freq3_detrend, sod3_out, sod3_sub)

## construct pyleo.series         
series1 = pyleo.Series(time=sod1_sub, value=freq1_interp, \
    time_name='Time', time_unit='s', value_name = 'Freq', value_unit='Hz', label=file1_name[0:2])
series2 = pyleo.Series(time=sod2_sub, value=freq2_interp, \
    time_name='Time', time_unit='s', value_name = 'Freq', value_unit='Hz', label=file2_name[0:2])
series3 = pyleo.Series(time=sod3_sub, value=freq3_interp, \
    time_name='Time', time_unit='s', value_name = 'Freq', value_unit='Hz', label=file3_name[0:2])

## lowpass filter cutoff at 0.1 Hz
series1 = series1.interp().filter(cutoff_freq=0.1)
series2 = series2.interp().filter(cutoff_freq=0.1)
series3 = series3.interp().filter(cutoff_freq=0.1)

sod1_sub = series1.time
sod2_sub = series2.time
sod3_sub = series3.time
freq1_interp = series1.value
freq2_interp = series2.value
freq3_interp = series3.value

# wavelet transform
time1_vect, period1_vect, WaveletObj_arr1, WaveletCoeff_arr1, sub_wave_arr1 = \
    get_plot_WaveletAnalysis_of_var_vect(sod1_sub, freq1_interp, period_range=np.array([2,800]), num_periods=100)
WaveletCoeff_arr1 = np.transpose(WaveletCoeff_arr1)
time2_vect, period2_vect, WaveletObj_arr2, WaveletCoeff_arr2, sub_wave_arr2 = \
    get_plot_WaveletAnalysis_of_var_vect(sod2_sub, freq2_interp, period_range=np.array([2,800]), num_periods=100)
WaveletCoeff_arr2 = np.transpose(WaveletCoeff_arr2)
time3_vect, period3_vect, WaveletObj_arr3, WaveletCoeff_arr3, sub_wave_arr3 = \
    get_plot_WaveletAnalysis_of_var_vect(sod3_sub, freq3_interp, period_range=np.array([2,800]), num_periods=100)
WaveletCoeff_arr3 = np.transpose(WaveletCoeff_arr3)

timee1, periodd1 = np.meshgrid(time1_vect, period1_vect)
timee2, periodd2 = np.meshgrid(time2_vect, period2_vect)
timee3, periodd3 = np.meshgrid(time3_vect, period3_vect)

## extract wavelet component
period1_slc_ind = np.where((period1_vect > period_lb) & (period1_vect < period_ub))[0]
period2_slc_ind = np.where((period2_vect > period_lb) & (period2_vect < period_ub))[0]
period3_slc_ind = np.where((period3_vect > period_lb) & (period3_vect < period_ub))[0]
period1_slc = period1_vect[period1_slc_ind]
period2_slc = period2_vect[period2_slc_ind]
period3_slc = period3_vect[period3_slc_ind]
sub_wave1_slc = sub_wave_arr1[:,period1_slc_ind]
sub_wave2_slc = sub_wave_arr2[:,period2_slc_ind]
sub_wave3_slc = sub_wave_arr3[:,period3_slc_ind]

## wavelet coherence analysis
coh12 = series2.wavelet_coherence(series1, method='cwt')
coh12.wtc[coh12.wtc>1] = np.nan
lag12 = np.flipud(np.rot90(coh12.phase/2/np.pi/coh12.frequency[np.newaxis,:]))

coh13 = series3.wavelet_coherence(series1, method='cwt')
coh13.wtc[coh13.wtc>1] = np.nan
lag13 = np.flipud(np.rot90(coh13.phase/2/np.pi/coh13.frequency[np.newaxis,:]))

coh23 = series3.wavelet_coherence(series2, method='cwt')
coh23.wtc[coh23.wtc>1] = np.nan
lag23 = np.flipud(np.rot90(coh23.phase/2/np.pi/coh23.frequency[np.newaxis,:]))

scale_range = [np.log10(np.min(coh12.scale)), np.log10(np.max(coh12.scale))]

## plot figure
mpl.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['font.size'] = 7

fig = plt.figure(figsize=(12, 8))
gs = GridSpec(5, 2, width_ratios=[50, 1], hspace=0, figure=fig)
lag_min, lag_max = -30, 30

axes = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[1, 0]),
    fig.add_subplot(gs[2, 0]),
    fig.add_subplot(gs[3, 0]),
    fig.add_subplot(gs[4, 0]),
]
ax1, ax2, ax3, ax4, ax5 = axes
cax = fig.add_subplot(gs[1:5, 1])

ax1.axhline(y=0, color='gray', linestyle='-')
ax1.plot(coh12.time, freq1_interp, color='r', label=file1_name[0:2])
ax1.plot(coh23.time, freq2_interp, color='g', label=file2_name[0:2])
ax1.plot(coh23.time, freq3_interp, color='b', label=file3_name[0:2])
ax1.axvspan(time_lb, time_ub, color='gray', alpha=0.3, zorder=1)
ax1.set_ylabel('Freq (Hz)')
ax1.set_xlim(np.min(coh12.time), np.max(coh12.time))
ax1.set_ylim(-1, 1)
ax1.set_xticks([])

im2 = ax2.pcolormesh(*np.meshgrid(coh12.time, coh12.scale), lag12, cmap='bwr', shading='gouraud',
                     vmin=lag_min, vmax=lag_max)
levels = [0.8]
ax2.contour(*np.meshgrid(coh12.time, coh12.scale), np.flipud(np.rot90(coh12.wtc)),
            levels=levels, colors='k', vmin=0, vmax=1)
ax2.axvspan(time_lb, time_ub, color='gray', alpha=0.3, zorder=1)
ax2.axhspan(period_lb, period_ub, color='yellow', alpha=0.5, zorder=1)
ax2.set_yscale('log')
ax2.set_ylabel(f'{file1_name[0:2]} WTC \nScale (s)')
ax2.set_xlim(np.min(coh12.time), np.max(coh12.time))
ax2.set_ylim(np.min(coh12.scale), np.max(coh12.scale))
ax2.set_xticks([])

ax3.pcolormesh(*np.meshgrid(coh13.time, coh13.scale), lag13, cmap='bwr', shading='gouraud',
               vmin=lag_min, vmax=lag_max)
ax3.contour(*np.meshgrid(coh13.time, coh13.scale), np.flipud(np.rot90(coh13.wtc)),
            levels=levels, colors='k', vmin=0, vmax=1)
ax3.axvspan(time_lb, time_ub, color='gray', alpha=0.3, zorder=1)
ax3.axhspan(period_lb, period_ub, color='yellow', alpha=0.5, zorder=1)
ax3.set_yscale('log')
ax3.set_ylabel(f'{file2_name[0:2]} WTC \nScale (s)')
ax3.set_xlim(np.min(coh12.time), np.max(coh12.time))
ax3.set_ylim(np.min(coh13.scale), np.max(coh13.scale))
ax3.set_xticks([])

ax4.pcolormesh(*np.meshgrid(coh23.time, coh23.scale), lag23, cmap='bwr', shading='gouraud',
               vmin=lag_min, vmax=lag_max)
ax4.contour(*np.meshgrid(coh23.time, coh23.scale), np.flipud(np.rot90(coh23.wtc)),
            levels=levels, colors='k', vmin=0, vmax=1)
ax4.axvspan(time_lb, time_ub, color='gray', alpha=0.3, zorder=1)
ax4.axhspan(period_lb, period_ub, color='yellow', alpha=0.5, zorder=1)
ax4.set_yscale('log')
ax4.set_ylabel(f'{file3_name[0:2]} WTC \nScale (s)')
ax4.set_xlim(np.min(coh12.time), np.max(coh12.time))
ax4.set_ylim(np.min(coh23.scale), np.max(coh23.scale))
ax4.set_xticks([])

ax5.axhline(y=0, color='gray', linestyle='-')
ax5.plot(coh12.time, np.sum(sub_wave1_slc, axis=1), color='r', label=file1_name[0:2])
ax5.plot(coh23.time, np.sum(sub_wave2_slc, axis=1), color='g', label=file2_name[0:2])
ax5.plot(coh23.time, np.sum(sub_wave3_slc, axis=1), color='b', label=file3_name[0:2])
ax5.axvspan(time_lb, time_ub, color='gray', alpha=0.3, zorder=1)
ax5.set_xlim(np.min(coh12.time), np.max(coh12.time))
ax5.set_ylabel('Reconstructed \nFreq (Hz)')
ax5.set_ylim(-0.4, 0.4)

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.95, 0.8), frameon=True)

cbar = fig.colorbar(im2, cax=cax)
cbar.set_label('Lag [s]')

xposs = [0.01, 1/4, 2/4, 3/4, 0.99]
xticks = [int(xpos*(sod_end-sod_beg)+sod_beg) for xpos in xposs]
xlabels = [convert_to_HHMM(xtick) for xtick in xticks]
ax5.set_xticks(xticks)
ax5.set_xticklabels(xlabels)

for ax in axes:
    ax.set_xlim(ax1.get_xlim())

plt.show()

db

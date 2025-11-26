import matplotlib.pyplot as plt
import pandas as pd
import pyleoclim as pyleo
import numpy as np
from matplotlib.colors import SymLogNorm, LogNorm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys

sys.path.append(r'E:/Research/Program/else')
import python_utils_JSHEPT
from python_utils_JSHEPT import get_plot_WaveletAnalysis_of_var_vect, wavelet_reconstruction

sys.path.append(r'E:/Research/Program/SuperiorSolarConjunction')
import utils_signal_analyse
from utils_signal_analyse import convert_to_second_of_day, convert_to_HHMM, \
    eliminate_outliers, interpolate, detrend, log_linear_fit

## import data
i_case = 3
save_or_not = 0
if i_case == 1: # 2021/09/30(273), 12:00-13:00, Ht-Sv, Ht-Wz, Sv-Wz, (latitudinal fluctuaion)
    file_dir = 'E:/Research/Data/Tianwen/m1930x_up_new/'
    save_dir = 'E:/Research/Work/tianwen_IPS/m1930x_up_new/'
    file_Ht = 'HtHtchan3_1frephase1s.dat'
    file_Sv = 'SvSvchan3_1frephase1s.dat'
    file_Wz = 'WzWzchan3_1frephase1s.dat'
    fs = 1
    file1_name = file_Ht
    file2_name = file_Sv
    time_beg = 2021273123000 # time interval for plot
    time_end = 2021273130000
    time_lb = 2021273123000 # time interval for calculate spectral index
    time_ub = 2021273123500
elif i_case == 2: # 2021/10/04(277), 05:40-08:20, Js-Bd, Bd-Yg, Yg-Hh, (inward propagation, latitudinal fluctuaion, outward propagation)
    file_dir = 'E:/Research/Data/Tianwen/m1a04x_renew/'
    save_dir = 'E:/Research/Work/tianwen_IPS/m1a04x_renew/'
    file_Js = 'JsJschan3_1frephase1s.dat' # time has been formatted as 'seconds of day'
    file_Bd = 'BdBdchan3_1frephase1s.dat'
    file_Yg = 'YgYgchan3_1frephase4s.dat'
    file_Hh = 'HhHhchan3_1frephase1s.dat'
    fs = 1
    file1_name = file_Js
    file2_name = file_Bd
    time_beg = 2021277080000
    time_end = 2021277083000
    time_lb = 2021277081000
    time_ub = 2021277081500
elif i_case == 3: # 2021/10/07(280), 03:30-04:00, sh-km, (polar region fluctuation)
    file_dir = 'E:/Research/Data/Tianwen/m1a07x_up/'
    save_dir = 'E:/Research/Work/tianwen_IPS/m1a07x_up/'
    file_sh = 'shshchan3_1frephase5s.dat'
    file_km = 'kmkmchan3_1frephase5s.dat'
    fs = 0.2
    file1_name = file_sh
    file2_name = file_km
    time_beg = 2021280033000
    time_end = 2021280040000
    time_lb = 2021280034500
    time_ub = 2021280035000
elif i_case == 4: # 2021/10/15(288), 01:00-04:00, Bd-Hh, Bd-Ys, Hh-Ys, (fine structure)
                  #                  07:40-13:00, Bd-Hh, Bd-Ys, Hh-Ys, (CME)
    file_dir = 'E:/Research/Data/Tianwen/m1a15y_copy/'
    save_dir = 'E:/Research/Work/tianwen_IPS/m1a15y_copy/'
    file_Bd = 'BdBdchan3_1frephase1s.dat'
    file_Hh = 'HhHhchan3_1frephase1s.dat'
    file_Ys = 'YsYschan3_1frephase1s.dat'
    fs = 1
    file1_name = file_Bd
    file2_name = file_Hh
    time_beg = 2021288090000
    time_end = 2021288093000
    time_lb = 2021288090000
    time_ub = 2021288090500
elif i_case == 5: # 2021/10/01(274), 04:40-08:00, km, sh
    file_dir = 'E:/Research/Data/Tianwen/m1a01x_up/'
    save_dir = 'E:/Research/Work/tianwen_IPS/m1a01x_up/'
    file_km = 'kmkmchan3frephase1s.dat'
    file_sh = 'shshchan3frephase1s.dat'
    fs = 1
    file1_name = file_km
    file2_name = file_sh
    time_beg = 2021274073000
    time_end = 2021274080000
    time_lb = 2021274074000
    time_ub = 2021274074500
elif i_case == 6: # 2021/10/10(283), 01:00-03:00, km
    file_dir = 'E:/Research/Data/Tianwen/m1a10x_up/'
    save_dir = 'E:/Research/Work/tianwen_IPS/m1a10x_up/'
    file_km = 'kmkmchan3_1frephase1s.dat'
    fs = 1
    file1_name = file_km
    file2_name = file_km
    time_beg = 2021283023000
    time_end = 2021283030000
    time_lb = 2021283025000
    time_ub = 2021283025500
elif i_case == 7: # 2021/10/12(285), 09:50-11:00, Ht, Wz, Ys
    file_dir = 'E:/Research/Data/Tianwen/m1a12x_up_new/'
    save_dir = 'E:/Research/Work/tianwen_IPS/m1a12x_up_new/'
    file_Ht = 'HtHtchan3_1frephase1s.dat'
    file_Wz = 'WzWzchan3_1frephase1s.dat'
    file_Ys = 'YsYschan3_1frephase1s.dat'
    fs = 1
    file1_name = file_Ht
    file2_name = file_Wz
    time_beg = 2021285100000
    time_end = 2021285103000
    time_lb = 2021285100000
    time_ub = 2021285100500
# convert subplot time interval
str_beg = str(time_lb)[-6:-2]
str_end = str(time_ub)[-6:-2]
sod_beg = convert_to_second_of_day(time_beg)[0]
sod_end = convert_to_second_of_day(time_end)[0]
sod_lb = convert_to_second_of_day(time_lb)[0]
sod_ub = convert_to_second_of_day(time_ub)[0]
    
## read data
file1_path = file_dir + file1_name
file2_path = file_dir + file2_name
data1 = np.loadtxt(file1_path)
data2 = np.loadtxt(file2_path)

## extrcat data
time1 = data1[:, 0]
time2 = data2[:, 0]
sod1 = convert_to_second_of_day(time1)
sod2 = convert_to_second_of_day(time2)
if i_case == 2 and file1_name == 'JsJschan3_1frephase1s.dat':
    sod1 = time1 # time has been formatted as 'seconds of day'
freq1 = data1[:,1]
freq2 = data2[:,1]

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

plt.suptitle('PSD on '+file_dir[25:-1])

if save_or_not == 1:
    plt.savefig(save_dir + 'psd-' + str_beg + '-' + str_end + '.png')

plt.show()

# db

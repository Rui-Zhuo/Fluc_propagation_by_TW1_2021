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

## Selecting station pair and time interval
i_case = 9
period_lb, period_ub = 100, 300
save_or_not = 1
########## Case 01-19 are 2021 Conjunction ##########
if i_case == 1: # 2021/09/15(258), 02:00-04:00, sh-ur
    file_dir = 'E:/Research/Data/Tianwen/m1915x/'
    save_dir = 'E:/Research/Work/tianwen_IPS/m1915x/recon/'
    file_sh = 'shshchan3_1frephase1s.dat' # 00:00-04:00, 1s
    file_ur = 'ururchan3_1frephase1s.dat' # 01:08-04:00, 1s
    file1_name = file_sh
    file2_name = file_ur
    time_beg = 2021258033000
    time_end = 2021258040000
elif i_case == 2: # 2021/09/16(259), 12:00-14:00, Ht-Ys
    file_dir = 'E:/Research/Data/Tianwen/m1916x/'
    save_dir = 'E:/Research/Work/tianwen_IPS/m1916x/recon/'
    file_Ht = 'HtHtchan3_1frephase1s.dat' # 12:00-14:00, 1s/2s
    file_Ys = 'YsYschan3_1frephase1s.dat' # 12:00-14:00, 1s/2s
    file1_name = file_Ht
    file2_name = file_Ys
    time_beg = 2021259124500
    time_end = 2021259130800
elif i_case == 3: # 2021/09/23(266), 05:15-08:15, Ht-km, Ht-ur, km-ur
    file_dir = 'E:/Research/Data/Tianwen/m1923x/'
    save_dir = 'E:/Research/Work/tianwen_IPS/m1923x/recon/'
    file_Ht = 'HtHtchan3_1frephase2s.dat' # 05:15-08:15, 1s/2s
    file_km = 'kmkmchan3_1frephase1s.dat' # 05:15-08:15, 1s/2s
    file_ur = 'ururchan3_1frephase1s.dat' # 05:15-08:15, 1s/2s
    file1_name = file_Ht
    file2_name = file_ur
    time_beg = 2021266051500
    time_end = 2021266054500
elif i_case == 4: # 2021/09/26(269), 02:30-06:10, sh-ur
    file_dir = 'E:/Research/Data/Tianwen/m1926x/'
    save_dir = 'E:/Research/Work/tianwen_IPS/m1926x/recon/'
    file_sh = 'shshchan3frephase.dat' # 02:30-06:10, 5s
    file_ur = 'ururchan3frephase.dat' # 02:30-06:10, 5s
    file1_name = file_sh
    file2_name = file_ur
    time_beg = 2021269054500
    time_end = 2021269061000
elif i_case == 5: # 2021/09/29(272), 05:00-09:00, Js-ur
    file_dir = 'E:/Research/Data/Tianwen/m1929x/'
    save_dir = 'E:/Research/Work/tianwen_IPS/m1929x/recon/'
    file_Js = 'JSfreq.dat'                # 08:00-08:10, 1s, time has been formatted as 'sod', put as file1
    file_ur = 'ururchan3_1frephase1s.dat' # 05:00-09:00, 1s
    file1_name = file_Js
    file2_name = file_ur
    time_beg = 2021272080000
    time_end = 2021272083000
elif i_case == 6: # 2021/09/30(273), 09:00-14:20(12:00-13:00), Ht-Sv, Ht-Wz, Sv-Wz, (latitudinal fluctuaion)
    file_dir = 'E:/Research/Data/Tianwen/m1930x/'
    save_dir = 'E:/Research/Work/tianwen_IPS/m1930x/recon/'
    file_Ht = 'HtHtchan3_1frephase1s.dat' # 09:00-14:20, 1s/2s/5s
    file_Sv = 'SvSvchan3_1frephase1s.dat' # 09:00-14:20, 1s/2s/5s
    file_Wz = 'WzWzchan3_1frephase1s.dat' # 12:20-14:20, 1s/2s/5s
    file1_name = file_Sv
    file2_name = file_Wz
    time_beg = 2021273134500
    time_end = 2021273141200
elif i_case == 7: # 2021/10/01(274), 04:40-07:20, sh-km, sh-Ks, km-Ks
    file_dir = 'E:/Research/Data/Tianwen/m1a01x_up/'
    save_dir = 'E:/Research/Work/tianwen_IPS/m1a01x_up/recon/'
    file_sh = 'shshchan3frephase1s.dat' # 04:40-08:00, 1s
    file_km = 'kmkmchan3frephase1s.dat' # 04:40-08:00, 1s
    file_Ks = 'KSfreq.dat'              # 06:58-09:00, 1s, time has been formatted as 'sod', put as file1
    file1_name = file_sh
    file2_name = file_km
    time_beg = 2021274071500
    time_end = 2021274074500
elif i_case == 8: # 2021/10/03(276), 09:00-13:40(09:30-10:00), Ht-Wz, Ht-Zc, Wz-Zc, (latitudinal fluctuaion)
    file_dir = 'E:/Research/Data/Tianwen/m1a03x/'
    save_dir = 'E:/Research/Work/tianwen_IPS/m1a03x/recon/'
    file_Ht = 'HtHtchan3_1frephase1s.dat' # 09:00-13:40, 1s
    file_Wz = 'WzWzchan3_1frephase1s.dat' # 09:00-13:40, 1s
    file_Zc = 'ZcZcchan3_1frephase1s.dat' # 09:00-13:40, 1s
    file1_name = file_Ht
    file2_name = file_Wz
    time_beg = 2021276090000
    time_end = 2021276093000
elif i_case == 9: # 2021/10/04(277), 05:40-08:20, Js-Bd, Bd-Yg(4s), Yg-Hh(4s); Bd-Hh, Js-Hh, Js-Yg(4s), (inward propagation, latitudinal fluctuaion)
    file_dir = 'E:/Research/Data/Tianwen/m1a04x_renew/'
    save_dir = 'E:/Research/Work/tianwen_IPS/m1a04x_renew/recon/'
    file_Js = 'JsJschan3_1frephase1s.dat' # 07:00-09:00, 1s, time has been formatted as 'sod', put as file1
    file_Bd = 'BdBdchan3_1frephase1s.dat' # 05:40-08:20, 1s/2s/4s
    file_Yg = 'YgYgchan3_1frephase4s.dat' # 05:40-08:20, 4s
    file_Hh = 'HhHhchan3_1frephase1s.dat' # 05:43-08:20, 1s/2s/4s
    file1_name = file_Js
    file2_name = file_Bd
    time_beg = 2021277080000
    time_end = 2021277082000
elif i_case == 10: # 2021/10/05(278), 09:50-12:20, Hh-Mc, Hh-Ys, Mc-Ys
    file_dir = 'E:/Research/Data/Tianwen/m1a05x/'
    save_dir = 'E:/Research/Work/tianwen_IPS/m1a05x/recon/'
    file_Hh = 'HhHhchan3_1frephase1s.dat' # 09:50-12:20, 1s
    file_Mc = 'McMcchan3_1frephase1s.dat' # 09:50-12:20, 1s
    file_Ys = 'YsYschan3_1frephase1s.dat' # 09:50-12:20, 1s
    file1_name = file_Hh
    file2_name = file_Mc
    time_beg = 2021278111500
    time_end = 2021278114500
elif i_case == 11: # 2021/10/07(280), 03:30-05:30(03:30-04:00), sh-km, (polar region fluctuation)
    file_dir = 'E:/Research/Data/Tianwen/m1a07x/'
    save_dir = 'E:/Research/Work/tianwen_IPS/m1a07x/recon/'
    file_sh = 'shshchan3_1frephase5s.dat' # 02:30-08:00, 1s/2s/5s
    file_km = 'kmkmchan3_1frephase5s.dat' # 03:38-09:02, 1s/2s/5s
    file1_name = file_sh
    file2_name = file_km
    time_beg = 2021280034000
    time_end = 2021280040000
elif i_case == 12: # 2021/10/12(285), 09:50-11:00, Ht-Wz, Ht-Ys, Wz-Ys
    file_dir = 'E:/Research/Data/Tianwen/m1a12x/'
    save_dir = 'E:/Research/Work/tianwen_IPS/m1a12x/recon/'
    file_Ht = 'HtHtchan3_1frephase1s.dat' # 09:50-11:00, 1s
    file_Wz = 'WzWzchan3_1frephase1s.dat' # 09:50-14:00, 1s
    file_Ys = 'YsYschan3_1frephase1s.dat' # 09:50-11:00, 1s
    file1_name = file_Ht
    file2_name = file_Wz
    time_beg = 2021285094500
    time_end = 2021285101500
elif i_case == 13: # 2021/10/15(288), 01:00-04:00, sh-km, (fine structure)
    file_dir = 'E:/Research/Data/Tianwen/m1a15x/'
    save_dir = 'E:/Research/Work/tianwen_IPS/m1a15x/recon/'
    file_hb = 'hbhbchan3_1frephase5s.dat' # 00:00-03:59, 5s
    file_sh = 'shshchan3_1frephase1s.dat' # 01:00-04:00, 1s/5s
    file_km = 'kmkmchan3_1frephase1s.dat' # 01:00-04:00, 1s/5s
    file1_name = file_sh
    file2_name = file_km
    time_beg = 2021288024500
    time_end = 2021288031500
elif i_case == 14: # 2021/10/15(288), 07:40-13:00, Bd-Ys, Bd-Hh, Ys-Hh, Js-Bd, Js-Ys, Js-Hh, (CME)
    file_dir = 'E:/Research/Data/Tianwen/m1a15y/'
    save_dir = 'E:/Research/Work/tianwen_IPS/m1a15y/recon/'
    file_Bd = 'BdBdchan3_1frephase1s.dat' # 07:40-09:10, 1s
    file_Ys = 'YsYschan3_1frephase1s.dat' # 07:40-13:00, 1s
    file_Hh = 'HhHhchan3_1frephase1s.dat' # 07:40-13:00, 1s, time has been formatted as 'sod', put as file2
    file_Js = 'JsJschan3_1frephase1s.dat' # 07:00-09:00, 1s, time has been formatted as 'sod', put as file1
    file1_name = file_Bd
    file2_name = file_Hh
    time_beg = 2021288080000
    time_end = 2021288082000
elif i_case == 15: # 2021/10/18(291), 07:40-09:10, Bd-Zc, Hh-Ks, Hh-Ys, Hh-Zc, Ys-Zc, Zc-Ks; Bd-Hh, Bd-Ys, Bd-Ks, Ys-Ks
    file_dir = 'E:/Research/Data/Tianwen/m1a18x/'
    save_dir = 'E:/Research/Work/tianwen_IPS/m1a18x/recon/'
    file_Bd = 'BdBdchan3_1frephase1s.dat' # 04:50-08:55, 1s/2s
    file_Hh = 'HhHhchan3_1frephase1s.dat' # 06:30-09:10, 1s/2s
    file_Ks = 'KsKschan3_1frephase1s.dat' # 01:40-11:00, 1s, time has been formatted as 'sod', put as file2
    file_Ys = 'YsYschan3_1frephase1s.dat' # 07:40-09:10, 1s/2s
    file_Zc = 'ZcZcchan3_1frephase1s.dat' # 06:30-09:10, 1s/2s
    file1_name = file_Ys
    file2_name = file_Zc
    time_beg = 2021291080000
    time_end = 2021291083000
elif i_case == 16: # 2021/10/19(292), 03:00-08:00, sh-km, sh-ur, km-ur; 9min data-7min gap-9min data-7min gap-...
    file_dir = 'E:/Research/Data/Tianwen/s1a19x/'
    save_dir = 'E:/Research/Work/tianwen_IPS/s1a19x/recon/'
    file_sh = 'shshchan3_1frephase.dat' # 03:00-07:57, 0.01s, time has been formatted as 'sod'
    file_km = 'kmkmchan3_1frephase.dat' # 03:00-07:57, 0.01s, time has been formatted as 'sod'
    file_ur = 'ururchan3_1frephase.dat' # 03:00-07:57, 0.01s, time has been formatted as 'sod'
    file1_name = file_km
    file2_name = file_ur
    time_beg = 2021292033300
    time_end = 2021292034200
elif i_case == 17: # 2021/10/20(293), 01:30-08:00, sh-km, sh-ur, km-ur; 14min-16min-16min-16min-...
    file_dir = 'E:/Research/Data/Tianwen/s1a20x/'
    save_dir = 'E:/Research/Work/tianwen_IPS/s1a20x/recon/'
    file_sh = 'shshchan3_1frephase.dat' # 01:30-08:00, 0.01s, time has been formatted as 'sod'
    file_km = 'kmkmchan3_1frephase.dat' # 01:30-05:40, 0.01s, time has been formatted as 'sod'
    file_ur = 'ururchan3_1frephase.dat' # 01:30-08:00, 0.01s, time has been formatted as 'sod'
    file1_name = file_sh
    file2_name = file_ur
    time_beg = 2021293052800
    time_end = 2021293054400
elif i_case == 18: # 2021/10/23(296), 03:30-06:30, sh-km, sh-ur, km-ur; gap point(035600.5, 041329.5)
    file_dir = 'E:/Research/Data/Tianwen/m1a23x/'
    save_dir = 'E:/Research/Work/tianwen_IPS/m1a23x/recon/'
    file_sh = 'shshchan5_1frephase1s.dat' # 03:42-06:30, 1s
    file_km = 'kmkmchan5_1frephase1s.dat' # 03:43-06:30, 1s
    file_ur = 'ururchan5_1frephase1s.dat' # 03:45-06:30, 1s
    file1_name = file_sh
    file2_name = file_ur
    time_beg = 2021296054500
    time_end = 2021296061500
elif i_case == 19: # 2021/10/26(299), 04:40-08:00, sh-km, sh-ur, km-ur; 14min-16min-16min-16min-...
    file_dir = 'E:/Research/Data/Tianwen/s1a26x/'
    save_dir = 'E:/Research/Work/tianwen_IPS/s1a26x/recon/'
    file_sh = 'shshchan3_1frephase.dat' # 04:40-08:00, 0.01s, time has been formatted as 'sod'
    file_km = 'kmkmchan3_1frephase.dat' # 04:40-08:00, 0.01s, time has been formatted as 'sod'
    file_ur = 'ururchan3_1frephase.dat' # 04:40-08:00, 0.01s, time has been formatted as 'sod'
    file1_name = file_sh
    file2_name = file_ur
    time_beg = 2021299073400
    time_end = 2021299075000
########## Case 20-22 are 2023 Conjunction ##########
elif i_case == 20: # 2023/10/28(301), Ht, S6, Sv
    file_dir = 'E:/Research/Data/Tianwen/m3a28x/'
    save_dir = 'E:/Research/Work/tianwen_IPS/m3a28x/recon/'
    file_Ht = 'HTHTsignum113frephase1s.dat' # 04:00-09:00, 1s/2s/5s
    file_S6 = 'S6S6signum113frephase1s.dat' # 04:00-09:00, 1s
    file_Sv = 'SVSVsignum113frephase1s.dat' # 07:50-09:00, 1s/2s/5s
    file1_name = file_Ht
    file2_name = file_S6
    time_beg = 2023301083000
    time_end = 2023301085900
elif i_case == 21: # 2023/11/14(318), CD, Hh, Sv, ur
    file_dir = 'E:/Research/Data/Tianwen/m3b14x/'
    save_dir = 'E:/Research/Work/tianwen_IPS/m3b14x/'
    file_CD = 'CDCDsignum113frephase2s.dat' # 02:00-06:14, 1s/2s/5s
    file_Hh = 'HHHHsignum113frephase2s.dat' # 04:00-08:30, 2s/5s
    file_Sv = 'SVSVsignum113frephase2s.dat' # 08:50-10:10, 2s/5s
    file_ur = 'URURsignum113frephase2s.dat' # 03:00-10:00, 1s/2s/5s
    file1_name = file_Sv
    file2_name = file_ur
    time_beg = 2023318093000
    time_end = 2023318100000
elif i_case == 22: # 2023/11/16(320), Hh, Sv, TM, ur
    file_dir = 'E:/Research/Data/Tianwen/m3b16x/'
    save_dir = 'E:/Research/Work/tianwen_IPS/m3b16x/'
    file_Hh = 'HHHHsignum113frephase2s.dat' # 04:00-07:00, 2s/5s
    file_Sv = 'SVSVsignum113frephase2s.dat' # 09:00-10:00, 2s/5s
    file_TM = 'TMTMsignum113frephase1s.dat' # 02:20-08:00, 1s/2s/5s
    file_ur = 'URURsignum113frephase1s.dat' # 02:20-10:00, 1s/2s/5s
    file1_name = file_TM
    file2_name = file_ur
    time_beg = 2023320023000
    time_end = 2023320030000
    
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
sod1 = convert_to_second_of_day(time1)
sod2 = convert_to_second_of_day(time2)
if file1_name == 'JsJschan3_1frephase1s.dat' or file1_name == 'JSfreq.dat':
    sod1 = time1 # time has been formatted as 'sod'
if file2_name == 'KSfreq.dat':
    sod2 = time2 # time has been formatted as 'sod'
if file_dir[-7:-1] == 'm1a15y' and file2_name == 'HhHhchan3_1frephase1s.dat':
    sod2 = time2 # time has been formatted as 'sod'
if file_dir[-7:-1] == 'm1a18x' and file2_name == 'KsKschan3_1frephase1s.dat':
    sod2 = time2 # time has been formatted as 'sod'
if file_dir[-7:-6] == 's': # PLL data 
    sod1, sod2 = time1, time2 # time has been formatted as 'sod'
freq1 = data1[:,1]
freq2 = data2[:,1]
if i_case == 20 and file2_name == 'S6S6signum113frephase1s.dat':
    freq2 = -freq2 # frequency should take its negative value

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
fig.update_layout(title={'text': 'Wavelet Reconstruction on '+file_dir[25:-1],'x': 0.5, 'y': 0.95})

if save_or_not == 1:
    fig.write_html(save_dir + file1_name[0:2] + '-' + file2_name[0:2] + '-' + str_beg + '-' + str_end + '-Reconstruct.html')
else:
    fig.show()

# db

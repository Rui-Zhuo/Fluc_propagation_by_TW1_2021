import numpy as np
from scipy.stats import zscore
from scipy.interpolate import interp1d, CubicSpline

## function: convert the time data to 'seconds of day'
def convert_to_second_of_day(time_array):
    if isinstance(time_array, int):
        time_array = [time_array]
    sod_array = []
    for time in time_array:
        year = time // 1000000000
        doy = (time % 1000000000) // 1000000
        hour = (time % 1000000) // 10000
        minute = (time % 10000) // 100
        second = time % 100
        sod = hour * 3600 + minute * 60 + second
        sod_array.append(sod)
    return np.array(sod_array)

## function: convert the 'seconds of day' data to 'HHMM' for plot
def convert_to_HHMM(sod):
    HH = int(sod) // 3600
    MM = (int(sod) % 3600) // 60
    HH_str = str(HH).zfill(2)
    MM_str = str(MM).zfill(2)
    HHMM = HH_str + MM_str
    return HHMM

## function: eliminate outliers with deviation > threshold*std_error
def eliminate_outliers(freq, time, threshold):
    freq_zscore = zscore(freq)
    outliers = np.abs(freq_zscore) > threshold
    outliers[0] = False
    outliers[-1] = False # retain beginning/end data for interpolation
    freq = freq[~outliers]
    time = time[~outliers]
    return freq, time

# function: interpolate frequency time series
def interpolate(freq, time, time_std, method='linear'):
    if method == 'linear':
        f = interp1d(time, freq)
    elif method == 'quad':
        f = interp1d(time, freq, kind='quadratic')
    elif method == 'cubic':
        f = CubicSpline(time, freq)
    freq_std = f(time_std)
    return freq_std

## function: detrend frequency time series and plot it
def detrend(freq, time, order):
    coef = np.polyfit(time, freq, order)
    freq_fit = np.polyval(coef, time)
    freq_detrended = freq - freq_fit
    return freq_fit, freq_detrended

## function: calculate PSD spectral index
def log_linear_fit(freq, psd, freq_thres):
    freq_calc_ind = np.where((freq > freq_thres))[0]
    freq_calc = freq[freq_calc_ind]
    psd_calc = psd[freq_calc_ind]
    log_freq = np.log10(freq_calc)
    log_psd = np.log10(psd_calc)
    coef = np.polyfit(log_freq, log_psd, 1)
    slope, intercept = coef[0], coef[1]
    log_psd_fit = np.polyval(coef, log_freq)
    psd_fit = 10**log_psd_fit
    return slope, freq_calc, psd_fit
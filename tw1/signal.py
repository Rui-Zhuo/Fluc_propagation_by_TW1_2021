"""Shared signal preprocessing and spectral fitting."""
import numpy as np
from scipy.stats import zscore
from scipy.interpolate import interp1d, CubicSpline

## function: eliminate outliers with deviation > threshold*std_error
def eliminate_outliers(freq, time, threshold):
    if len(freq) < 2:
        raise ValueError('Selected interval contains fewer than two samples; check UTC range and time format')
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
    else:
        raise ValueError(f'Unknown interpolation method: {method}')
    freq_std = f(time_std)
    return freq_std

## function: detrend frequency time series and plot it
def detrend(freq, time, order):
    if len(time) <= order:
        raise ValueError(f'Polynomial order {order} needs at least {order + 1} selected samples')
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

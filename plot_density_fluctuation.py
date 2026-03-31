import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

file_path = 'E:/Research/Work/tianwen_IPS/multiple_baseline.xlsx'
sheet_name = 'density_fluc'

excel_file = pd.ExcelFile(file_path)
df = excel_file.parse(sheet_name)

case_ind = df['case']
rs = df['rs']
vp = df['vp']
station = df['station']
f_prime = df['f_prime']
N_prime = df['N_prime']
n_prime = df['n_prime']
n0 = df['n0']
npn0 = df['npn0']
method = df['method']

mask_peak = (method == 'peak')
mask_inte = (method == 'integral')

rs_peak = rs[mask_peak]
rs_inte = rs[mask_inte]
vp_peak = vp[mask_peak]
vp_inte = vp[mask_inte]
n0_peak = n0[mask_peak]
n0_inte = n0[mask_inte]
f_prime_peak, N_prime_peak, n_prime_peak, npn0_peak = \
    f_prime[mask_peak], N_prime[mask_peak], n_prime[mask_peak], npn0[mask_peak]
f_prime_inte, N_prime_inte, n_prime_inte, npn0_inte = \
    f_prime[mask_inte], N_prime[mask_inte], n_prime[mask_inte], npn0[mask_inte]

plt.figure()

plt.subplot(221)
plt.scatter(rs_peak, f_prime_peak, color='r', label='peak method')
plt.scatter(rs_inte, f_prime_inte, color='b', label='integral method')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.xlabel('Heliocentric distance [Rs]')
plt.ylabel('Frequency fluctuation [Hz]')

plt.subplot(222)
plt.scatter(rs_peak, N_prime_peak, color='r')
plt.scatter(rs_inte, N_prime_inte, color='b')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Heliocentric distance [Rs]')
plt.ylabel('STEC fluctuation [m-2]')

plt.subplot(223)
plt.scatter(rs_peak, n_prime_peak, color='r')
plt.scatter(rs_inte, n_prime_inte, color='b')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Heliocentric distance [Rs]')
plt.ylabel('Density fluctuation [cm-3]')

plt.subplot(224)
plt.scatter(rs_peak, npn0_peak, color='r')
plt.scatter(rs_inte, npn0_inte, color='b')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Heliocentric distance [Rs]')
plt.ylabel('Fractional density fluctuation')

def calc_mean_std(arr):
    mean_val = np.mean(arr)
    std_val = np.std(arr)
    return mean_val, std_val

def log_poly__fit(x, y, n=1):
    x_arr = np.array(x)
    y_arr = np.array(y)
    x_arr_log = np.log10(x_arr)
    y_arr_log = np.log10(y_arr)
    coef = np.polyfit(x_arr_log, y_arr_log, n)
    slope_log, intercept_log = coef
    y_fit_log = slope_log * x_arr_log + intercept_log
    corr_coef, p_value = stats.pearsonr(x_arr_log, y_arr_log)
    
    slope = slope_log
    intercept = 10 ** intercept_log
    y_fit = 10 ** y_fit_log
    
    return slope, intercept, y_fit, corr_coef, p_value

rs_inte_arr = np.array(rs_inte)
rs_12, rs_3, rs_45 = rs_inte_arr[0], rs_inte_arr[2*3], rs_inte_arr[3*3]

n_prime_inte_arr = np.array(n_prime_inte)
n_prime_12, n_prime_3, n_prime_45 = n_prime_inte_arr[:2*3], n_prime_inte_arr[2*3:3*3], n_prime_inte_arr[3*3:]
n_prime_mean_12, n_prime_std_12 = calc_mean_std(n_prime_12)
n_prime_mean_3, n_prime_std_3 = calc_mean_std(n_prime_3)
n_prime_mean_45, n_prime_std_45 = calc_mean_std(n_prime_45)

npn0_inte_arr = np.array(npn0_inte)
npn0_12, npn0_3, npn0_45 = npn0_inte_arr[:2*3], npn0_inte_arr[2*3:3*3], npn0_inte_arr[3*3:]
npn0_mean_12, npn0_std_12 = calc_mean_std(npn0_12)
npn0_mean_3, npn0_std_3 = calc_mean_std(npn0_3)
npn0_mean_45, npn0_std_45 = calc_mean_std(npn0_45)

n_prime_slope, n_prime_intercept, n_prime_fit, n_prime_corr, n_pirme_p \
    = log_poly__fit([rs_12, rs_3, rs_45], [n_prime_mean_12, n_prime_mean_3, n_prime_mean_45])
npn0_slope, npn0_intercept, npn0_fit, npn0_corr, npn0_p \
    = log_poly__fit([rs_12, rs_3, rs_45], [npn0_mean_12, npn0_mean_3, npn0_mean_45])
    
plt.figure()
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.size'] = 16

def mark_text(ax, slope, intercept, corr_coef, p_value):
    stats_text = (f'slope = {slope:.2f}\n'
                  f'intercept = {intercept:.0f}\n'
                  f'corr.coef. = {corr_coef:.4f}')
    ax.text(0.05, 0.05, stats_text, transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    return 

ax1 = plt.subplot(121)
plt.scatter(rs_inte, n_prime_inte, color='b')
plt.plot([rs_3, rs_12, rs_45], [n_prime_fit[1], n_prime_fit[0], n_prime_fit[2]], color='k')
plt.errorbar(rs_12, n_prime_mean_12, yerr=n_prime_std_12, fmt='o', color='r', capsize=10)
plt.errorbar(rs_3,  n_prime_mean_3,  yerr=n_prime_std_3,  fmt='o', color='r', capsize=10)
plt.errorbar(rs_45, n_prime_mean_45, yerr=n_prime_std_45, fmt='o', color='r', capsize=10)
mark_text(ax1, n_prime_slope, n_prime_intercept, n_prime_corr, n_pirme_p)
plt.xlim([5,20])
plt.ylim([1e2,1e4])
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Heliocentric distance [Rs]')
plt.ylabel('Density fluctuation [cm-3]')

ax2 = plt.subplot(122)
plt.scatter(rs_inte, npn0_inte, color='b')
# plt.plot([rs_3, rs_12, rs_45], [npn0_fit[1], npn0_fit[0], npn0_fit[2]], color='k')
plt.errorbar(rs_12, npn0_mean_12, yerr=npn0_std_12, fmt='o', color='r', capsize=10)
plt.errorbar(rs_3,  npn0_mean_3,  yerr=npn0_std_3,  fmt='o', color='r', capsize=10)
plt.errorbar(rs_45, npn0_mean_45, yerr=npn0_std_45, fmt='o', color='r', capsize=10)
# mark_text(ax2, npn0_slope, npn0_intercept, npn0_corr, n_pirme_p)
plt.xlim([5,20])
plt.ylim([1e-2,1])
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Heliocentric distance [Rs]')
plt.ylabel('Fractional density fluctuation')

def calc_energe_flux(n0, vp, npn0):
    '''
    Calculate the energe flux based on the formula provided by Maoli Ma
    
    Input:
    - n0: background density, in [cm-3]
    - vp: phase velocity, in [km/s]
    '''
    mp = 1.6726231e-27
    n0_SI = n0 * 1e6
    vp_SI = vp * 1e3
    Fe = 1/2 * n0_SI * mp * vp_SI ** 3 * npn0 ** 2 # [W/m2] 
    
    return Fe

Fe_inte = calc_energe_flux(n0_inte, vp_inte, npn0_inte)
Fe_inte_arr = np.array(Fe_inte)
Fe_12, Fe_3, Fe_45 = Fe_inte_arr[:2*3], Fe_inte_arr[2*3:3*3], Fe_inte_arr[3*3:]
Fe_mean_12, Fe_std_12 = calc_mean_std(Fe_12)
Fe_mean_3, Fe_std_3 = calc_mean_std(Fe_3)
Fe_mean_45, Fe_std_45 = calc_mean_std(Fe_45)

Fe_slope, Fe_intercept, Fe_fit, Fe_corr, Fe_p \
    = log_poly__fit([rs_12, rs_3, rs_45], [Fe_mean_12, Fe_mean_3, Fe_mean_45])
    
plt.figure()
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.size'] = 16

ax1 = plt.subplot(111)
plt.scatter(rs_inte, Fe_inte, color='b')
# plt.plot([rs_3, rs_12, rs_45], [Fe_fit[1], Fe_fit[0], Fe_fit[2]], color='k')
plt.errorbar(rs_12, Fe_mean_12, yerr=Fe_std_12, fmt='o', color='r', capsize=10)
plt.errorbar(rs_3,  Fe_mean_3,  yerr=Fe_std_3,  fmt='o', color='r', capsize=10)
plt.errorbar(rs_45, Fe_mean_45, yerr=Fe_std_45, fmt='o', color='r', capsize=10)
# mark_text(ax1, Fe_slope, Fe_intercept, Fe_corr, n_pirme_p)
plt.xlim([5,20])
# plt.ylim([1e2,1e4])
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Heliocentric distance [Rs]')
plt.ylabel('Energe Flux [W/m-2]')

plt.show()

db
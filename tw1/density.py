"""Density fluctuation calculations and the original diagnostic plotting helpers."""
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from astropy.coordinates import get_sun, get_body_barycentric, solar_system_ephemeris

def calculate_solar_offset(time):
    with solar_system_ephemeris.set('builtin'):
        sun_position = get_sun(time)
        earth_position = get_body_barycentric('earth', time)
        mars_position = get_body_barycentric('mars', time)
    # calculate vector of Sun to Earth and vector of Mars to Earth
    vec_S2E = sun_position.cartesian.xyz - earth_position.xyz
    vec_M2E = mars_position.xyz - earth_position.xyz
    # calculate including angle between Sun and Mars to Earth
    angle_SEM = np.arccos(np.dot(vec_S2E, vec_M2E) / (np.linalg.norm(vec_S2E) * np.linalg.norm(vec_M2E)))
    solar_offset = np.linalg.norm(vec_S2E) * np.sin(angle_SEM)
    AU = 1.49597871e11
    solar_radius = 6.955e8
    solar_offset_Rs = solar_offset.value * AU / solar_radius
    print('solar_offset: ', round(solar_offset_Rs,3))
    return solar_offset_Rs

def linear_detrend(phase_sequence, time, title, show=True):
    coefficients = np.polyfit(time, phase_sequence, 1)
    fit_sequence = np.polyval(coefficients, time)
    detrended_sequence = phase_sequence - fit_sequence
    plt.figure()
    plt.plot(time,phase_sequence,label='phase')
    plt.plot(time,fit_sequence,label='fit')
    plt.legend()
    plt.title(title)
    if show:
        plt.show()
    return detrended_sequence

def quadratic_detrend(frequency_sequence,time,title, show=True):
    frequency_mean = np.mean(frequency_sequence)
    frequency_sequence[np.abs(frequency_sequence-frequency_mean)>2] = 0
    coefficients = np.polyfit(time, frequency_sequence, 2)
    fit_sequence = np.polyval(coefficients, time)
    detrended_sequence = frequency_sequence - fit_sequence
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    axs[0].plot(time, frequency_sequence)
    axs[0].plot(time, fit_sequence)
    axs[0].set_ylabel('Frequency [Hz]')
    axs[0].legend()
    axs[1].plot(time, detrended_sequence)
    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel('Detrended Frequency [Hz]')
    plt.suptitle(title)
    if show:
        plt.show()
    return detrended_sequence

def cal_sigma_Ne(sigma_FM, nu_c, L_LOS, r):
    re = 2.8179e-15 # electron radius, [m]
    Rs = 6.955e8 # solar radius, [m]
    R = r*Rs
    sigma_Ne = sigma_FM / re / nu_c / np.sqrt(L_LOS*R)
    print('sigma_Ne: ', round(sigma_Ne,3))
    return sigma_Ne

def cal_Ne(r):
    Ne = (30/r**6 + 1/r**2.2) * 1e12 # model
    return Ne

def cal_L_LOS(r):
    L_LOS = 3.35e6*r**0.918
    return L_LOS

def cal_sigma_FM(FM, freq, nu_a, nu_b):
    FM2 = FM ** 2
    def integrand(x):
        return np.interp(x, freq, FM2)
    integrate_result = integrate.quad(integrand, nu_a, nu_b)
    sigma_FM = np.sqrt(integrate_result[0])
    print('sigma_FM: ', round(sigma_FM,3))
    return sigma_FM

def cal_nu_c(beta, nu_a, nu_b):
    nu_c = np.sqrt((beta+1)/(beta-1) * (nu_b**(1-beta)-nu_a**(1-beta))/(nu_b**(-1-beta)-nu_a**(-1-beta)))
    print('nu_c: ', round(nu_c,3))
    return nu_c

def cal_FM(FF2, freq_band):
    c = 3e8
    lambda_band = c / freq_band
    FM = np.sqrt(FF2) / lambda_band
    return FM

def cal_beta(FF2, freq, nu_a, nu_b, show=True, unit_name='Hz'):
    indices_sub = (freq >= nu_a) & (freq <= nu_b)
    freq_sub = freq[indices_sub]
    FF2_sub = FF2[indices_sub]
    fit_coef = np.polyfit(np.log10(freq_sub), np.log10(FF2_sub), deg=1)
    FF2_log10_fit = np.polyval(fit_coef, np.log10(freq_sub))
    FF2_fit = np.power(10, FF2_log10_fit)
    slope = round(fit_coef[0],3)
    beta = -slope
    print('beta: ', beta)
    plt.figure()
    plt.plot(freq, FF2)
    plt.plot(freq_sub, FF2_fit)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Spectral Density [' + unit_name +'^2/Hz]')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Frequency Fluctuation PSD')
    plt.annotate(f'slope={slope}', xy=(1e-3,1e-3), xytext=(1e-3,1e-3))
    if show:
        plt.show()
    return beta

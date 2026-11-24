"""calc density fluctuation. Use main() or python run_analysis.py."""

def main(case=None, start=None, end=None, save=None, plot=None, file1=None, file2=None, show=True):
    """Run locally; times are UTC HH:MM[:SS] or YYYYDDDHHMMSS."""
    from tw1.density import calculate_solar_offset, quadratic_detrend, cal_sigma_Ne, cal_Ne, cal_L_LOS, cal_sigma_FM, cal_nu_c, cal_FM, cal_beta
    from project_paths import data_directory
    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.time import Time

    file_dir = data_directory('m1a04x_renew/')
    file_name = 'BdBdchan3_1frephase1s.dat'
    var_list = ['Residual Frequency', 'Residual Phase', 'Signal Density', 'Noise Density']
    unit_list = ['Hz', 'rad', 'dB', 'dB']

    file_path = file_dir + file_name
    data = np.loadtxt(file_path)
    time = np.linspace(500, len(data), len(data))
    slt_indices = np.arange(0,len(data))
    time = time[slt_indices]
    var = data[slt_indices, 1:]
    time = time - time[0]

    for i in range(var.shape[1]):
        if i == 0: # only for frequency sequence
            signal = var[:, i]
            var_name = var_list[i] if i < len(var_list) else f'variable {i + 1}'
            unit_name =  unit_list[i] if i < len(unit_list) else ''
            title = file_path[24:] + ' --- ' + var_name

            # quadratic
            signal = quadratic_detrend(signal, time, title, show=show)

            # calculate FTT
            n = len(time)
            dt = time[1] - time[0]
            fs = 1/dt
            print(round(dt,3))
            fft_freq = np.fft.fftfreq(n, dt)
            fft_amp = np.fft.fft(signal)
            fft_amp = fft_amp[fft_freq > 0]
            FF2 = np.abs(fft_amp) ** 2 / (n/fs)
            fft_freq = fft_freq[fft_freq > 0]

            # calculate spectral index beta
            nu_Nyq = 1/(2*dt)
            nu_a = 1e-3
            nu_b = 0.1 * nu_Nyq
            beta = cal_beta(FF2, fft_freq, nu_a, nu_b, show=show)

            # calculate scaling frequency factor
            nu_c = cal_nu_c(beta, nu_a, nu_b)

            # calcualte fluctuation measure FM
            freq_band = 8.4e9
            FM = cal_FM(FF2, freq_band)
            plt.figure()
            plt.plot(fft_freq, FM**2)
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Flctuation Measure [' + unit_name +'^2/Hz]')
            plt.xscale('log')
            plt.yscale('log')
            plt.title('Fluctuation Measure PSD')
            if show:
                plt.show()

            # calculate Ne and L_LOS based on models
            UTC_time = Time('2021-10-04T12:00:00.00')
            r = calculate_solar_offset(UTC_time)
            r = 4.94
            Ne = cal_Ne(r)
            L_LOS = cal_L_LOS(r)

            # calculate sigma_FM and sigma_Ne
            sigma_FM = cal_sigma_FM(FM, fft_freq, nu_a, nu_b)
            sigma_Ne = cal_sigma_Ne(sigma_FM, nu_c, L_LOS, r)

            # calculate fractional density fluctuations
            epsilon = sigma_Ne / Ne
            print('epsilon: ', round(epsilon,3))

"""plot wavelet spectrum. Use main() or python run_analysis.py."""

def main(case=None, start=None, end=None, save=None, plot=None, file1=None, file2=None, show=True):
    """Run locally; times are UTC HH:MM[:SS] or YYYYDDDHHMMSS."""
    import numpy as np
    import matplotlib.pyplot as plt
    import pywt
    from matplotlib.colors import LogNorm

    # import data
    file_type = 4 if case is None else int(case)
    from tw1.cases import get_case
    config = get_case('plot_wavelet_spectrum', file_type, start=start, end=end, file1=file1, file2=file2)
    file_dir = config.input_directory
    save_dir = config.output_directory(create=False)
    file_name = config.parameters['file_name']
    unit_list = config.parameters['unit_list']
    var_list = config.parameters['var_list']

    file_path = file_dir + file_name
    data = np.loadtxt(file_path)
    time = np.linspace(0, len(data), len(data))
    slt_indices = np.arange(500,len(data))
    time = time[slt_indices]
    var = data[slt_indices, 1:]
    if file_type == 2:
        time = data[:, 1]
        var = data[:, 2:]
    time = time - time[0]
    if file_type == 0:
        time = np.linspace(0,int(file_name[19])*(len(time)-1),len(time))
    if file_type == 1:
        var = var[:, 1:]

    def eliminate_weekly_ambiguity(phase_sequence):
        for i in range(1, len(phase_sequence)):
            if phase_sequence[i] > phase_sequence[i - 1]:
                phase_sequence[i:] -= 2 * np.pi
        return phase_sequence

    def linear_detrend(phase_sequence, time, title):
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

    def quadratic_detrend(frequency_sequence,time,title):
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

    for i in range(var.shape[1]):
        signal = var[:, i]
        var_name = var_list[i] if i < len(var_list) else f'variable {i + 1}'
        unit_name =  unit_list[i] if i < len(unit_list) else ''
        title = file_path[24:] + ' --- ' + var_name
    
        # eliminate ambiguity for phase sequence
        if file_type == 0 and i == 1:
            signal = eliminate_weekly_ambiguity(signal)
        elif file_type == 4 and i == 1:
            signal = eliminate_weekly_ambiguity(signal)
    
        # linear detrend for phase sequence
        if file_type == 3 and (i == 1 or i == 3):
            signal = linear_detrend(signal, time, title)
        elif file_type == 4 and i == 1:
            signal = linear_detrend(signal, time, title)
    
        # quadratic detrend for frequency sequence
        if file_type == 3 and (i == 0 or i == 2):
            signal = quadratic_detrend(signal, time, title)
        elif file_type == 4 and i == 0:
            signal = quadratic_detrend(signal, time, title)

        # calculate FTT
        n = len(time)
        dt = time[1] - time[0]
        fs = 1/dt
        print(dt)
        fft_freq = np.fft.fftfreq(n, dt)
        fft_amp = np.fft.fft(signal)
        fft_amp = fft_amp[fft_freq > 0]
        fft_freq = fft_freq[fft_freq > 0]
        fft_ps = np.abs(fft_amp)**2 / (n/fs)

        # calculate CWT
        wavename = 'cmor'
        num_freq = 50
        wave_freq = np.logspace(-4, np.log10(1/(2*dt)), num_freq)
        if file_type == 1:
            wave_freq = np.logspace(-1, 0.8*np.log10(1/(2*dt)), num_freq)
        scales = 1 / wave_freq
        cwtmatr, cwt_freq = pywt.cwt(signal, scales, wavename, 1.0 / dt)
        cwt_ps = np.abs(cwtmatr)**2 / (n/fs)

        # plot figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 8), gridspec_kw={'height_ratios': [3, 5], 'width_ratios': [5, 4]})

        # time series
        axes[0, 0].plot(time, signal)
        axes[0, 0].set_xlabel('Time [s]')
        axes[0, 0].set_ylabel(var_name + '[' + unit_name + ']') 
        axes[0, 0].set_xlim(time.min(), time.max())
        axes[0, 0].tick_params(direction='in')
        axes[0, 0].set_title('Time Series')

        # wavelet transform
        pcwt = axes[1, 0].pcolormesh(time, cwt_freq, cwt_ps, cmap='jet', norm=LogNorm())
        axes[1, 0].set_title('Wavelet Transform')
        axes[1, 0].set_ylabel('Frequency [Hz]')
        axes[1, 0].set_yscale('log')
        cbar = plt.colorbar(pcwt, ax=axes[1, 0], orientation='horizontal', pad=0.1)
    
        # no right-upper subplot
        axes[0, 1].axis('off')
    
        # fourier transform
        axes[1, 1].plot(fft_freq, fft_ps)
        axes[1, 1].set_xlabel('Frequency [Hz]')
        axes[1, 1].set_ylabel('Spectral Density [' + unit_name +'^2/Hz]')
        axes[1, 1].set_xscale('log')
        axes[1, 1].set_yscale('log')
        axes[1, 1].tick_params(direction='in')
        axes[1, 1].set_title('Fourier Transform')

        fig.suptitle(title)
        # plt.savefig(result_directory() + f"{file_name[0:2]}_{var_name}.png")
        if show:
            plt.show()


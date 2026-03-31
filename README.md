# SuperiorSolarConjunction

This repository contains the codes and results for the analysis of compressional waves propagation based on the radio sounding of Tianwen-1 and multiple ground-based radio telescopes. It is maintained by Rui Zhuo (ruizhuo@pku.edu.cn)

the manuscript titled *Radio Sounding of Compressional Waves Propagation and Reflection in Solar Wind Acceleration Region* submitted to *Nature Astronomy* by Maoli Ma, Rui Zhuo, Ziqi Wu and Liangliang Yuan. The emails of corresponding authors are jshept@pku.edu.cn and mamaoli@shao.ac.cn.

- *analyse_wavelet_coherence.py*: given the radio signals from two stations, calculate the coherency spectrum and time-lag spectrum between them, and obtain the propagation speed along the baseline finnally.

- *reconstruct_wavelet.py*: given the wavelet coefficients and base functions, reconstruct the time series of the signal.

- *calc_solar_offset.py*: given the UTC time and two stations, calculate the projected points on the sky plane, define the baseline, and calculate the baseline length and the solar offset. 

- *calc_power_index.py*: given the radio signal from one station, calculate its PSD spectrum and power index.

- *calc_density_fluctuation.py*: given the power spectrum of one station, calculate the density fluctuation. 

- *statisticize_wavelet_coherence.py*: given the dataset of wavelet coherence, statisticize the dependency of propagation speed on the solar offset and temperal scale. 

- *analyse_multiple_baseline.py*: given the dataset of the *identical* density fluctuations and multiple baselines, show their relationships, and obtain the 2-D propagation velocity on the sky plane. 

- *wavelet_coherence.xlsx*: the dataset of wavelet coherence and propagation speeds along the baselines. 

- *multiple_baseline.xlsx*: the dataset of the *identical* density fluctuations observed by multiple baselines.

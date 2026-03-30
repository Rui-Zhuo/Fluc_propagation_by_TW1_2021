import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib as mpl

# Importing data
file_dir = 'E:/Research/Work/Fluc_propagation_by_TW1_2021/'
file_name = 'wavelet_coherence.xlsx'

data = pd.ExcelFile(file_dir + file_name)
df_dis = pd.read_excel(data, sheet_name='distribution', header=0)
# print('df_dis columns: ', df_dis.columns)
df_coh = pd.read_excel(data, sheet_name='coherency', header=0)
# print('df_coh columns: ', df_coh.columns)

date, stations, length, acute_angle = df_coh['date'], df_coh['stations'], df_coh['length'], df_coh['acute_angle']
int_time = df_coh['integral_time']
coh_time, scale = df_coh['coh_time'], df_coh['scale']
lag, vel, sign, vel_type  = df_coh['lag'], df_coh['vel'], df_coh['sign'], df_coh['type']

# Eliminating fallible data with 'lag < 3 * integral_time'
for i_case in range(len(lag)):
    if np.abs(lag[i_case]) < 3 * int_time[i_case]:
        vel[i_case] = np.nan

# Defining radial-positive velocity
vel_sign = vel * sign

# Dividing acute angle into three bins
bins = [0, 30, 60, 90]
scale, vel_sign = np.array(scale), np.array(vel_sign)
# Quasi-radial bin
ind_qr = np.where((acute_angle > bins[0]) & (acute_angle < bins[1]))
scale_qr, vel_qr = scale[ind_qr], vel_sign[ind_qr]
# Inclined bin
ind_ic = np.where((acute_angle > bins[1]) & (acute_angle < bins[2]))
scale_ic, vel_ic = scale[ind_ic], vel_sign[ind_ic]
# Quasi-latitudinal bin
ind_qt = np.where((acute_angle > bins[2]) & (acute_angle < bins[3]))
scale_qt, vel_qt = scale[ind_qt], vel_sign[ind_qt]

# Counting inward and outward propagation
num_outward = np.sum(np.array(vel_qr) > 0) # 276 cases
num_inward = np.sum(np.array(vel_qr) < 0) # 38 cases
outward_perc = num_outward / (num_outward + num_inward)# 87.9%

# Extracting inward propagation
cases_inward = np.arange(0, num_inward)
ind_inward = np.where((vel_qr) < 0)
scale_inward, vel_inward = scale_qr[ind_inward], np.abs(vel_qr[ind_inward])

width_inward = vel_inward * scale_inward

# Plotting inward cases
plt.figure()

plt.subplot(2,2,1)
plt.hist(vel_inward, bins=10)
plt.xlabel('velocity of inward propagation [km/s]')
plt.ylabel('counts')

plt.subplot(2,2,2)
plt.hist(scale_inward, bins=10)
plt.xlabel('period of inward propagation [s]')
plt.ylabel('counts')

plt.subplot(2,2,3)
plt.scatter(vel_inward, scale_inward, c=cases_inward, cmap='cool')
cb = plt.colorbar()
cb.set_label('counts')
plt.xlabel('velocity of inward propagation [km/s]')
plt.ylabel('period of inward propagation [s]')

plt.subplot(2,2,4)
plt.hist(width_inward, bins=10)
plt.xlabel('spatial scale of inward propagation [km]')
plt.ylabel('counts')

plt.show()

db
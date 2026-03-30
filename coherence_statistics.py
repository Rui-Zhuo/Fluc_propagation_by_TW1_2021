import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib as mpl
import matplotlib.ticker as ticker 
from datetime import datetime

# Importing data
file_dir = 'E:/Research/Work/Fluc_propagation_by_TW1_2021/'
file_name = 'wavelet_coherence.xlsx'

data = pd.ExcelFile(file_dir + file_name)
df_dis = pd.read_excel(data, sheet_name='distribution', header=0)
# print('df_dis columns: ', df_dis.columns)
df_coh = pd.read_excel(data, sheet_name='coherency', header=0)
# print('df_coh columns: ', df_coh.columns)

date, so = df_coh['date'], df_coh['solar_offset']
stations, length, acute_angle = df_coh['stations'], df_coh['length'], df_coh['acute_angle']
int_time = df_coh['integral_time']
coh_time, scale = df_coh['coh_time'], df_coh['scale']
lag, vel, sign, vel_type  = df_coh['lag'], df_coh['vel'], df_coh['sign'], df_coh['type']

def convert_to_epoch(date, coh_time, year):
    date_str = str(date).zfill(4)
    time_str = str(coh_time).zfill(4)
    
    month = int(date_str[:2])
    day = int(date_str[2:])
    hour = int(time_str[:2])
    minute = int(time_str[2:])
    
    dt = datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=0)
    epoch = int(dt.timestamp())
    
    return epoch

incoh_epoch_1 = convert_to_epoch('1019', '0000', 2021)
incoh_epoch_2 = convert_to_epoch('1019', '0410', 2021) # between 1-2 is coherent state (should be removed)
incoh_epoch_3 = convert_to_epoch('1019', '0540', 2021)
incoh_epoch_4 = convert_to_epoch('1019', '2359', 2021) # between 3-4 is coherent state (should be removed)
incoh_epoch_5 = convert_to_epoch('1020', '0000', 2021)
incoh_epoch_6 = convert_to_epoch('1020', '0120', 2021) # between 5-6 is coherent state (should be removed)
incoh_epoch_7 = convert_to_epoch('1020', '0240', 2021)
incoh_epoch_8 = convert_to_epoch('1020', '0500', 2021) # between 7-8 is coherent state (should be removed)
incoh_epoch_9 = convert_to_epoch('1020', '0540', 2021)
incoh_epoch_10 = convert_to_epoch('1020', '2359', 2021) # between 9-10 is coherent state (should be removed)

# Eliminating fallible data
for i_case in range(len(lag)):
    
    if np.abs(lag[i_case]) < int_time[i_case]: # lag < integral_time
        vel[i_case] = np.nan
    
    # if scale[i_case] < 100: # scale < 100s
    #     vel[i_case] = np.nan
    
    epoch = convert_to_epoch(date[i_case], coh_time[i_case], 2021)
    if (incoh_epoch_1 < epoch < incoh_epoch_2 or
        incoh_epoch_3 < epoch < incoh_epoch_4 or
        incoh_epoch_5 < epoch < incoh_epoch_6 or
        incoh_epoch_7 < epoch < incoh_epoch_8 or
        incoh_epoch_9 < epoch < incoh_epoch_10):# coherent state
        vel[i_case] = np.nan

# Defining radial-positive velocity
vel_sign = vel * sign * np.cos(np.deg2rad(acute_angle))

# Dividing acute angle into three bins
bins = [0, 30, 60, 90]
date, stations, acute_angle, coh_time = np.array(date), np.array(stations), np.array(acute_angle), np.array(coh_time)
so, scale, vel_sign = np.array(so), np.array(scale), np.array(vel_sign)
# Quasi-radial bin
ind_qr = np.where((acute_angle > bins[0]) & (acute_angle < bins[1]))
so_qr, scale_qr, vel_qr = so[ind_qr], scale[ind_qr], vel_sign[ind_qr]
ind_qri_qr = np.where(vel_qr < 0)[0]
ind_qri = ind_qr[0][ind_qri_qr]
# Inclined bin
ind_ic = np.where((acute_angle > bins[1]) & (acute_angle < bins[2]))
so_ic, scale_ic, vel_ic = so[ind_ic], scale[ind_ic], vel_sign[ind_ic]
# Quasi-latitudinal bin
ind_qt = np.where((acute_angle > bins[2]) & (acute_angle < bins[3]))
so_qt, scale_qt, vel_qt = so[ind_qt], scale[ind_qt], vel_sign[ind_qt]

# Counting inward and outward propagation
num_outward = np.sum(np.array(vel_qr) > 0)
num_inward = np.sum(np.array(vel_qr) < 0)
outward_perc = num_outward / (num_outward + num_inward) # 84.6%

############################################################################################

def plot_distrib_hist2d(fig, ax, x, y, 
                        bins_x, bins_y, range_x, range_y, xlabel, ylabel, 
                        cmap='jet', cmin=1, fontsize=10, 
                        add_hline=False, hline_y=0,
                        add_vline=False, vline_x=0):
    hist = ax.hist2d(x, y, 
                     bins=[bins_x, bins_y], 
                     range=[range_x, range_y], 
                     cmap=cmap, 
                     cmin=cmin)
    cbar = fig.colorbar(hist[3], ax=ax)
    cbar.set_label('counts', fontsize=fontsize)
    cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    cbar.ax.tick_params(labelsize=fontsize)
    
    if add_vline:
        ax.axvline(x=vline_x, color='k', linewidth=2)
    if add_hline:
        ax.axhline(y=hline_y, color='k', linewidth=2)
    
    ax.tick_params(axis='both', which='both', direction='in')
    ax.set_xlim(range_x)
    ax.set_ylim(range_y)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    return

def plot_distrib_scatter(fig, ax, x, y, z, 
                 xlabel, ylabel, clabel, 
                 xlim, ylim, 
                 cmap='Blues_r', add_grid=True):
    scatter = ax.scatter(x, y, c=z, cmap=cmap)
    cbar = fig.colorbar(scatter, ax=ax, label=clabel)
    
    if add_grid:
        ax.grid()
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return

def plot_binned_stats(fig, ax, x, y, 
                      bins_x, range_x,  
                      color_pos='red', color_neg='orange',
                      linewidth=2, capsize=5, 
                      label_pos='outward', label_neg='inward'):
    # Dividing by the sign of y-value
    mask_pos = y > 0
    mask_neg = y < 0
    
    x_pos, y_pos = x[mask_pos], y[mask_pos]
    x_neg, y_neg = x[mask_neg], y[mask_neg]
    
    # Determining bin edegs
    bin_edges = np.histogram_bin_edges(x, bins=bins_x, range=range_x)
    
    # Calculating stat values
    def _compute_bin_stats(x, y, bin_edges):
        bin_indices = np.digitize(x, bin_edges) - 1
        bin_means, bin_stds, bin_centers = [], [], []
        for i in range(len(bin_edges)-1):
            mask_bin = (bin_indices == i)
            y_bin = y[mask_bin]
            if len(y_bin) > 0:
                bin_means.append(np.mean(y_bin))
                bin_stds.append(np.std(y_bin))
                bin_centers.append((bin_edges[i] + bin_edges[i+1])/2)
        return bin_centers, bin_means, bin_stds

    centers_pos, means_pos, stds_pos = _compute_bin_stats(x_pos, y_pos, bin_edges)
    centers_neg, means_neg, stds_neg = _compute_bin_stats(x_neg, y_neg, bin_edges)
    
    # Plotting errorbar
    ax.errorbar(
        x=centers_pos, y=means_pos, yerr=stds_pos,
        color=color_pos, linewidth=linewidth, capsize=capsize,
        fmt='-o', label=label_pos
    )
    ax.errorbar(
        x=centers_neg, y=means_neg, yerr=stds_neg,
        color=color_neg, linewidth=linewidth, capsize=capsize,
        fmt='-s', label=label_neg
    )
    
    return

############################################################################################
# # Plotting quasi-radial velocity distribution (hist2d: vel vs scale)

# fig1, ax1 = plt.subplots(figsize=(8, 6))
# plot_distrib_hist2d(
#     fig=fig1,
#     ax=ax1,
#     x=vel_qr,
#     y=scale_qr,
#     bins_x=40,
#     bins_y=40,
#     range_x=[-1000, 1000],
#     range_y=[0, 800],
#     xlabel='$v_{proj}$ (km/s)',
#     ylabel='Scale (s)',
#     add_vline=True, 
#     vline_x=0
# )
# ax1.set_title('Along quasi-radial baselines')
# plt.tight_layout()

############################################################################################
# Plotting inclined and quasi-latitudinal velocity distribution (hist2d: vel vs scale)

fig2, (ax2_1, ax2_2) = plt.subplots(1, 2, figsize=(14, 6))

plot_distrib_hist2d(
    fig=fig2,
    ax=ax2_1,
    x=scale_ic,
    y=np.abs(vel_ic),
    bins_x=40,
    bins_y=20,
    range_x=[0, 800],
    range_y=[0, 1000],
    xlabel='Scale (s)',
    ylabel='|$v_{proj}$| (km/s)'
)
ax2_1.set_title('velocity distribution with scale (oblique)')

plot_distrib_hist2d(
    fig=fig2,
    ax=ax2_2,
    x=so_ic,
    y=np.abs(vel_ic),
    bins_x=20,
    bins_y=10,
    range_x=[0, 30],
    range_y=[0, 1000],
    xlabel='Solar Offset (Rs)',
    ylabel='|$v_{proj}$| (km/s)'
)
ax2_2.set_title('velocity distribution with solar-offset (oblique)')

plt.tight_layout()

############################################################################################
# # 3. Plotting quasi-radial velocity distribution (hist2d: vel vs solar-offset)
# fig3, ax3 = plt.subplots(figsize=(8, 6))
# plot_distrib_hist2d(
#     fig=fig3,
#     ax=ax3,
#     x=vel_qr,
#     y=so_qr,
#     bins_x=40,
#     bins_y=15,
#     range_x=[-1000, 1000],
#     range_y=[0, 30],
#     xlabel='$v_{proj}$ (km/s)',
#     ylabel='Solar Offset (Rs)',
#     add_vline=True,
#     vline_x=0
# )
# ax3.set_title('Along quasi-radial baselines')
# plt.tight_layout()

############################################################################################
# Plotting inclined and quasi-latitudinal velocity distribution (hist2d: vel abs vs solar-offset)
fig4, (ax4_1, ax4_2) = plt.subplots(1, 2, figsize=(14, 6))

plot_distrib_hist2d(
    fig=fig4,
    ax=ax4_1,
    x=scale_qt,
    y=np.abs(vel_qt),
    bins_x=15,
    bins_y=30,
    range_x=[0, 800],
    range_y=[0, 1000],
    xlabel='Scale (s)',
    ylabel='|$v_{proj}$| (km/s)'
)
ax4_1.set_title('velocity distribution with scale (quasi-latitudinal)')

plot_distrib_hist2d(
    fig=fig4,
    ax=ax4_2,
    x=so_qt,
    y=np.abs(vel_qt),
    bins_x=15,
    bins_y=15,
    range_x=[0, 30],
    range_y=[0, 1000],
    xlabel='Solar Offset (Rs)',
    ylabel='|$v_{proj}$| (km/s)'
)
ax4_2.set_title('velocity distribution with solar-offset (quasi-latitudinal)')

plt.tight_layout()

############################################################################################
# # Plotting quasi-radial velocity distribution (scatter: vel vs solar-offset, c=scale)
# fig5_1, ax5_1 = plt.subplots(figsize=(8, 6))
# plot_distrib_scatter(
#     fig=fig5_1,
#     ax=ax5_1,
#     x=vel_qr,
#     y=so_qr,
#     z=scale_qr,
#     xlabel='$v_{proj}$ (km/s)',
#     ylabel='Solar Offset (Rs)',
#     clabel='Scale (s)',
#     xlim=[-1000, 1000],
#     ylim=[0, 30],
#     cmap='Reds_r',
#     add_grid=True
# )
# ax5_1.axvline(x=0, color='k', linewidth=2)
# ax5_1.set_title('Along quasi-radial baselines')
# plt.tight_layout()

############################################################################################
# # Plotting inclined and quasi-latitudinal velocity distribution
# fig5_2, (ax5_21, ax5_22) = plt.subplots(1, 2, figsize=(14, 6))
# plot_distrib_scatter(
#     fig=fig5_2,
#     ax=ax5_21,
#     x=np.abs(vel_ic),
#     y=so_ic,
#     z=scale_ic,
#     xlabel='|$v_{proj}$| (km/s)',
#     ylabel='Solar Offset (Rs)',
#     clabel='Scale (s)',
#     xlim=[0, 1500],
#     ylim=[0, 30],
#     cmap='Reds_r'
# )
# ax5_21.set_title('Along oblique baselines')

# plot_distrib_scatter(
#     fig=fig5_2,
#     ax=ax5_22,
#     x=np.abs(vel_qt),
#     y=so_qt,
#     z=scale_qt,
#     xlabel='|$v_{proj}$| (km/s)',
#     ylabel='Solar Offset (Rs)',
#     clabel='Scale (s)',
#     xlim=[0, 1500],
#     ylim=[0, 30],
#     cmap='Reds_r'
# )
# ax5_22.set_title('Along quasi-latitudinal baselines')

# plt.tight_layout()

############################################################################################
# # Plotting quasi-radial velocity distribution (scatter: vel vs scale, c=solar-offset)
# fig6_1, ax6_1 = plt.subplots(figsize=(8, 6))

# plot_distrib_scatter(
#     fig=fig6_1,
#     ax=ax6_1,
#     x=vel_qr,
#     y=scale_qr,
#     z=so_qr,
#     xlabel='$v_{proj}$ (km/s)',
#     ylabel='Scale (s)',
#     clabel='Solar Offset (Rs)',
#     xlim=[-1000, 1000],
#     ylim=[0, 800],
#     cmap='Blues_r'
# )
# ax6_1.axvline(x=0, color='k', linewidth=2)
# ax6_1.set_title('Along quasi-radial baselines')
# plt.tight_layout()

############################################################################################
# # Plotting inclined and quasi-latitudinal velocity distribution
# fig6_2, (ax6_21, ax6_22) = plt.subplots(1, 2, figsize=(14, 6))

# plot_distrib_scatter(
#     fig=fig6_2,
#     ax=ax6_21,
#     x=np.abs(vel_ic),
#     y=scale_ic,
#     z=so_ic,
#     xlabel='|$v_{proj}$| (km/s)',
#     ylabel='Scale (s)',
#     clabel='Solar Offset (Rs)',
#     xlim=[0, 1500],
#     ylim=[0, 800],
#     cmap='Blues_r'
# )
# ax6_21.set_title('Along oblique baselines')

# plot_distrib_scatter(
#     fig=fig6_2,
#     ax=ax6_22,
#     x=np.abs(vel_qt),
#     y=scale_qt,
#     z=so_qt,
#     xlabel='|$v_{proj}$| (km/s)',
#     ylabel='Scale (s)',
#     clabel='Solar Offset (Rs)',
#     xlim=[0, 1500],
#     ylim=[0, 800],
#     cmap='Blues_r'
# )
# ax6_22.set_title('Along quasi-latitudinal baselines')

# plt.tight_layout()

############################################################################################
# Plotting quasi-radial velocity distribution (hist2d: scale/so vs vel)
fig7, (ax7_1, ax7_2) = plt.subplots(1, 2, figsize=(14, 6))
fontsize = 14

plot_distrib_hist2d(
    fig=fig7,
    ax=ax7_1,
    x=scale_qr,
    y=vel_qr,
    bins_x=40,
    bins_y=40,
    range_x=[0, 800],
    range_y=[-1000, 1000],
    xlabel='Scale (s)',
    ylabel='$v_{proj}$ (km s$^{-1})$',
    fontsize=fontsize, 
    add_hline=True,
    hline_y=0
)
# plot_binned_stats(fig=fig7, ax=ax7_1, x=scale_qr, y=vel_qr, 
#                   bins_x=40, range_x=[0, 800],  
#                   color_pos='red', color_neg='orange',
#                   linewidth=1, capsize=5, 
#                   label_pos='outward', label_neg='inward')
ax7_1.set_title('velocity distribution with scale', fontsize=fontsize)

plot_distrib_hist2d(
    fig=fig7,
    ax=ax7_2,
    x=so_qr,
    y=vel_qr,
    bins_x=30,
    bins_y=40,
    range_x=[0, 30],
    range_y=[-1000, 1000],
    xlabel='Solar Offset (Rs)',
    ylabel='$v_{proj}$ (km s$^{-1})$',
    fontsize=fontsize, 
    add_hline=True,
    hline_y=0,
)
# plot_binned_stats(fig=fig7, ax=ax7_2, x=so_qr, y=vel_qr, 
#                   bins_x=30, range_x=[0, 30],  
#                   color_pos='red', color_neg='orange',
#                   linewidth=1, capsize=5, 
#                   label_pos='outward', label_neg='inward')
ax7_2.set_title('velocity distrbution with solar-offset', fontsize=fontsize)

plt.tight_layout()

plt.show()

db


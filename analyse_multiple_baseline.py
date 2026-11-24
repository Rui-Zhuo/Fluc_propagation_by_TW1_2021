from project_paths import CODE_DIR, EXTERNAL_UTILS_DIR, data_directory, result_directory, result_path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from pathlib import Path
from multibaseline_fit import (correct_vel_direction, LSM_optimization, plot_loss,
                              signed_radial_angle, fit_velocity, summary_row, export_summary)

# Importing data
file_dir = result_path('').as_posix().rstrip('/') + '/'
file_name = 'multiple_baseline.xlsx'
zoom_in_or_out = 0 # 0 for zoom-in, 1 for zoom-out
Rs_km = 696000

def unit_vector(point_str, point_end):
    vector = point_end - point_str
    magnitude = np.linalg.norm(vector)
    unit_vector = vector / magnitude
    return unit_vector

def plot_projection(proj_x, proj_y, size, colors):
    plt.scatter(proj_x[0] / Rs_km, proj_y[0] / Rs_km, s=size, c=colors[0], label='P-point')
    plt.scatter(proj_x[1] / Rs_km, proj_y[1] / Rs_km, s=size, c=colors[1])
    plt.scatter(proj_x[2] / Rs_km, proj_y[2] / Rs_km, s=size, c=colors[2])

def plot_radial_direct(con_point_str, con_point_end, color, scale, y_offset=-0.0004, width=0.005):
    qv = plt.quiver(*con_point_str, *con_point_end, angles='xy', scale_units='xy', \
        scale=scale, color=color, width=width)
    plt.text(con_point_str[0], con_point_str[1]+y_offset, 'Radial', color=color)

def plot_baselines(proj_x, proj_y, linewidth=3):
    plt.plot([proj_x[0] / Rs_km, proj_x[1] / Rs_km], [proj_y[0]/ Rs_km, proj_y[1] / Rs_km], 'k--', linewidth=linewidth, label='Baseline')
    plt.plot([proj_x[0] / Rs_km, proj_x[2] / Rs_km], [proj_y[0]/ Rs_km, proj_y[2] / Rs_km], 'k--', linewidth=linewidth)
    plt.plot([proj_x[1] / Rs_km, proj_x[2] / Rs_km], [proj_y[1]/ Rs_km, proj_y[2] / Rs_km], 'k--', linewidth=linewidth)

def plot_vel(point_str, vel, color, scale, width=0.005):
    plt.quiver(*point_str, *vel, angles='xy', scale_units='xy', scale=scale, color=color, width=width)

def plot_quiver_scale(quiver_pos, quiver_direct, color, scale, width=0.005):
    plt.quiver(*quiver_pos, *quiver_direct, angles='xy', scale_units='xy', \
        scale=scale, color=color, width=width)
    plt.text(quiver_pos[0], quiver_pos[1]+0.0002, 'velocity')
    plt.text(quiver_pos[0], quiver_pos[1]-0.0005, '100 km/s')

def select_start_point(proj_x):
    str_idx = np.argmax(np.abs(proj_x))
    return str_idx

def select_connect_point(proj_x):
    con_idx = np.argmin(np.abs(proj_x))
    return con_idx

def get_theta_kr(vel, er):
    return signed_radial_angle(vel, er)

def mark_vel(pos, vel_mod, theta_kr, y_gap, vel_std, theta_std):
    # +/- here denotes conditional local standard errors, not 68% profile bounds.
    plt.text(pos[0], pos[1], rf'$v_p={vel_mod:.1f}\pm{vel_std:.1f}$ km/s', fontsize=15)
    plt.text(pos[0], pos[1]+y_gap,
             rf'$\theta_{{kr}}={theta_kr:+.2f}\pm{theta_std:.2f}^\circ$', fontsize=15)
    return

def get_wavefront(point_str, vel_opt, scale, line_length=2):
    point_end = point_str + vel_opt / scale
    perp_vector = np.array([-vel_opt[1], vel_opt[0]])
    perp_vector = perp_vector / np.linalg.norm(perp_vector) * line_length

    wavefront_start = point_end - perp_vector / 2
    wavefront_end = point_end + perp_vector / 2
    return point_end, wavefront_start, wavefront_end

def plot_wavefront(wavefront_start, wavefront_end, linewidth):
    plt.plot([wavefront_start[0], wavefront_end[0]], [wavefront_start[1], wavefront_end[1]], \
        'r--', linewidth=linewidth, label='Wavefront')
    return wavefront_start, wavefront_end

def plot_wavefront_propagate(wavefront_start, wavefront_end, wavelength, num_line_per, num_line, \
                             line_gap=0.05, prop_direct='right', color_map='Blues', color_scale=0.75):
    x1, y1 = wavefront_start[0], wavefront_start[1]
    x2, y2 = wavefront_end[0], wavefront_end[1]
    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1

    cmap = plt.get_cmap(color_map)

    wavelength_Rs = wavelength / Rs_km
    dy = wavelength_Rs * np.sqrt(k**2+1) / num_line_per
    x_arr = np.linspace(min(x1, x2), max(x1, x2), 100)
    for i in range(num_line):
        if prop_direct == 'right':
            b1 = b + dy * i
            b2 = b + dy * (i + 1)
        elif prop_direct == 'left':
            b1 = b - dy * i
            b2 = b - dy * (i + 1)
        y1_arr = k * x_arr + b1
        y2_arr = k * x_arr + b2

        line_deci = (i % num_line_per) / num_line_per
        if line_deci <= 0.5:
            color_deci = 0.5 + (1 - line_deci*2 - 0.5) * color_scale
        elif line_deci > 0.5:
            color_deci = 0.5 + (line_deci*2 - 1 - 0.5) * color_scale
        plt.fill_between(x_arr, y1_arr, y2_arr, color=cmap(color_deci))

def main(show=True, zoom=0):
    global zoom_in_or_out
    zoom_in_or_out = zoom
    output_dir = Path(result_directory('multiple_baseline_rmse')) / ('zoom_out' if zoom else 'zoom_in')
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    data = pd.ExcelFile(file_dir + file_name)
    df_coh = pd.read_excel(data, sheet_name='coherency', header=0)
    # print('df_coh columns: ', df_coh.columns)
    df_info = pd.read_excel(data, sheet_name='station_info', header=0)
    # print('df_info columns: ', df_info.columns)

    vel = np.array(df_coh['vel'])
    scale_min = np.array(df_coh['scale_min'])
    scale_max = np.array(df_coh['scale_max'])
    proj_x = np.array(df_info['proj_x'])
    proj_y = np.array(df_info['proj_y'])

    for i_case in range(5):
        vel_case = vel[i_case*4:i_case*4+3]
        scale_min_case = scale_min[i_case*4:i_case*4+3]
        scale_max_case = scale_max[i_case*4:i_case*4+3]
        proj_x_case = proj_x[i_case*4:i_case*4+3]
        proj_y_case = proj_y[i_case*4:i_case*4+3]

        e01 = unit_vector(np.array([proj_x_case[0],proj_y_case[0]]), np.array([proj_x_case[1],proj_y_case[1]]))
        e02 = unit_vector(np.array([proj_x_case[0],proj_y_case[0]]), np.array([proj_x_case[2],proj_y_case[2]]))
        e12 = unit_vector(np.array([proj_x_case[1],proj_y_case[1]]), np.array([proj_x_case[2],proj_y_case[2]]))

        mpl.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.linewidth'] = 2.0
        if i_case == 0 or i_case == 1 or i_case == 3 or i_case == 4:
            plt.rcParams['font.size'] = 20
            prop_direct = 'left'
        elif i_case== 2:
            plt.rcParams['font.size'] = 20
            prop_direct = 'right'

        # Consistent to the positive direction of unit vectors!
        vel_case = -vel_case
        vel01 = vel_case[0] * e01
        vel02 = vel_case[1] * e02
        vel12 = vel_case[2] * e12

        fit = fit_velocity([e01, e02, e12], vel_case)
        S_loss, vp_range, theta_range = fit['loss'], fit['speeds'], fit['angles']
        min_loss, min_vp, min_theta_deg = plot_loss(S_loss, vp_range, theta_range, i_case)

        plt.savefig(output_dir / f'case_{i_case+1:02d}_loss.png', dpi=180, bbox_inches='tight')
        plt.close()

        vel_opt_x, vel_opt_y = min_vp * np.cos(np.radians(min_theta_deg)), min_vp * np.sin(np.radians(min_theta_deg))
        vel_opt = np.array([vel_opt_x, vel_opt_y])
        vel_opt_mod = np.sqrt(vel_opt_x**2 + vel_opt_y**2)

        # Calculating the wavelength
        scale_mean = (scale_min_case[0] + scale_max_case[0]) / 2
        wavelength = vel_opt_mod * scale_mean

        plt.figure(figsize=(9,9))

        # Plotting the baselines
        plot_baselines(proj_x_case, proj_y_case)

        # Selecting the start point for plotting the velocities
        str_idx = select_start_point(proj_x_case)
        point_str = np.array([proj_x_case[str_idx] / Rs_km, proj_y_case[str_idx] / Rs_km])

        # Determining the wavefront location
        scale = 100000
        point_end, wavefront_start, wavefront_end = get_wavefront(point_str, vel_opt, scale)

        # Plotting the wavefront and its propagtation
        if zoom_in_or_out == 0:
            plot_wavefront_propagate(wavefront_start, wavefront_end, wavelength, num_line_per=30, num_line=60, \
                line_gap=0.03, prop_direct=prop_direct, color_map='Blues', color_scale=0.8) # colormap: YlGn, Blues, Greys
        elif zoom_in_or_out == 1:
            plot_wavefront_propagate(wavefront_start, wavefront_end, wavelength, num_line_per=8, num_line=16, \
                line_gap=0.03, prop_direct=prop_direct, color_map='Blues', color_scale=0.8)
        plot_wavefront(wavefront_start, wavefront_end, linewidth=5)

        # Plotting the station projections
        plot_projection(proj_x_case, proj_y_case, size=80, colors='kkk')

        # Calculating the radial direction
        con_idx = select_connect_point(proj_x_case)
        con_point_str = np.array([proj_x_case[con_idx] / Rs_km, proj_y_case[con_idx] / Rs_km])
        con_point_abs = np.sqrt(con_point_str[0]**2 + con_point_str[1]**2)
        if i_case == 0 or i_case == 1 or i_case == 2:
            con_point_end = con_point_str / con_point_abs * 250
            y_offset = -0.0004
        elif i_case == 3 or i_case == 4:
            con_point_end = con_point_str / con_point_abs * 150
            y_offset = 0.0004

        # Calculating the theta_kr
        er = unit_vector(np.array([0,0]), con_point_str)
        theta_kr = get_theta_kr(vel_opt, er)
        rows.append(summary_row(fit, i_case+1, theta_kr,
                                date=int(df_coh.iloc[i_case*4]['date']),
                                time_beg=int(df_coh.iloc[i_case*4]['time_beg']),
                                time_end=int(df_coh.iloc[i_case*4]['time_end']),
                                radial_reference_station=str(df_info.iloc[i_case*4+con_idx]['stations'])))
        vel_opt_mod_mark = np.round(vel_opt_mod, 2)
        theta_kr_mark = np.round(theta_kr, 2)
        if i_case == 0 or i_case == 1:
            mark_pos = np.array([point_end[0]-0.002, point_end[1]+0.001])
            y_gap = -0.0009
        elif i_case == 3 or i_case == 4:
            mark_pos = np.array([point_end[0], point_end[1]+0.0008])
            y_gap = -0.0007
        elif i_case == 2:
            mark_pos = np.array([point_end[0]+0.0005, point_end[1]])
            y_gap = -0.0006

        # Plotting the measured and resulted velocities
        if zoom_in_or_out == 0:
            plot_radial_direct(con_point_str, con_point_end, 'w', scale=scale, y_offset=y_offset, width=0.0075)
            plot_vel(point_str, vel01, 'orange', scale)
            plot_vel(point_str, vel02, 'orange', scale)
            plot_vel(point_str, vel12, 'orange', scale)
            plot_vel(point_str, vel_opt, 'red', scale, width=0.005)
            mark_vel(mark_pos, vel_opt_mod_mark, theta_kr_mark, y_gap,
                     fit['speed_std'], fit['theta_std_deg'])
        elif zoom_in_or_out == 1:
            if i_case == 0 or i_case == 1 or i_case == 3:
                plot_vel(point_str, vel_opt*60, 'red', scale, width=0.005*4)
            elif i_case == 2:
                plot_vel(point_str, vel_opt*30, 'red', scale, width=0.005*8)
            elif i_case == 4:
                plot_vel(point_str, vel_opt*20, 'red', scale, width=0.005*3)

        # Plotting the measuring scale of quivers
        quiver_pos = np.array([proj_x_case[str_idx] / Rs_km + 0.003, proj_y_case[str_idx] / Rs_km + 0.003])
        quiver_direct = np.array([100,0])
        if i_case == 2:
            if zoom_in_or_out == 0:
                plot_quiver_scale(quiver_pos, quiver_direct, 'k', scale)
                plt.legend()

        plt.gca().set_aspect(1)
        if zoom_in_or_out == 0:
            plt.xlim(proj_x_case[str_idx] / Rs_km - 0.01, proj_x_case[str_idx] / Rs_km + 0.01)
            plt.ylim(proj_y_case[str_idx] / Rs_km - 0.01, proj_y_case[str_idx] / Rs_km + 0.01)
            x_major_locator = MultipleLocator(0.003)
            y_major_locator = MultipleLocator(0.003)
            plt.gca().xaxis.set_major_locator(x_major_locator)
            plt.gca().yaxis.set_major_locator(y_major_locator)
            plt.xlabel('X (Rs)')
            plt.ylabel('Y (Rs)')
            plt.title('Plane of Sky @ Case' + str(i_case+1))
        elif zoom_in_or_out == 1:
            plt.xticks([])
            plt.yticks([])

        plt.tight_layout()
        plt.savefig(output_dir / f'case_{i_case+1:02d}_sky_plane.png', dpi=180, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    frame = export_summary(rows, output_dir)
    print(frame[['case','vp_km_s','theta_p_deg','theta_kr_deg','rmse_km_s','vp_std_local_km_s','theta_std_local_deg']].to_string(index=False))
    print('Outputs:', output_dir)
    return frame


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Original multibaseline products with RMSE and conditional uncertainty.')
    parser.add_argument('--no-show', action='store_true', help='Save figures without opening plot windows.')
    parser.add_argument('--zoom-out', action='store_true', help='Use the original zoomed-out display mode.')
    args = parser.parse_args()
    main(show=not args.no_show, zoom=int(args.zoom_out))

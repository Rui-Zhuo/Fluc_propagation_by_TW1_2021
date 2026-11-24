from project_paths import CODE_DIR, EXTERNAL_UTILS_DIR, data_directory, result_directory, result_path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from multibaseline_fit import (correct_vel_direction, LSM_optimization, plot_loss,
                              signed_radial_angle, fit_velocity, summary_row, export_summary)

file_dir = result_path('base_line_spikes_multiple/').as_posix().rstrip('/') + '/'
file_name = 'lagvelocity_baseline_neaten.txt'
Rs_km = 696000

######################################
def unit_vector(point_str, point_end):
    vector = point_end - point_str
    magnitude = np.linalg.norm(vector)
    unit_vector = vector / magnitude
    return unit_vector

def plot_projection(proj_x, proj_y, colors):
    plt.plot([0, proj_x[0] / Rs_km], [0, proj_y[0] / Rs_km], linewidth=2, c=colors[0])
    plt.plot([0, proj_x[1] / Rs_km], [0, proj_y[1] / Rs_km], linewidth=2, c=colors[1])
    plt.plot([0, proj_x[2] / Rs_km], [0, proj_y[2] / Rs_km], linewidth=2, c=colors[2])
    plt.plot([proj_x[0] / Rs_km, proj_x[1] / Rs_km], [proj_y[0]/ Rs_km, proj_y[1] / Rs_km], 'k--', linewidth=2)
    plt.plot([proj_x[0] / Rs_km, proj_x[2] / Rs_km], [proj_y[0]/ Rs_km, proj_y[2] / Rs_km], 'k--', linewidth=2)
    plt.plot([proj_x[1] / Rs_km, proj_x[2] / Rs_km], [proj_y[1]/ Rs_km, proj_y[2] / Rs_km], 'k--', linewidth=2)

def plot_vel(point_str, vel, color, scale=100000):
    plt.quiver(*point_str, *vel, angles='xy', scale_units='xy', scale=scale, color=color, width=0.005)

def select_start_point(proj_x):
    str_idx = np.argmax(np.abs(proj_x))
    return str_idx

def plot_perpendicular_line(point_str, vel_opt, scale, line_length=2):
    endpoint = point_str + vel_opt / scale
    perp_vector = np.array([-vel_opt[1], vel_opt[0]])
    perp_vector = perp_vector / np.linalg.norm(perp_vector) * line_length

    line_start = endpoint - perp_vector / 2
    line_end = endpoint + perp_vector / 2

    plt.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], 'm--')
######################################

def main(show=True):
    output_dir = Path(result_directory('multiple_baseline_rmse/spikes'))
    rows = []
    df = pd.read_csv(file_dir + file_name, sep=r'\s+')
    variables = list(df.columns)
    print(variables)

    year, month, day, second = df['year'], df['month'], df['day'], df['second']
    stat1, stat2 = df['stat1'], df['stat2']
    coef = df['coef']
    lag, lag_err = df['lag'], df['lag_err']
    vel, vel_err = df['vel'], df['vel_err']
    lat_proj, r_proj = df['lat_proj'], df['r_proj']
    bl_t, bl_r = df['bl_t'], df['bl_r']

    for i_case in range(1):
        vel_case = np.asarray(vel.iloc[i_case*3:i_case*3+3], dtype=float)
        bl_r1, bl_t1 = bl_r[i_case*3], bl_t[i_case*3]
        bl_r2, bl_t2 = bl_r[i_case*3+1], bl_t[i_case*3+1]

        proj_x_case = [0, bl_r1, bl_r2]
        proj_y_case = [0, bl_t1, bl_t2]
        print(proj_x_case)
        print(proj_y_case)

        e01 = unit_vector(np.array([proj_x_case[0],proj_y_case[0]]), np.array([proj_x_case[1],proj_y_case[1]]))
        e02 = unit_vector(np.array([proj_x_case[0],proj_y_case[0]]), np.array([proj_x_case[2],proj_y_case[2]]))
        e21 = unit_vector(np.array([proj_x_case[2],proj_y_case[2]]), np.array([proj_x_case[1],proj_y_case[1]]))
        print(e01, e02, e21)

        vel01 = vel_case[0] * e01
        vel02 = vel_case[1] * e02
        vel21 = vel_case[2] * e21

        fit = fit_velocity([e01, e02, e21], vel_case)
        S_loss, vp_range, theta_range = fit['loss'], fit['speeds'], fit['angles']
        min_loss, min_vp, min_theta_deg = plot_loss(S_loss, vp_range, theta_range, i_case)
        vel_opt_x, vel_opt_y = min_vp * np.cos(np.radians(min_theta_deg)), min_vp * np.sin(np.radians(min_theta_deg))
        plt.savefig(output_dir / f'case_{i_case+1:02d}_loss.png', dpi=180, bbox_inches='tight')
        plt.close()
        vel_opt = np.array([vel_opt_x, vel_opt_y])
        theta_kr = signed_radial_angle(vel_opt, np.array([1., 0.]))
        rows.append(summary_row(fit, i_case+1, theta_kr, radial_reference_station='local +e_r axis'))

        plt.figure(figsize=(6,6))
        plt.title(f'$v_p$={min_vp:.2f} km/s, $\\theta_{{kr}}$={theta_kr:+.2f} deg')

        plot_projection(proj_x_case, proj_y_case, 'www')

        str_idx = 0
        point_str = np.array([proj_x_case[str_idx] / Rs_km, proj_y_case[str_idx] / Rs_km])

        scale = 100000
        plot_vel(point_str, vel01, 'orange', scale)
        plot_vel(point_str, vel02, 'orange', scale)
        plot_vel(point_str, vel21, 'orange', scale)
        plot_vel(point_str, vel_opt, 'purple', scale)
        plot_perpendicular_line(point_str, vel_opt, scale)

        plt.xlim(proj_x_case[str_idx] / Rs_km - 0.01, proj_x_case[str_idx] / Rs_km + 0.01)
        plt.ylim(proj_y_case[str_idx] / Rs_km - 0.01, proj_y_case[str_idx] / Rs_km + 0.01)

        plt.gca().set_aspect(1)
        plt.xlabel('e_r (Rs)')
        plt.ylabel('e_t (Rs)')
        plt.suptitle('Plane of Sky @ case' + str(i_case+1))

        plt.tight_layout()
        plt.savefig(output_dir / f'case_{i_case+1:02d}_sky_plane.png', dpi=180, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    frame = export_summary(rows, output_dir)
    print(frame[['case','vp_km_s','theta_p_deg','theta_kr_deg','rmse_km_s']].to_string(index=False))
    return frame


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    main(show=not args.no_show)

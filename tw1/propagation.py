"""Unweighted velocity RMSE fit, with fixed measured signs and conditional errors.

The model is unchanged: u_i = v_p / cos(phi_i - theta_p). Negative measured
velocities are represented by reversing the corresponding baseline. Nonpositive
cosines are outside that sign-consistent domain, not clipped predictions.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

COS_EPS = 1e-10
SPEED_BOUNDS = (10.0, 500.0)


def correct_vel_direction(phi, vel):
    phi = np.asarray(phi, dtype=float).copy()
    vel = np.asarray(vel, dtype=float).copy()
    phi[vel < 0] += np.pi
    return phi % (2 * np.pi), np.abs(vel)


def signed_radial_angle(vel, er):
    """Degrees: positive counterclockwise from outward radial in plotted X/Y."""
    vel, er = np.asarray(vel), np.asarray(er)
    if np.linalg.norm(vel) == 0 or np.linalg.norm(er) == 0:
        raise ValueError('A zero vector has no direction.')
    return float(np.degrees(np.arctan2(er[0]*vel[1]-er[1]*vel[0], np.dot(er, vel))))


def _golden_minimum(fun, left, right):
    """Bounded scalar minimization; avoids an additional SciPy dependency."""
    ratio = (np.sqrt(5)-1)/2
    x1, x2 = right-ratio*(right-left), left+ratio*(right-left)
    f1, f2 = fun(x1), fun(x2)
    for _ in range(90):
        if right-left < 1e-12:
            break
        if f1 < f2:
            right, x2, f2 = x2, x1, f1
            x1 = right-ratio*(right-left)
            f1 = fun(x1)
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left+ratio*(right-left)
            f2 = fun(x2)
    return min(((x1, f1), (x2, f2)), key=lambda pair: pair[1])


def fit_velocity(directions, velocities):
    directions = np.asarray(directions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    if directions.shape != (3, 2) or velocities.shape != (3,):
        raise ValueError('Expected three 2-D baseline unit vectors and three velocities.')
    if not np.all(np.isfinite(directions)) or not np.all(np.isfinite(velocities)):
        raise ValueError('Baseline directions and velocities must be finite.')
    if np.any(velocities == 0) or not np.allclose(np.linalg.norm(directions, axis=1), 1):
        raise ValueError('Require nonzero velocities and unit baseline directions.')
    phi, observed = correct_vel_direction(np.arctan2(directions[:, 1], directions[:, 0]), velocities)
    lower, upper = SPEED_BOUNDS

    def profile(theta):
        cosine = np.cos(theta-phi)
        if np.any(cosine <= COS_EPS):
            return np.inf, np.nan
        sec = 1/cosine
        # At fixed angle the original unweighted loss is quadratic in speed.
        speed = float(np.clip(np.dot(observed, sec)/np.dot(sec, sec), lower, upper))
        return float(np.sum((observed-speed*sec)**2)), speed

    # Explicit sign-domain boundaries prevent missing a narrow feasible interval.
    half_width = np.arccos(COS_EPS)
    boundaries = np.unique(np.r_[0., 2*np.pi, (phi-half_width) % (2*np.pi), (phi+half_width) % (2*np.pi)])
    candidates = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        if not np.isfinite(profile((left+right)/2)[0]):
            continue
        angles = np.linspace(left+1e-12, right-1e-12, max(5, int(np.ceil((right-left)/np.radians(.25)))+1))
        values = np.array([profile(angle)[0] for angle in angles])
        for i in range(len(angles)):
            if values[i] <= values[max(0, i-1)] and values[i] <= values[min(len(angles)-1, i+1)]:
                candidates.append(_golden_minimum(lambda t: profile(t)[0], angles[max(0,i-1)], angles[min(len(angles)-1,i+1)]))
    if not candidates:
        raise ValueError('No propagation direction agrees with all three measured signs.')
    theta, sse = min(candidates, key=lambda pair: pair[1])
    speed = profile(theta)[1]
    cosine = np.cos(theta-phi)
    prediction = speed/cosine
    residual = observed-prediction
    # n=3, p=2: only one degree of freedom. These are conditional local errors,
    # not calibrated measurement errors or systematic/model uncertainties.
    dof = len(observed)-2
    sigma2 = sse/dof
    jacobian = np.column_stack((1/cosine, speed*np.sin(theta-phi)/cosine**2))
    _, singular_values, vt = np.linalg.svd(jacobian, full_matrices=False)
    rank_ok = singular_values[-1] > singular_values[0]*1e-12
    covariance = (vt.T @ np.diag(1/singular_values**2) @ vt)*sigma2 if rank_ok else np.full((2,2),np.nan)
    std = np.sqrt(np.diag(covariance))

    speeds = np.unique(np.r_[np.linspace(lower, upper, 981), speed])
    angles = np.unique(np.r_[np.linspace(0, 2*np.pi, 1440, endpoint=False), theta])
    cosines = np.cos(angles[:,None]-phi)
    valid = np.all(cosines > COS_EPS, axis=1)
    loss = np.full((len(speeds),len(angles)), np.inf)
    # Evaluate actual squared residuals to avoid cancellation near a perfect fit.
    for i in np.flatnonzero(valid):
        loss[:,i] = np.sqrt(np.mean((observed[None,:]-speeds[:,None]/cosines[i])**2, axis=1))
    result = dict(speed=speed, theta=theta, rmse=np.sqrt(sse/len(observed)), sse=sse,
                  sigma=np.sqrt(sigma2), dof=dof, covariance=covariance,
                  speed_std=std[0], theta_std_deg=np.degrees(std[1]),
                  corr=covariance[0,1]/(std[0]*std[1]) if np.all(std > 0) else np.nan,
                  condition=singular_values[0]/singular_values[-1], rank_ok=rank_ok,
                  speed_boundary=bool(np.isclose(speed,lower) or np.isclose(speed,upper)),
                  prediction=prediction*np.sign(velocities), residual=residual*np.sign(velocities),
                  speeds=speeds, angles=angles, loss=loss)
    # Approximate profile-F intervals account for estimating the common unknown
    # variance from one residual dof. F(1,1) quantile = tan(pi*level/2)^2.
    # These intervals assume independent Gaussian equal-variance velocity errors;
    # the nonlinear model makes coverage approximate, especially near boundaries.
    speed_profile = np.min(loss, axis=1)**2*len(observed)
    angle_profile = np.min(loss, axis=0)**2*len(observed)
    delta = np.arctan2(np.sin(angles-theta),np.cos(angles-theta))
    for level, label in ((.68,'68'),(.95,'95')):
        threshold = sse*(1+np.tan(np.pi*level/2)**2/dof)
        tolerance = max(1e-18,sse*1e-10)
        selected_v = speed_profile <= threshold+tolerance
        selected_t = angle_profile <= threshold+tolerance
        result['speed_lo'+label] = float(speeds[selected_v].min())
        result['speed_hi'+label] = float(speeds[selected_v].max())
        result['theta_minus'+label] = float(-np.degrees(delta[selected_t].min()))
        result['theta_plus'+label] = float(np.degrees(delta[selected_t].max()))
        result['interval_boundary'+label] = bool(selected_v[0] or selected_v[-1])
    return result


def LSM_optimization(e01, e02, e12, vel):
    fit = fit_velocity([e01,e02,e12],vel)
    return fit['loss'],fit['speeds'],fit['angles']


def plot_loss(loss, speeds, angles, i_case):
    """Keep the original loss-map product; color now has actual km/s units."""
    i,j = np.unravel_index(np.argmin(loss),loss.shape)
    minimum, speed, theta = loss[i,j],speeds[i],np.degrees(angles[j])
    fig,ax = plt.subplots(figsize=(10,6))
    positive = loss[np.isfinite(loss) & (loss > 0)]
    floor = max(float(positive.min()) if positive.size else 1e-12,1e-12)
    shown = np.ma.masked_invalid(np.where(np.isfinite(loss),np.maximum(loss,floor),np.nan))
    mesh = ax.pcolormesh(speeds,np.degrees(angles),shown.T,shading='auto',cmap='jet',norm=LogNorm(vmin=floor,vmax=max(floor*10,float(positive.max()) if positive.size else 1)))
    fig.colorbar(mesh,ax=ax,label='RMSE (km/s)')
    ax.plot(speed,theta,'kx',ms=8,label=f'Min RMSE={minimum:.2f} km/s\n$v_p$={speed:.2f} km/s, $\\theta_p$={theta:.2f} deg')
    # For p=2, dof=1, F(2,1) gives joint RMSE / min = 1/(1-level).
    if minimum > 1e-10:
        contours = ax.contour(speeds,np.degrees(angles),shown.T,levels=[minimum/(1-.68),minimum/(1-.95)],colors=['white','black'],linewidths=1)
        ax.clabel(contours,fmt={minimum/(1-.68):'joint 68%',minimum/(1-.95):'joint 95%'},fontsize=9)
    ax.set(xlabel='Phase speed $v_p$ (km/s)',ylabel='Propagation angle $\\theta_p$ (deg)',title=f'Velocity RMSE @ Case {i_case+1}')
    ax.legend(fontsize=11,loc='upper right')
    fig.tight_layout()
    return float(minimum),float(speed),float(theta)


def summary_row(fit,case,theta_kr,**metadata):
    row = dict(case=case,**metadata,vp_km_s=fit['speed'],theta_p_deg=np.degrees(fit['theta']),
               theta_kr_deg=theta_kr,radial_side='CCW' if theta_kr>0 else ('CW' if theta_kr<0 else 'radial'),
               rmse_km_s=fit['rmse'],vp_std_local_km_s=fit['speed_std'],
               theta_std_local_deg=fit['theta_std_deg'],speed_angle_correlation=fit['corr'],
               residual_sigma_km_s=fit['sigma'],residual_dof=fit['dof'],
               optimum_at_speed_boundary=fit['speed_boundary'],jacobian_full_rank=fit['rank_ok'],
               jacobian_condition=fit['condition'],
               uncertainty_method='residual iid equal-variance Gaussian; local std + approximate profile-F',
               speed_interval_grid_step_km_s=.5,angle_interval_grid_step_deg=.25)
    for label in ('68','95'):
        row.update({f'vp_lo{label}_km_s':fit['speed_lo'+label],f'vp_hi{label}_km_s':fit['speed_hi'+label],
                    f'theta_minus{label}_deg':fit['theta_minus'+label],f'theta_plus{label}_deg':fit['theta_plus'+label],
                    f'interval{label}_touches_speed_boundary':fit['interval_boundary'+label]})
    for i in range(3):
        row[f'predicted_vel_{i+1}_km_s']=fit['prediction'][i]
        row[f'residual_vel_{i+1}_km_s']=fit['residual'][i]
    return row


def export_summary(rows,output_dir,stem='summary'):
    output_dir.mkdir(parents=True,exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir/(stem+'.csv'),index=False,encoding='utf-8-sig')
    cells=[]
    for row in rows:
        cells.append([str(row['case']),f"{row['vp_km_s']:.2f}",f"{row['theta_p_deg']:.2f}",f"{row['theta_kr_deg']:+.2f}",f"{row['rmse_km_s']:.2f}",
                      f"[{row['vp_lo68_km_s']:.1f}, {row['vp_hi68_km_s']:.1f}]"+('*' if row['interval68_touches_speed_boundary'] else ''),
                      f"-{row['theta_minus68_deg']:.2f} / +{row['theta_plus68_deg']:.2f}"])
    fig,ax=plt.subplots(figsize=(14,max(3,1.8+.5*len(rows))))
    ax.axis('off')
    table=ax.table(cellText=cells,colLabels=['Case','Speed\n(km/s)','Direction\n(deg)','Radial angle\n(deg, signed)','RMSE\n(km/s)','Speed 68% interval\n(km/s)','Angle 68% offsets\n(deg)'],loc='center',cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1,2.3)
    for (r,c),cell in table.get_celld().items():
        cell.set_facecolor('#dce8f2' if r==0 else ('#f3f6f8' if r%2 else 'white'))
        cell.set_edgecolor('white')
    ax.set_title('Multiple-baseline propagation fit',fontsize=17,pad=15)
    fig.text(.5,.04,'Conditional approximate profile-F intervals; 3 baselines, 2 parameters, 1 residual dof.\nAngles: positive CCW from outward radial; intervals wrap with direction. * Search-boundary limited.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.13,1,1])
    fig.savefig(output_dir/(stem+'.png'),dpi=200,bbox_inches='tight')
    plt.close(fig)
    return frame

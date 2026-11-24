# Multiple-baseline RMSE update

## Model and fitting

The original forward model and unweighted baseline residuals are retained:

    predicted_i = vp / cos(phi_i - theta_p)
    RMSE = sqrt(sum((observed_i - predicted_i)**2) / n)

The previous implementation actually stored SSE, rather than its square root.
SSE, its square root, and RMSE have the same minimizer for a fixed n.
The updated color bar reports actual RMSE in km/s, rather than SSE/min(SSE).

Measured negative velocities are represented by flipping the associated baseline
angle by pi and taking the absolute velocity, using copies of the input arrays.
After that correction, only directions with cos(theta_p - phi_i) > 1e-10 for
every baseline are admissible. Other directions have infinite loss and are masked
in plots. The threshold excludes the singularity; it is never substituted into
the prediction. This retains the original intended fixed-sign constraint. It does
not model uncertainty in a measured sign.

The original 10--500 km/s search bounds are retained. For each trial angle the
best speed is obtained from the quadratic SSE, and all feasible angular intervals
are searched and refined. This changes numerical resolution, not the loss or
physical model. The loss display and profile intervals use 0.5 km/s and 0.25 degree
grids, augmented with the refined optimum. Periodic angular endpoints are not
duplicated. The original five main cases and first spike triplet remain selected.

## Signed angle

    theta_kr = degrees(atan2(er_x * v_y - er_y * v_x, dot(er, v)))

This is the angle from the outward radial direction to propagation in the plotted
X/Y axes, in [-180, 180] degrees. Positive means counterclockwise; negative means
clockwise. For the main cases the radial reference is selected using the same
station as before, and recorded in the CSV. For the spike plot the reference is
the local +e_r axis. This convention does not by itself imply heliographic north
or south. The magnitude preserves the old arccos angle.

## Conditional uncertainty

The main workbook has no calibrated velocity standard deviations. Its lag_min
and lag_max values are NOT treated as measurement confidence limits. The
following estimates assume independent Gaussian, equal-variance velocity errors,
fixed baseline geometry and fixed measured signs. They exclude systematic errors,
shared-station correlations, uncertainty in the radial reference and wavefront
model discrepancy. The spike variant uses the same residual-based method for
comparability; its supplied vel_err is not used to change the loss or calibrate
these intervals.

There are n=3 observations and p=2 fitted parameters, so nu=n-p=1:

    sigma_hat**2 = SSE_min / nu = n * RMSE_min**2 / nu
    J_i = [sec(theta-phi_i), vp * sin(theta-phi_i) / cos(theta-phi_i)**2]
    Cov(vp, theta) ~= sigma_hat**2 * inverse(J.T @ J)

Theta is in radians within the covariance calculation, then converted to degrees.
The square roots of the covariance diagonal are conditional local standard
errors. With a fixed radial reference, theta_kr has the same local angular
standard error as theta_p. Plot annotations use these local +/- errors; they
are not 68% profile intervals. Boundary optima and rank deficiencies invalidate
ordinary symmetric-error interpretations and are identified in the CSV.

To show the effect of estimating variance from only one residual degree of
freedom, the CSV also reports approximate nonlinear profile-F intervals. For
one profiled parameter at confidence c, use

    SSE_profile <= SSE_min * (1 + F_quantile(c; 1, nu) / nu)

Here F_quantile(c;1,1) = tan(pi*c/2)**2. The 68% and 95% profile sets are evaluated
on the display grid; bounds are approximate at the listed grid resolution.
Angular bounds are offsets from the optimum using circular differences; the same
offsets apply to theta_kr. Do not interpret a wrap across 0/360 or -180/180 as a
large change in direction. These are conditional approximate confidence sets, not
posterior probabilities. Standard F coverage is exact for linear Gaussian models;
its use for this nonlinear model is approximate, especially near boundaries.

The loss maps also show joint 68% and 95% approximate F contours for both fitted
parameters. With p=2, nu=1 their RMSE thresholds are RMSE_min/(1-c). Joint contour
projections differ from the one-parameter profile intervals in the summary.

Search-boundary-limited speed intervals are flagged; a truncated interval does not
establish a physical upper/lower bound. One residual degree of freedom makes all
residual-based uncertainty estimates fragile. Reliable measurement covariance or
joint signal resampling would be needed for a stronger uncertainty statement.

## Run and outputs

From CODE, run:

    python analyse_multiple_baseline.py --no-show
    python analyse_multiple_baseline.py --no-show --zoom-out
    python analyse_multiple_baseline_for_spikes.py --no-show
    python -m unittest test_multibaseline_fit -v

Omit --no-show for interactive plots. The first two commands generate the
original sky-plane/wavefront plots and loss maps for all five cases in
RESULT/multiple_baseline_rmse/zoom_in and zoom_out. Each directory also has
summary.csv (full numeric results, standard errors, 68/95% profiles, predictions,
residuals and diagnostics) and summary.png (compact table). Spike outputs go to
RESULT/multiple_baseline_rmse/spikes.

Original station selection, velocity sign inversion, scale-based wavelength,
wavefront coloring and plotting extents are retained. The axis locator typo and
trailing undefined `db` are fixed. Plot fonts are made available locally.

## Path migration

Python source files and notebook code cells use project_paths.py. CODE and RESULT
follow the checkout location; observation data remain under Tianwen-1. The older
m1930x_up_new, m1a07x_up, m1a15y_copy, m1a12x_up_new and m1a04x_PLL variants are
explicitly referenced under Tianwen-1/outdated, where their original files exist.
The external python_utils_JSHEPT helper remains in E:/Research/Program/[else].
Coordinate tables, SPICE kernels and textures resolve relative to the local
CODE/Sun_Mars_Earth directory. The MATLAB cases.xlsx input resolves under RESULT.
Observation filenames and selected time intervals are preserved.

The notebook's obsolete frequency_analyse_utils import is redirected to the
existing utils_signal_analyse module. Its IPython plotting magic is retained.

Validation: six scientific regression tests cover recovery of a known signed
plane wave, rotation across 0/360 degrees, uncertainty scaling, nonmutating sign
correction, signed radial angles, incompatible signs and boundary flags. Both
main plotting modes and the spike script have been run. All Python files pass
syntax checks; notebook code is checked after removing IPython magic lines. A
static scan of 172 literal observation filename references found all files at
their migrated locations. Other scientific workflows have not been rerun end to
end; they retain their own dependencies and case selections.

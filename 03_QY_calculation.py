"""
Quantum Yield Calculation — Ladányi method
Based on: Ladányi et al., Photochem. Photobiol. Sci., 2017, 16, 1757–1763
https://pubs.rsc.org/en/content/articlelanding/2017/pp/c7pp00315c
"""

import os
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy import constants
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def setup_logger(params):
    log_path = os.path.join(params['Results Folder'], params['Basename'] + '.log')
    logger = logging.getLogger("analysis_logger")
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    logger.info("Logging initialized")
    params['logger'] = logger
    return params


# ─────────────────────────────────────────────────────────────────────────────
# Photon flux
# ─────────────────────────────────────────────────────────────────────────────

CALIBRATION_DATA = {
    "wavelength": [
        260,270,280,290,300,310,320,330,340,350,360,370,380,390,400,
        410,420,430,440,450,460,470,480,490,500,510,520,530,540,550,
        560,570,580,590,600,610,620,630,640,650,660
    ],
    "high": [
        10.042,9.892,9.493,8.934,8.301,7.654,7.036,6.467,5.956,5.503,
        5.105,4.758,4.454,4.189,3.958,3.756,3.578,3.423,3.285,3.164,
        3.056,2.960,2.874,2.797,2.727,2.663,2.605,2.552,2.502,2.455,
        2.410,2.367,2.325,2.284,2.244,2.203,2.162,2.121,2.080,2.038,
        1.995
    ],
    "low": [
        0.502,0.495,0.475,0.447,0.415,0.383,0.352,0.323,0.298,0.275,
        0.255,0.238,0.223,0.209,0.198,0.188,0.179,0.171,0.164,0.158,
        0.153,0.148,0.144,0.140,0.136,0.133,0.130,0.128,0.125,0.123,
        0.120,0.118,0.116,0.114,0.112,0.110,0.108,0.106,0.104,0.102,
        0.100
    ]
}

def get_correction_factor(wavelength, channel):
    return np.interp(wavelength, CALIBRATION_DATA['wavelength'], CALIBRATION_DATA[channel])

def calculate_photon_flux(params):
    print_func = params['Print function']
    logger     = params['logger']
    wl         = params['Irradiation wavelength']
    voltage    = params['Voltage after ref']
    volume     = params['Volume']
    cf         = get_correction_factor(wl, params['Channel'])

    photon_flux_per_vol = wl * voltage * cf / (constants.Planck * constants.speed_of_light * constants.Avogadro * volume) * 1e-9
    photon_flux         = photon_flux_per_vol * volume * 1e-3
    abs_photon_flux     = photon_flux * constants.Avogadro

    print_func(f'Correction factor  = {cf:.2f} mW V\u207b\u00b9')
    print_func('Photon flux        = {:.2e} mol s\u207b\u00b9 L\u207b\u00b9 / {:.2e} mol s\u207b\u00b9 / {:.2e} s\u207b\u00b9'.format(
        photon_flux_per_vol, photon_flux, abs_photon_flux))
    logger.info(f'Correction factor = {cf:.2f} mW V-1')
    logger.info(f'Photon flux = {photon_flux_per_vol:.4e} mol s-1 L-1 / {photon_flux:.4e} mol s-1 / {abs_photon_flux:.4e} s-1')

    params['Photon flux per vol'] = photon_flux_per_vol
    params['Photon flux']         = photon_flux
    params['Abs photon flux']     = abs_photon_flux
    return params


# ─────────────────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────────────────

def calculate_concentration_profile(params):
    """Convert absorbance vs time into concentration profiles using absorption coefficients.

    For spectra series: uses the same weighted analytical formula as the E-step
    in the microgel iterative fit:
        x_B(lambda) = (A(lambda) - c*eps_A(lambda)) / (c*(eps_B - eps_A)(lambda))
    averaged over [Concentration WL min, Concentration WL max],
    weighted by |eps_B - eps_A|, clipped to [0, 1].
    For single-wavelength kinetics: falls back to the analytical single-point formula.
    """
    from scipy.interpolate import interp1d as _interp1d

    df_kinetics  = params['Kinetics df']
    df_abs_coeff = params['Abs coeff df']
    c            = params['Concentration']
    wl_min       = params['Concentration WL min']
    wl_max       = params['Concentration WL max']
    time         = df_kinetics.columns[1:].astype(float).to_numpy()
    has_spectra  = params['Has spectra']

    wl_eps = df_abs_coeff.iloc[:, 0].to_numpy(dtype=float)
    eps_A  = df_abs_coeff.iloc[:, 1].to_numpy(dtype=float)
    eps_B  = df_abs_coeff.iloc[:, 2].to_numpy(dtype=float)

    if has_spectra:
        wl_obs = df_kinetics.iloc[:, 0].to_numpy(dtype=float)

        wl_lo  = max(wl_obs.min(), wl_eps.min(), wl_min)
        wl_hi  = min(wl_obs.max(), wl_eps.max(), wl_max)
        wl_fit = wl_obs[(wl_obs >= wl_lo) & (wl_obs <= wl_hi)]

        eps_A_fit = _interp1d(wl_eps, eps_A, kind='cubic',
                               fill_value='extrapolate')(wl_fit)
        eps_B_fit = _interp1d(wl_eps, eps_B, kind='cubic',
                               fill_value='extrapolate')(wl_fit)

        delta_eps = eps_B_fit - eps_A_fit
        weight    = np.abs(delta_eps)

        x_B_arr         = []
        absorbance_list = []
        for col_idx in range(1, df_kinetics.shape[1]):
            A = _interp1d(wl_obs, df_kinetics.iloc[:, col_idx].to_numpy(dtype=float),
                          kind='linear', bounds_error=False,
                          fill_value=np.nan)(wl_fit)
            num    = A - c * eps_A_fit
            den    = c * delta_eps
            x_wl   = np.where(np.abs(den) > 1e-12, num / den, np.nan)
            x_clip = np.clip(x_wl, 0, 1)
            fin    = np.isfinite(x_clip)
            x_B_arr.append(
                float(np.average(x_clip[fin], weights=weight[fin]))
                if fin.any() else np.nan
            )
            absorbance_list.append(np.nanmean(A))

        x_B        = np.clip(np.array(x_B_arr), 0, 1)
        absorbance = np.array(absorbance_list)

    else:
        mask_eps   = (wl_eps >= wl_min) & (wl_eps <= wl_max)
        e_A        = float(np.nanmean(eps_A[mask_eps]))
        e_B        = float(np.nanmean(eps_B[mask_eps]))
        absorbance = df_kinetics.iloc[0, 1:].to_numpy(dtype=float)
        x_B        = np.clip((absorbance - c * e_A) / (c * (e_B - e_A)), 0, 1)

    x_A = 1 - x_B

    if x_A[0] >= x_A[-1]:
        names = [df_abs_coeff.columns[1], df_abs_coeff.columns[2]]
        c_A, c_B = x_A * c * 1e6, x_B * c * 1e6
    else:
        names = [df_abs_coeff.columns[2], df_abs_coeff.columns[1]]
        c_A, c_B = x_B * c * 1e6, x_A * c * 1e6

    df_conc = pd.DataFrame({'time / s': time, 'c_A / uM': c_A, 'c_B / uM': c_B})
    df_conc.to_csv(os.path.join(params['Results Folder'], f"{params['Basename']}_conc.csv"), index=False)

    params['Names']   = names
    params['Conc df'] = df_conc
    return params


def report_cis_fraction(params):
    """Print and log the cis and trans fractions at t=0 and at PSS (last time point)."""
    print_func = params['Print function']
    logger     = params['logger']
    df_conc    = params['Conc df']
    names      = params['Names']   # names[0] = dominant at t=0, names[1] = builds up under irradiation

    c_A = df_conc['c_A / uM'].to_numpy()
    c_B = df_conc['c_B / uM'].to_numpy()
    c_total = c_A + c_B

    # Always report both cis and trans fractions with fixed labels.
    # names[0] is the species dominant at t=0; names[1] builds up.
    # We identify which name contains 'cis' or 'trans' to assign labels correctly.
    name_A, name_B = names[0], names[1]
    frac_A_initial = c_A[0]  / c_total[0]  * 100
    frac_B_initial = c_B[0]  / c_total[0]  * 100
    frac_A_pss     = c_A[-1] / c_total[-1] * 100
    frac_B_pss     = c_B[-1] / c_total[-1] * 100

    print_func(f"Fraction {name_A} at t=0  : {frac_A_initial:.1f} %")
    print_func(f"Fraction {name_B} at t=0  : {frac_B_initial:.1f} %")
    print_func(f"Fraction {name_A} at PSS  : {frac_A_pss:.1f} %")
    print_func(f"Fraction {name_B} at PSS  : {frac_B_pss:.1f} %")
    logger.info(f"{name_A} fraction — t=0: {frac_A_initial:.2f} %, PSS: {frac_A_pss:.2f} %")
    logger.info(f"{name_B} fraction — t=0: {frac_B_initial:.2f} %, PSS: {frac_B_pss:.2f} %")

    params['Cis fraction initial'] = frac_B_initial if 'cis' in name_B.lower() else frac_A_initial
    params['Cis fraction PSS']     = frac_B_pss     if 'cis' in name_B.lower() else frac_A_pss
    return params


def ladanyi_fit(params):
    """Fit concentration profiles to the Ladányi ODE model to extract quantum yields."""
    print_func = params['Print function']
    logger     = params['logger']
    V          = params['Volume'] * 1e-3
    I0         = params['Abs photon flux']
    df_conc    = params['Conc df']
    df_abs_coeff = params['Abs coeff df']
    names      = params['Names']

    wl_idx = (df_abs_coeff.iloc[:, 0] - params['Irradiation wavelength']).abs().argmin()
    # e_A and e_B must correspond to the SAME species as c_A and c_B.
    # calculate_concentration_profile may swap the columns so that c_A is always
    # the species that decreases; params['Names'] records which original column
    # each corresponds to. Look up eps by species name, not by fixed column index.
    _coeff_cols = list(df_abs_coeff.columns)
    e_A = float(df_abs_coeff.iloc[wl_idx, _coeff_cols.index(names[0])])
    e_B = float(df_abs_coeff.iloc[wl_idx, _coeff_cols.index(names[1])])

    c_A   = df_conc['c_A / uM'].to_numpy() * 1e-6
    c_B   = df_conc['c_B / uM'].to_numpy() * 1e-6
    times = df_conc['time / s'].to_numpy()
    cA0, cB0 = c_A[0], c_B[0]

    # Linear least-squares pre-estimate for initial guess.
    # Derived directly from the ODE so the units match exactly:
    #   dcA/dt = P * (phi_B*cB*eB - phi_A*cA*eA),
    #   P = I0/(V*NA) * (1 - 10^-A_tot)/A_tot   (same prefactor as dcA_dt below)
    # Discretising dcA/dt ≈ ΔcA/Δt and evaluating P, cA, cB at interval midpoints
    # gives a linear system in [phi_A, phi_B]:
    #   ΔcA/Δt = (-P_m*eA*cA_m) * phi_A + (P_m*eB*cB_m) * phi_B
    dc_A  = np.diff(c_A)
    dt    = np.diff(times)
    c_A_m = 0.5 * (c_A[:-1] + c_A[1:])
    c_B_m = 0.5 * (c_B[:-1] + c_B[1:])
    A_m   = e_A * c_A_m + e_B * c_B_m

    # Prefactor P evaluated at the midpoints — identical grouping to the ODE
    with np.errstate(divide='ignore', invalid='ignore'):
        P_m = (I0 / (V * constants.Avogadro)) * (1 - 10**(-A_m)) / A_m

    dcA_dt_meas = dc_A / dt                       # measured rate of change
    X = np.column_stack([-P_m * e_A * c_A_m,      # coefficient of phi_A
                          P_m * e_B * c_B_m])     # coefficient of phi_B
    y = dcA_dt_meas

    finite_mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
    Phi_est, *_ = np.linalg.lstsq(X[finite_mask], y[finite_mask], rcond=None)
    # Clip the pre-estimate to physically plausible quantum yields [0, 1]
    phi_A_guess = float(np.clip(Phi_est[0], 0.0, 1.0))
    phi_B_guess = float(np.clip(Phi_est[1], 0.0, 1.0))
    initial_guess = [phi_A_guess, phi_B_guess]
    print_func(f"Pre-estimate: \u03a6({names[0]}\u2192{names[1]})\u2248{phi_A_guess:.3f}, "
               f"\u03a6({names[1]}\u2192{names[0]})\u2248{phi_B_guess:.3f}")

    def dcA_dt(t, y, phi_A, phi_B):
        cA, cB = y
        A_tot  = cA * e_A + cB * e_B
        if A_tot <= 0:
            return [0.0, 0.0]
        prefactor = I0 / (V * constants.Avogadro)
        dcA = prefactor * (1 - 10**(-A_tot)) / A_tot * (phi_B * cB * e_B - phi_A * cA * e_A)
        return [dcA, -dcA]

    def model(times, phi_A, phi_B):
        sol = solve_ivp(dcA_dt, (times[0], times[-1]), y0=[cA0, cB0],
                        t_eval=times, args=(phi_A, phi_B), method="LSODA")
        return sol.y[0]

    popt, _ = curve_fit(model, times, c_A, p0=initial_guess)
    phi_A_fit, phi_B_fit = popt

    print_func(f"\u03a6({names[0]}\u2192{names[1]}) = {phi_A_fit:.4e}  ({phi_A_fit*100:.2f} %)")
    print_func(f"\u03a6({names[1]}\u2192{names[0]}) = {phi_B_fit:.4e}  ({phi_B_fit*100:.2f} %)")
    logger.info(f'QY {names[0]}->{names[1]} = {phi_A_fit:.4e} ({phi_A_fit*100:.2f} %), '
                f'{names[1]}->{names[0]} = {phi_B_fit:.4e} ({phi_B_fit*100:.2f} %)')

    params['Ladanyi QY A->B'] = phi_A_fit
    params['Ladanyi QY B->A'] = phi_B_fit

    cA_fit = model(times, phi_A_fit, phi_B_fit)
    pd.DataFrame({'time / s': times, 'cA exp': c_A, 'cA fit': cA_fit}).to_csv(
        os.path.join(params['Results Folder'], f"{params['Basename']}_ladanyi_fit.csv"), index=False)

    # ── Photons required to reach the PSS ─────────────────────────────────────
    # Integrate the fitted ODE on a fine grid far beyond the measured window to
    # find the asymptotic (PSS) composition, then determine the time at which the
    # system first comes within PSS_TOL of that asymptote.
    PSS_TOL = 0.01   # fraction of the total change remaining (1 %)

    # Integrate well past the last measured time to approximate the true PSS
    t_long = np.linspace(times[0], times[-1] * 20, 20000)
    sol_long = solve_ivp(dcA_dt, (t_long[0], t_long[-1]), y0=[cA0, cB0],
                         t_eval=t_long, args=(phi_A_fit, phi_B_fit), method="LSODA")
    cA_long = sol_long.y[0]
    cB_long = sol_long.y[1]
    cA_pss  = cA_long[-1]   # asymptotic c_A at PSS

    # Time to reach PSS: first time |cA(t) - cA_pss| <= PSS_TOL * |cA_pss - cA0|
    total_change = abs(cA_pss - cA0)
    if total_change > 0:
        within = np.abs(cA_long - cA_pss) <= PSS_TOL * total_change
        idx_pss = np.argmax(within) if within.any() else len(t_long) - 1
        t_pss   = t_long[idx_pss]
    else:
        t_pss = 0.0

    # Incident photons up to t_pss
    Q_incident = I0 * t_pss   # photons s^-1 * s = photons (since I0 = Abs photon flux in s^-1)

    # Absorbed photons up to t_pss: integrate q_abs(t) = I0 * (1 - 10^-A_tot(t)) dt
    mask_pss   = t_long <= t_pss
    A_tot_long = cA_long * e_A + cB_long * e_B
    q_abs_t    = I0 * (1 - 10.0**(-A_tot_long))
    Q_absorbed = np.trapz(q_abs_t[mask_pss], t_long[mask_pss])

    # Express also in mol of photons (einstein)
    Q_incident_mol = Q_incident / constants.Avogadro
    Q_absorbed_mol = Q_absorbed / constants.Avogadro

    x_A_pss = cA_pss / (cA_pss + (cA0 + cB0 - cA_pss))  # = cA_pss / (cA0+cB0)

    print_func(f"\nPSS reached at t = {t_pss:.1f} s  (within {PSS_TOL*100:.0f} % of asymptote)")
    print_func(f"  Incident photons to PSS: {Q_incident:.3e}  ({Q_incident_mol:.3e} mol)")
    print_func(f"  Absorbed photons to PSS: {Q_absorbed:.3e}  ({Q_absorbed_mol:.3e} mol)")
    logger.info(f"PSS at t={t_pss:.2f} s; Q_incident={Q_incident:.4e} ({Q_incident_mol:.4e} mol), "
                f"Q_absorbed={Q_absorbed:.4e} ({Q_absorbed_mol:.4e} mol)")

    params['t_PSS / s']           = t_pss
    params['Q_incident_to_PSS']   = Q_incident
    params['Q_absorbed_to_PSS']   = Q_absorbed
    params['Q_incident_mol']      = Q_incident_mol
    params['Q_absorbed_mol']      = Q_absorbed_mol

    return params


def run_analysis(params):
    setup_logger(params)
    logger = params['logger']
    logger.info('Starting Ladányi QY analysis')
    logger.info(f"Irradiation: {params['Irradiation wavelength']} nm, "
                f"channel: {params['Channel']}, voltage: {params['Voltage after ref']} V")
    logger.info(f"Concentration WL range: {params['Concentration WL min']}-{params['Concentration WL max']} nm")
    logger.info(f"Volume: {params['Volume']} mL, concentration: {params['Concentration']*1e6:.1f} µM")

    calculate_photon_flux(params)

    df_abs_coeff = pd.read_csv(params['Abs coeff file'])
    params['Abs coeff df'] = df_abs_coeff

    calculate_concentration_profile(params)
    report_cis_fraction(params)
    ladanyi_fit(params)
    return params


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plotting_spectra(params, ax):
    df        = params['Kinetics df']
    times     = df.columns[1:].astype(float).to_numpy()
    palette   = params['Color palette']
    ymin, ymax = params['Plotting A_min'], params['Plotting A_max']
    xmin, xmax = params['Plotting wl_min'], params['Plotting wl_max']

    cmap       = plt.get_cmap(palette)
    norm       = mcolors.Normalize(vmin=times[0], vmax=times[-1])
    colors     = [cmap(norm(t)) for t in times]

    for i, col in enumerate(df.columns[1:]):
        ax.plot(df.iloc[:, 0], df[col], color=colors[i])

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='irradiation time / s')

    ax.set_xlabel('wavelength / nm')
    ax.set_ylabel('absorbance')
    if xmin != 0: ax.set_xlim(left=xmin)
    if xmax != 0: ax.set_xlim(right=xmax)
    if ymin != 0: ax.set_ylim(bottom=ymin)
    if ymax != 0: ax.set_ylim(top=ymax)


def plotting_conc_and_fit(params, ax):
    """Combined plot: concentration profiles (scatter) with Ladányi fit overlaid."""
    V            = params['Volume'] * 1e-3
    I0           = params['Abs photon flux']
    df_conc      = params['Conc df']
    df_abs_coeff = params['Abs coeff df']
    names        = params['Names']
    t_min        = params['Plotting t_min']
    t_max        = params['Plotting t_max']

    wl_idx = (df_abs_coeff.iloc[:, 0] - params['Irradiation wavelength']).abs().argmin()
    # e_A and e_B must correspond to the SAME species as c_A and c_B.
    # calculate_concentration_profile may swap the columns so that c_A is always
    # the species that decreases; params['Names'] records which original column
    # each corresponds to. Look up eps by species name, not by fixed column index.
    _coeff_cols = list(df_abs_coeff.columns)
    e_A = float(df_abs_coeff.iloc[wl_idx, _coeff_cols.index(names[0])])
    e_B = float(df_abs_coeff.iloc[wl_idx, _coeff_cols.index(names[1])])

    c_A   = df_conc['c_A / uM'].to_numpy() * 1e-6
    c_B   = df_conc['c_B / uM'].to_numpy() * 1e-6
    times = df_conc['time / s'].to_numpy()
    cA0, cB0 = c_A[0], c_B[0]
    phi_A = params['Ladanyi QY A->B']
    phi_B = params['Ladanyi QY B->A']

    def dcA_dt(t, y, phi_A, phi_B):
        cA, cB = y
        A_tot  = cA * e_A + cB * e_B
        if A_tot <= 0: return [0.0, 0.0]
        prefactor = I0 / (V * constants.Avogadro)
        dcA = prefactor * (1 - 10**(-A_tot)) / A_tot * (phi_B * cB * e_B - phi_A * cA * e_A)
        return [dcA, -dcA]

    sol    = solve_ivp(dcA_dt, (times[0], times[-1]), y0=[cA0, cB0],
                       t_eval=times, args=(phi_A, phi_B), method="LSODA")
    cA_fit = sol.y[0]
    cB_fit = (cA0 + cB0) - cA_fit   # mass balance in mol/L, convert below

    # Scatter points
    ax.scatter(times, c_A * 1e6, color='C0', s=12, zorder=2, label=names[0])
    ax.scatter(times, c_B * 1e6, color='C1', s=12, zorder=2, label=names[1])
    # Fit lines overlaid in matching colours
    ax.plot(times, cA_fit * 1e6, color='C0', linewidth=1.5, zorder=3)
    ax.plot(times, cB_fit * 1e6, color='black', linewidth=1.5, zorder=3, label='fit')

    ax.set_xlabel('irradiation time / s')
    ax.set_ylabel('concentration / µM')
    if t_min != 0: ax.set_xlim(left=t_min)
    if t_max != 0: ax.set_xlim(right=t_max)
    ax.legend()


def save_individual_plot(plot_func, params, suffix):
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_func(params, ax)
    fig.tight_layout()
    fig.savefig(os.path.join(params['Results Folder'], params['Basename'] + suffix + '.svg'),
                bbox_inches='tight')
    plt.close(fig)


def plot_results(params, figure):
    figure.clf()
    has_spectra = params['Has spectra']
    n_plots = 2 if has_spectra else 1
    axs = figure.subplots(1, n_plots)

    if has_spectra:
        plotting_spectra(params, axs[0])
        plotting_conc_and_fit(params, axs[1])
        save_individual_plot(plotting_spectra,      params, '_spectra')
        save_individual_plot(plotting_conc_and_fit, params, '_conc_fit')
    else:
        plotting_conc_and_fit(params, axs[0])
        save_individual_plot(plotting_conc_and_fit, params, '_conc_fit')

    figure.savefig(os.path.join(params['Results Folder'], params['Basename'] + '_results.svg'),
                   bbox_inches='tight')


# ─────────────────────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────────────────────

def run_gui():
    root = tk.Tk()
    root.title("QY Calculation")
    root.resizable(True, True)

    root.grid_columnconfigure(0, weight=0, minsize=320)
    root.grid_columnconfigure(1, weight=1)
    root.grid_rowconfigure(0, weight=1)

    # ── Helpers ──────────────────────────────────────────────────────────────
    def choose_file(var, title="Open file"):
        path = filedialog.askopenfilename(title=title, filetypes=[("All files", "*.*")])
        if path: var.set(path)

    def choose_folder(var):
        path = filedialog.askdirectory()
        if path: var.set(path)

    def file_row(parent, label, var, row, title="Open file"):
        ttk.Label(parent, text=label).grid(row=row*2, column=0, sticky="w", pady=(6,0))
        f = ttk.Frame(parent)
        f.grid(row=row*2+1, column=0, sticky="ew", pady=2)
        f.grid_columnconfigure(0, weight=1)
        ttk.Entry(f, textvariable=var).grid(row=0, column=0, sticky="ew", padx=(0,4))
        ttk.Button(f, text="Browse", command=lambda: choose_file(var, title), width=7).grid(row=0, column=1)

    # ── Left panel ───────────────────────────────────────────────────────────
    left = ttk.Frame(root, padding=10)
    left.grid(row=0, column=0, sticky="nsew")
    left.grid_columnconfigure(0, weight=1)

    # File selection
    file_frame = ttk.LabelFrame(left, text="Files", padding=8)
    file_frame.grid(row=0, column=0, sticky="ew", pady=(0,8))
    file_frame.grid_columnconfigure(0, weight=1)

    var_kinetics    = tk.StringVar()
    var_abs_coeff   = tk.StringVar()
    var_results     = tk.StringVar()

    file_row(file_frame, "Kinetics DataFrame (CSV):",         var_kinetics,  0, "Open kinetics CSV")
    file_row(file_frame, "Absorption coefficients (CSV):",    var_abs_coeff, 1, "Open absorption coefficients CSV")

    ttk.Label(file_frame, text="Results folder:").grid(row=4, column=0, sticky="w", pady=(6,0))
    f_res = ttk.Frame(file_frame)
    f_res.grid(row=5, column=0, sticky="ew", pady=2)
    f_res.grid_columnconfigure(0, weight=1)
    ttk.Entry(f_res, textvariable=var_results).grid(row=0, column=0, sticky="ew", padx=(0,4))
    ttk.Button(f_res, text="Browse", command=lambda: choose_folder(var_results), width=7).grid(row=0, column=1)

    # Parameters
    param_frame = ttk.LabelFrame(left, text="Parameters", padding=8)
    param_frame.grid(row=1, column=0, sticky="ew", pady=(0,8))
    param_frame.grid_columnconfigure(1, weight=1)

    var_wl      = tk.StringVar(value='325')
    var_wl_conc_min = tk.StringVar(value='330')
    var_wl_conc_max = tk.StringVar(value='380')
    var_voltage = tk.StringVar(value='7.465')
    var_volume  = tk.StringVar(value='3')
    var_conc    = tk.StringVar(value='40')
    var_channel = tk.StringVar(value='low')

    param_entries = [
        ("Irradiation wavelength / nm",      var_wl),
        ("Concentration WL min / nm",         var_wl_conc_min),
        ("Concentration WL max / nm",         var_wl_conc_max),
        ("Voltage after reference / V",       var_voltage),
        ("Volume / mL",                       var_volume),
        ("Concentration / µM",                var_conc),
    ]
    for i, (lbl, var) in enumerate(param_entries):
        ttk.Label(param_frame, text=lbl + ":").grid(row=i, column=0, sticky="w", pady=2, padx=(0,6))
        ttk.Entry(param_frame, textvariable=var, width=12).grid(row=i, column=1, sticky="w", padx=4)

    ttk.Label(param_frame, text="Channel:").grid(row=6, column=0, sticky="w", pady=2, padx=(0,6))
    ttk.Combobox(param_frame, textvariable=var_channel, values=["low","high"],
                 state="readonly", width=10).grid(row=6, column=1, sticky="w", padx=4)

    # Plotting parameters
    plot_frame = ttk.LabelFrame(left, text="Plot Parameters", padding=8)
    plot_frame.grid(row=2, column=0, sticky="ew", pady=(0,8))
    plot_frame.grid_columnconfigure(1, weight=1)

    var_palette = tk.StringVar(value='viridis')
    var_A_min   = tk.StringVar(value='0')
    var_A_max   = tk.StringVar(value='0')
    var_wl_min  = tk.StringVar(value='0')
    var_wl_max  = tk.StringVar(value='0')
    var_t_min   = tk.StringVar(value='0')
    var_t_max   = tk.StringVar(value='0')

    plot_entries = [
        ("Color palette",  var_palette),
        ("A min",          var_A_min),
        ("A max",          var_A_max),
        ("λ min / nm",     var_wl_min),
        ("λ max / nm",     var_wl_max),
        ("t min / s",      var_t_min),
        ("t max / s",      var_t_max),
    ]
    for i, (lbl, var) in enumerate(plot_entries):
        ttk.Label(plot_frame, text=lbl + ":").grid(row=i, column=0, sticky="w", pady=2, padx=(0,6))
        ttk.Entry(plot_frame, textvariable=var, width=12).grid(row=i, column=1, sticky="w", padx=4)

    # Run button
    run_btn = ttk.Button(left, text="Run Analysis")
    run_btn.grid(row=3, column=0, sticky="ew", pady=(4,0), ipady=4)

    # ── Right panel ──────────────────────────────────────────────────────────
    right = ttk.Frame(root, padding=10)
    right.grid(row=0, column=1, sticky="nsew")
    right.grid_rowconfigure(1, weight=1)
    right.grid_rowconfigure(3, weight=3)
    right.grid_columnconfigure(0, weight=1)

    ttk.Label(right, text="Output", font=("TkDefaultFont", 10, "bold")).grid(row=0, column=0, sticky="w")
    output_box = tk.Text(right, height=8, wrap="word", font=("TkFixedFont", 9))
    output_box.grid(row=1, column=0, sticky="nsew", pady=(4,10))

    ttk.Label(right, text="Plots", font=("TkDefaultFont", 10, "bold")).grid(row=2, column=0, sticky="w")
    fig = plt.Figure(figsize=(14, 5), dpi=100)
    canvas = FigureCanvasTkAgg(fig, master=right)
    canvas.get_tk_widget().grid(row=3, column=0, sticky="nsew", pady=(4,0))

    def _float(s):
        """Parse float accepting both '.' and ',' as decimal separator."""
        return float(str(s).replace(',', '.'))

    # ── Run callback ─────────────────────────────────────────────────────────
    def run():
        output_box.delete("1.0", tk.END)
        fig.clear()
        canvas.draw()

        def printer(msg):
            output_box.insert(tk.END, msg + "\n")
            output_box.see(tk.END)
            root.update_idletasks()

        # Validate files
        kin_file    = var_kinetics.get().strip()
        coeff_file  = var_abs_coeff.get().strip()
        results_dir = var_results.get().strip()

        if not kin_file or not os.path.isfile(kin_file):
            messagebox.showerror("Missing file", "Please select a valid kinetics DataFrame CSV.")
            return
        if not coeff_file or not os.path.isfile(coeff_file):
            messagebox.showerror("Missing file", "Please select a valid absorption coefficients CSV.")
            return
        if not results_dir or not os.path.isdir(results_dir):
            messagebox.showerror("Missing folder", "Please select a valid results folder.")
            return

        # Load kinetics df and detect whether it has spectra (>1 row)
        df_kinetics = pd.read_csv(kin_file)
        df_kinetics = df_kinetics.dropna(axis=1)   # drop time columns that contain any NaN
        has_spectra = df_kinetics.shape[0] > 1
        basename    = os.path.splitext(os.path.basename(kin_file))[0]

        params = {
            'Print function':       printer,
            'Kinetics df':          df_kinetics,
            'Has spectra':          has_spectra,
            'Abs coeff file':       coeff_file,
            'Results Folder':       results_dir,
            'Basename':             basename,
            'Irradiation wavelength':  _float(var_wl.get()),
            'Concentration WL min':      _float(var_wl_conc_min.get()),
            'Concentration WL max':      _float(var_wl_conc_max.get()),
            'Voltage after ref':    _float(var_voltage.get()),
            'Volume':               _float(var_volume.get()),
            'Concentration':        _float(var_conc.get()) * 1e-6,
            'Channel':              var_channel.get(),
            'Color palette':        var_palette.get(),
            'Plotting A_min':       _float(var_A_min.get()),
            'Plotting A_max':       _float(var_A_max.get()),
            'Plotting wl_min':      _float(var_wl_min.get()),
            'Plotting wl_max':      _float(var_wl_max.get()),
            'Plotting t_min':       _float(var_t_min.get()),
            'Plotting t_max':       _float(var_t_max.get()),
        }

        printer(f"Kinetics df: {df_kinetics.shape[0]} row(s) × {df_kinetics.shape[1]} columns "
                f"({'spectra series' if has_spectra else 'single-wavelength kinetics'})")

        try:
            run_analysis(params)
            plot_results(params, fig)
            canvas.draw()
            printer("\nDone. Results saved to results folder.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            raise

    run_btn.config(command=run)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            root.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    import tkinter as _tk_check
    try:
        _tk_check.Tk().destroy()
    except Exception as _e:
        print(f"Warning: could not initialise Tk ({_e}). Try restarting the Python kernel.")
    run_gui()

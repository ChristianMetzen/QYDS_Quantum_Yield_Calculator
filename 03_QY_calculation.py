"""
Quantum Yield Calculation — Ladányi method
Based on: Ladányi et al., Photochem. Photobiol. Sci., 2017, 16, 1757–1761
https://pubs.rsc.org/en/content/articlelanding/2017/pp/c7pp00315c

Kinetic model (eqn 5/6 of the paper, written for species A = 1st column and
B = 2nd column of the absorption coefficient file, path length 1 cm):

    dc_A/dt = q/V * (1 - 10^-A_tot)/A_tot * (Phi_B*c_B*eps_B - Phi_A*c_A*eps_A)
              + k_BA*c_B - k_AB*c_A
    c_B     = c_0 - c_A
    A_tot   = eps_A*c_A + eps_B*c_B

q    : incident photon flux / mol s-1  (from 01a_Ferrioxalate_Actinometry.ipynb)
V    : irradiated volume / L
k_BA : thermal B -> A rate constant (e.g. cis -> trans), k_AB the reverse.
       Only one of them is used (thermal relaxation goes one way).

Fitted parameters: Phi_A (A->B), Phi_B (B->A) and the initial fraction x_A(0).
c_0 is the mean of c_A + c_B over all spectra (as in the paper).
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
from scipy.integrate import solve_ivp, trapezoid
from scipy.interpolate import interp1d

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


SAVE_RC = {'font.size': 14}   # larger fonts for the saved 8x6 figures


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
# LED / absorption coefficients
# ─────────────────────────────────────────────────────────────────────────────

# ── LED lookup table (nominal WL -> (central WL nm, FWHM nm)) ────────────────
LED_CATALOGUE = {
    "278 nm  (LG Innotek LEUVA66G00HV00)":  (280,  12),
    "305 nm  (LG Innotek LEUVA66H100KU00)": (307,  13),
    "325 nm  (Dowa EOLS-325-969)":           (328,  14),
    "340 nm  (Seoul Viosys CUD4AF1B)":       (348,  10),
    "365 nm  (Seoul Viosys CUN6GB1A)":       (370,   9),
    "385 nm  (Seoul Viosys CUN8GF1A)":       (387,  10),
    "400 nm  (Luxeon LHUV-0400-0450)":       (403,  11),
    "415 nm  (Luxeon LHUV-0415-0600)":       (421,  13),
    "420 nm  (Luxeon LHUV-0420-0650)":       (426,  13),
    "425 nm  (Luxeon LHUV-0425-0600)":       (425,  13),
    "455 nm  (Osram LD-CQ7P-1U3U)":         (455,  21),
    "470 nm  (Osram LBCP7P-GYHY)":          (468,  16),
    "505 nm  (Osram LVCK7P-JYKZ)":          (507,  33),
    "528 nm  (Osram LTCP7P-KXKZ)":          (529,  36),
    "567 nm  (Luxeon Rebel ES Lime)":        (567, 103),
    "590 nm  (Osram LYCP7P-JRJT)":          (592,  17),
    "617 nm  (Osram LACP7P-KPKR)":          (620,  18),
    "625 nm  (Osram LRCP7P-JRJT)":          (630,  19),
    "645 nm  (Osram LHCP7P-2T3T)":          (656,  23),
    "660 nm  (Osram Hyper Red)":             (656,  17),
    "690 nm  (Roithner SMB1N-690D)":         (687,  22),
    "730 nm  (Osram Far Red)":               (728,  31),
    "780 nm  (Roithner SMB1N-780-02)":       (772,  28),
    "810 nm  (Roithner SMB1N-810D-02)":      (802,  27),
    "850 nm  (Osram ILH-IW01-85SL)":        (858,  30),
    "890 nm  (Roithner SMB1N-890DS)":        (880,  32),
    "940 nm  (Osram ILH-IO01-94SL)":        (940,  38),
    "980 nm  (Roithner SMB1N-980D)":         (949,  38),
    "1050 nm (Roithner SMB1N-1050-02)":      (1050, 600),
    "Custom (enter below)":                  (None, None),
}


def effective_epsilon(wl_eps, eps_values, led_center, led_fwhm):
    """Flux-weighted effective absorption coefficient for a Gaussian LED spectrum."""
    sigma = led_fwhm / (2 * np.sqrt(2 * np.log(2)))
    S     = np.exp(-0.5 * ((wl_eps - led_center) / sigma) ** 2)
    mask  = (wl_eps >= led_center - 4*led_fwhm) & (wl_eps <= led_center + 4*led_fwhm)
    if mask.sum() < 2:
        return float(eps_values[np.abs(wl_eps - led_center).argmin()])
    norm = trapezoid(S[mask], wl_eps[mask])
    return float(trapezoid(S[mask] * eps_values[mask], wl_eps[mask]) / norm)


def irradiation_epsilons(params):
    """eps_A, eps_B at the irradiation wavelength (LED-weighted if FWHM > 0)."""
    df  = params['Abs coeff df']
    wl  = df.iloc[:, 0].to_numpy(dtype=float)
    eA  = df.iloc[:, 1].to_numpy(dtype=float)
    eB  = df.iloc[:, 2].to_numpy(dtype=float)
    ctr = params['Irradiation wavelength']
    fw  = params.get('LED FWHM', 0.0)
    if fw > 0:
        return effective_epsilon(wl, eA, ctr, fw), effective_epsilon(wl, eB, ctr, fw)
    i = np.abs(wl - ctr).argmin()
    return float(eA[i]), float(eB[i])


# ─────────────────────────────────────────────────────────────────────────────
# Kinetic model (shared by fit and plot)
# ─────────────────────────────────────────────────────────────────────────────

def thermal_rates(params):
    """Return (k_AB, k_BA) in s-1 from the thermal relaxation setting."""
    k   = params.get('Thermal k', 0.0)
    mode = params.get('Thermal relaxation', 'none')
    if mode == 'B->A':
        return 0.0, k
    if mode == 'A->B':
        return k, 0.0
    return 0.0, 0.0


def simulate_xA(times, phi_A, phi_B, xA0, c0, e_A, e_B, q_per_V, k_AB, k_BA):
    """Solve the Ladányi ODE for the fraction x_A = c_A/c_0.

    Solved in fractions (order 1) instead of mol/L (order 1e-5), so the default
    absolute tolerance of the solver does not swamp the solution.
    """
    def rhs(t, y):
        xA = y[0]
        xB = 1.0 - xA
        A  = c0 * (e_A * xA + e_B * xB)
        F  = np.log(10) if A <= 1e-12 else (1 - 10**(-A)) / A   # limit A -> 0
        return [q_per_V * F * (phi_B * xB * e_B - phi_A * xA * e_A)
                + k_BA * xB - k_AB * xA]

    sol = solve_ivp(rhs, (times[0], times[-1]), y0=[xA0], t_eval=times,
                    method="LSODA", rtol=1e-8, atol=1e-10)
    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")
    return sol.y[0]


# ─────────────────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────────────────

def calculate_concentration_profile(params):
    """Convert absorbance vs time into concentrations c_A, c_B.

    Spectra series: two-component linear least squares in the selected
    wavelength range, A(λ) = eps_A(λ)*c_A + eps_B(λ)*c_B (d = 1 cm), done
    independently for every spectrum. No total concentration is needed;
    c_0 is taken as the mean of c_A + c_B (as in the paper).

    Single-wavelength kinetics: x_B from the absorbance at the observation
    wavelength and the entered total concentration.
    """
    print_func   = params['Print function']
    df_kinetics  = params['Kinetics df']
    df_abs_coeff = params['Abs coeff df']
    time         = df_kinetics.columns[1:].astype(float).to_numpy()

    wl_eps = df_abs_coeff.iloc[:, 0].to_numpy(dtype=float)
    eps_A  = df_abs_coeff.iloc[:, 1].to_numpy(dtype=float)
    eps_B  = df_abs_coeff.iloc[:, 2].to_numpy(dtype=float)

    if params['Has spectra']:
        wl_obs = df_kinetics.iloc[:, 0].to_numpy(dtype=float)
        wl_lo  = max(wl_obs.min(), wl_eps.min(), params['Concentration WL min'])
        wl_hi  = min(wl_obs.max(), wl_eps.max(), params['Concentration WL max'])
        mask   = (wl_obs >= wl_lo) & (wl_obs <= wl_hi)
        if mask.sum() < 2:
            raise ValueError("Less than 2 wavelengths in the concentration range.")

        E = np.column_stack([interp1d(wl_eps, eps_A)(wl_obs[mask]),
                             interp1d(wl_eps, eps_B)(wl_obs[mask])])
        A_mat = df_kinetics.iloc[:, 1:].to_numpy(dtype=float)[mask]   # n_wl x n_t
        conc, *_ = np.linalg.lstsq(E, A_mat, rcond=None)
        c_A, c_B = conc
        resid = A_mat - E @ conc
        rms   = np.sqrt(np.mean(resid**2))
        c0    = float(np.mean(c_A + c_B))

        print_func(f"Concentrations: 2-component LSQ, {wl_lo:.0f}-{wl_hi:.0f} nm, "
                   f"RMS residual = {rms:.2e}")
        print_func(f"c0 (mean c_A + c_B) = {c0*1e6:.2f} µM "
                   f"(spread {np.std(c_A + c_B)/c0*100:.1f} %)")
        params['logger'].info(f"LSQ concentrations: range {wl_lo}-{wl_hi} nm, "
                              f"RMS {rms:.3e}, c0 = {c0*1e6:.3f} uM")
    else:
        wl_obs = float(df_kinetics.iloc[0, 0])
        e_A    = float(np.interp(wl_obs, wl_eps, eps_A))
        e_B    = float(np.interp(wl_obs, wl_eps, eps_B))
        c0     = params['Concentration']
        A      = df_kinetics.iloc[0, 1:].to_numpy(dtype=float)
        x_B    = (A - c0 * e_A) / (c0 * (e_B - e_A))
        c_B    = x_B * c0
        c_A    = c0 - c_B
        print_func(f"Concentrations from single wavelength {wl_obs:.0f} nm, "
                   f"c0 = {c0*1e6:.2f} µM (entered)")

    names = [df_abs_coeff.columns[1], df_abs_coeff.columns[2]]
    df_conc = pd.DataFrame({'time / s': time, 'c_A / uM': c_A * 1e6, 'c_B / uM': c_B * 1e6})
    df_conc.to_csv(os.path.join(params['Results Folder'], f"{params['Basename']}_conc.csv"),
                   index=False)

    params['Names']   = names
    params['Conc df'] = df_conc
    params['c0']      = c0
    return params


def report_fractions(params):
    """Print and log the fractions of both species at t=0 and at the last point."""
    print_func = params['Print function']
    df_conc    = params['Conc df']
    nA, nB     = params['Names']
    c_A = df_conc['c_A / uM'].to_numpy()
    c_B = df_conc['c_B / uM'].to_numpy()
    tot = c_A + c_B
    for name, c in ((nA, c_A), (nB, c_B)):
        f0, f1 = c[0] / tot[0] * 100, c[-1] / tot[-1] * 100
        print_func(f"Fraction {name}: t=0 {f0:.1f} %, last point {f1:.1f} %")
        params['logger'].info(f"{name} fraction: t=0 {f0:.2f} %, last {f1:.2f} %")
    return params


def ladanyi_fit(params):
    """Fit the Ladányi ODE (incl. optional thermal relaxation) to c_A(t)."""
    print_func = params['Print function']
    logger     = params['logger']
    df_conc    = params['Conc df']
    nA, nB     = params['Names']
    c0         = params['c0']
    q_per_V    = params['Photon flux'] / (params['Volume'] * 1e-3)   # mol s-1 L-1
    e_A, e_B   = irradiation_epsilons(params)
    k_AB, k_BA = thermal_rates(params)

    times = df_conc['time / s'].to_numpy()
    x_A   = df_conc['c_A / uM'].to_numpy() * 1e-6 / c0
    x_B   = 1 - x_A

    print_func(f"ε({nA}) = {e_A:.0f}, ε({nB}) = {e_B:.0f} L mol⁻¹ cm⁻¹ "
               f"at irradiation")
    if k_AB or k_BA:
        d = f"{nA}→{nB}" if k_AB else f"{nB}→{nA}"
        print_func(f"Thermal relaxation {d}: k = {max(k_AB, k_BA):.3e} s⁻¹ (fixed)")

    # ── Direction ────────────────────────────────────────────────────────────
    product_is_B = (x_B[-1] - x_B[0]) >= 0
    forward_only = params.get('Fit direction', 'both') == 'forward_only'

    # ── Linear pre-estimate (eqn 7) ──────────────────────────────────────────
    xA_m = 0.5 * (x_A[:-1] + x_A[1:])
    xB_m = 1 - xA_m
    A_m  = c0 * (e_A * xA_m + e_B * xB_m)
    with np.errstate(divide='ignore', invalid='ignore'):
        P_m = q_per_V * (1 - 10**(-A_m)) / A_m
    y = np.diff(x_A) / np.diff(times) - (k_BA * xB_m - k_AB * xA_m)
    X = np.column_stack([-P_m * e_A * xA_m, P_m * e_B * xB_m])
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    est, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    phi_A0, phi_B0 = np.clip(est, 1e-4, 1.0)
    print_func(f"Pre-estimate: Φ({nA}→{nB})≈{phi_A0:.3f}, "
               f"Φ({nB}→{nA})≈{phi_B0:.3f}")

    # ── Full fit: free parameters depend on direction ────────────────────────
    # free = which of (phi_A, phi_B) are fitted; the reverse QY is fixed to 0
    # in 'forward only' mode (reverse QY = QY of the product species).
    if not forward_only:
        free = [True, True]
    else:
        free = [product_is_B, not product_is_B]
        rev = f"{nB}→{nA}" if product_is_B else f"{nA}→{nB}"
        print_func(f"Forward only: Φ({rev}) fixed to 0")
        logger.info(f"Forward only: QY {rev} fixed to 0")

    def unpack(p):
        phis, i = [0.0, 0.0], 0
        for j in range(2):
            if free[j]:
                phis[j] = p[i]; i += 1
        return phis[0], phis[1], p[i]

    def model(t, *p):
        phi_A, phi_B, xA0 = unpack(p)
        return simulate_xA(t, phi_A, phi_B, xA0, c0, e_A, e_B, q_per_V, k_AB, k_BA)

    p0 = [g for g, f in zip((phi_A0, phi_B0), free) if f] + [float(np.clip(x_A[0], 0, 1))]
    nphi = sum(free)
    bounds = ([0.0] * nphi + [0.0], [1.0] * nphi + [1.0])
    popt, pcov = curve_fit(model, times, x_A, p0=p0, bounds=bounds)
    perr = np.sqrt(np.diag(pcov))

    phi_A, phi_B, xA0 = unpack(popt)
    err_A, err_B, _   = unpack(perr)
    if not free[0]: err_A = 0.0
    if not free[1]: err_B = 0.0

    print_func(f"\nΦ({nA}→{nB}) = {phi_A:.4f} ± {err_A:.4f}")
    print_func(f"Φ({nB}→{nA}) = {phi_B:.4f} ± {err_B:.4f}")
    print_func(f"x({nA}) at t=0 (fitted) = {xA0*100:.1f} %")
    print_func("(± = 1σ from the fit covariance, not including photon flux error)")
    logger.info(f"QY {nA}->{nB} = {phi_A:.5f} +- {err_A:.5f}, "
                f"{nB}->{nA} = {phi_B:.5f} +- {err_B:.5f}, xA0 = {xA0:.4f}")

    # Fit curves for the plot (smooth) and for export (at data points)
    t_smooth  = np.linspace(times[0], times[-1], 400)
    xA_smooth = model(t_smooth, *popt)
    xA_fit    = model(times, *popt)
    pd.DataFrame({'time / s': times,
                  'c_A exp / uM': x_A * c0 * 1e6,
                  'c_A fit / uM': xA_fit * c0 * 1e6}).to_csv(
        os.path.join(params['Results Folder'], f"{params['Basename']}_ladanyi_fit.csv"),
        index=False)

    params['Ladanyi QY A->B'] = phi_A
    params['Ladanyi QY B->A'] = phi_B
    params['Ladanyi t_smooth'] = t_smooth
    params['Ladanyi cA_smooth'] = xA_smooth * c0 * 1e6
    params['Ladanyi cB_smooth'] = (1 - xA_smooth) * c0 * 1e6
    return params


def fit_photoswitching_kinetics(params):
    """Fit a mono-exponential buildup curve to extract k, t1/2 and p1/2.

    Model:  c_B(t) = c_B_PSS - (c_B_PSS - c_B0) * exp(-k * t)
    t1/2  = ln(2) / k
    p1/2  = t1/2 * P   with P the incident optical power from the photon flux.
    """
    print_func = params['Print function']
    df_conc    = params['Conc df']
    nB         = params['Names'][1]

    times = df_conc['time / s'].to_numpy()
    c_B   = df_conc['c_B / uM'].to_numpy()
    cB0   = c_B[0]

    def model(t, k, cB_pss):
        return cB_pss - (cB_pss - cB0) * np.exp(-k * t)

    k_guess = 1.0 / times[-1] if times[-1] > 0 else 1.0
    popt, _ = curve_fit(model, times, c_B, p0=[k_guess, c_B[-1]], maxfev=10000)
    k_fit, cB_pss_fit = popt
    t_half = np.log(2) / k_fit

    # Optical power P = q * NA * h * c / λ, in mW
    wl_m       = params['Irradiation wavelength'] * 1e-9
    I_power_mW = (params['Photon flux'] * constants.Avogadro * constants.Planck
                  * constants.speed_of_light / wl_m * 1e3)
    p_half_mJ  = t_half * I_power_mW

    print_func(f"\nPhotoswitching kinetics ({nB} buildup, mono-exponential fit):")
    print_func(f"  k     = {k_fit:.4e} s⁻¹")
    print_func(f"  t1/2  = {t_half:.2f} s")
    print_func(f"  P     = {I_power_mW:.2f} mW")
    print_func(f"  p1/2  = {p_half_mJ:.2f} mJ")
    params['logger'].info(f"Kinetics fit: k={k_fit:.4e} s-1, t1/2={t_half:.3f} s, "
                          f"P={I_power_mW:.3f} mW, p1/2={p_half_mJ:.3f} mJ")

    pd.DataFrame({'time / s': times, 'cB exp / uM': c_B,
                  'cB fit / uM': model(times, k_fit, cB_pss_fit)}).to_csv(
        os.path.join(params['Results Folder'], f"{params['Basename']}_kinetics_fit.csv"),
        index=False)

    t_smooth = np.linspace(times[0], times[-1], 400)
    params['k']                  = k_fit
    params['t_half / s']         = t_half
    params['I_lambda / mW']      = I_power_mW
    params['p_half / mJ']        = p_half_mJ
    params['Kinetics times']     = times
    params['Kinetics cB']        = c_B
    params['Kinetics t_smooth']  = t_smooth
    params['Kinetics cB_smooth'] = model(t_smooth, k_fit, cB_pss_fit)
    params['Kinetics cB_pss']    = cB_pss_fit
    return params


def run_analysis(params):
    setup_logger(params)
    logger     = params['logger']
    print_func = params['Print function']
    logger.info('Starting Ladányi QY analysis')
    print_func(f"LED: {params.get('LED name', 'Custom')}")
    print_func(f"     central WL = {params['Irradiation wavelength']:.0f} nm,  "
               f"FWHM = {params.get('LED FWHM', 0.0):.0f} nm")
    print_func(f"Photon flux = {params['Photon flux']:.3e} mol s⁻¹, "
               f"V = {params['Volume']} mL")
    for key in ('LED name', 'Irradiation wavelength', 'LED FWHM', 'Photon flux', 'Volume',
                'Concentration WL min', 'Concentration WL max', 'Fit direction',
                'Thermal relaxation', 'Thermal k'):
        logger.info(f"{key}: {params.get(key)}")

    params['Abs coeff df'] = pd.read_csv(params['Abs coeff file'])

    calculate_concentration_profile(params)
    report_fractions(params)
    fit_photoswitching_kinetics(params)
    ladanyi_fit(params)
    return params


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plotting_spectra(params, ax):
    df        = params['Kinetics df']
    times     = df.columns[1:].astype(float).to_numpy()
    ymin, ymax = params['Plotting A_min'], params['Plotting A_max']
    xmin, xmax = params['Plotting wl_min'], params['Plotting wl_max']

    cmap = plt.get_cmap(params['Color palette'])
    norm = mcolors.Normalize(vmin=times[0], vmax=times[-1])
    for t, col in zip(times, df.columns[1:]):
        ax.plot(df.iloc[:, 0], df[col], color=cmap(norm(t)))

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    ax.figure.colorbar(sm, ax=ax, label='irradiation time / s')

    ax.set_xlabel('wavelength / nm')
    ax.set_ylabel('absorbance')
    if xmin != 0: ax.set_xlim(left=xmin)
    if xmax != 0: ax.set_xlim(right=xmax)
    if ymin != 0: ax.set_ylim(bottom=ymin)
    if ymax != 0: ax.set_ylim(top=ymax)


def plotting_conc_and_fit(params, ax):
    """Concentration profiles (scatter) with the Ladányi fit overlaid."""
    df_conc = params['Conc df']
    nA, nB  = params['Names']
    times   = df_conc['time / s'].to_numpy()
    t_s     = params['Ladanyi t_smooth']

    ax.scatter(times, df_conc['c_A / uM'], color='C0', s=12, zorder=2, label=nA)
    ax.scatter(times, df_conc['c_B / uM'], color='C1', s=12, zorder=2, label=nB)
    ax.plot(t_s, params['Ladanyi cA_smooth'], color='C0', lw=1.5, zorder=3)
    ax.plot(t_s, params['Ladanyi cB_smooth'], color='C1', lw=1.5, zorder=3)
    ax.plot([], [], color='black', lw=1.5, label='fit')

    ax.set_xlabel('irradiation time / s')
    ax.set_ylabel('concentration / µM')
    if params['Plotting t_min'] != 0: ax.set_xlim(left=params['Plotting t_min'])
    if params['Plotting t_max'] != 0: ax.set_xlim(right=params['Plotting t_max'])
    ax.legend()


def plotting_kinetics_fit(params, ax):
    """c_B(t) with mono-exponential fit, annotated with k, t1/2 and p1/2."""
    times   = params['Kinetics times']
    c_B     = params['Kinetics cB']
    cB_pss  = params['Kinetics cB_pss']
    t_half  = params['t_half / s']
    nB      = params['Names'][1]

    ax.scatter(times, c_B, color='C1', s=14, zorder=3, label=f'{nB} (data)')
    ax.plot(params['Kinetics t_smooth'], params['Kinetics cB_smooth'],
            color='black', lw=1.5, zorder=2, label='mono-exp. fit')

    ax.axhline(cB_pss, color='gray', ls=':', lw=1, zorder=1)
    if t_half <= times[-1]:
        cB_half = cB_pss - (cB_pss - c_B[0]) * 0.5
        ax.axvline(t_half, color='tomato', ls='--', lw=1, zorder=1)
        ax.scatter([t_half], [cB_half], color='tomato', s=30, zorder=4)

    txt = (f"k = {params['k']:.2e} s$^{{-1}}$\n"
           f"$t_{{1/2}}$ = {t_half:.1f} s\n"
           f"P = {params['I_lambda / mW']:.1f} mW\n"
           f"$p_{{1/2}}$ = {params['p_half / mJ']:.1f} mJ")
    ax.text(0.97, 0.05, txt, transform=ax.transAxes, ha='right', va='bottom',
            bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.85))

    ax.set_xlabel('irradiation time / s')
    ax.set_ylabel('concentration / µM')
    if params['Plotting t_min'] != 0: ax.set_xlim(left=params['Plotting t_min'])
    if params['Plotting t_max'] != 0: ax.set_xlim(right=params['Plotting t_max'])
    ax.legend(loc='center right')


def get_plot_functions(params):
    """Plotting functions to display, in order, for the current run."""
    if params['Has spectra']:
        return [plotting_spectra, plotting_conc_and_fit, plotting_kinetics_fit]
    return [plotting_conc_and_fit, plotting_kinetics_fit]


def save_plots(params):
    """Save each plot as an individual 8x6 SVG."""
    suffixes = {plotting_spectra: '_spectra', plotting_conc_and_fit: '_conc_fit',
                plotting_kinetics_fit: '_kinetics_fit'}
    with plt.rc_context(SAVE_RC):
        for func in get_plot_functions(params):
            fig, ax = plt.subplots(figsize=(8, 6))
            func(params, ax)
            fig.tight_layout()
            fig.savefig(os.path.join(params['Results Folder'],
                                     params['Basename'] + suffixes[func] + '.svg'),
                        bbox_inches='tight')
            plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────────────────────

def run_gui():
    root = tk.Tk()
    root.title("QY Calculation")
    root.resizable(True, True)
    root.geometry("1850x950")
    root.minsize(1100, 700)
    root.maxsize(3200, 2000)

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

    file_frame = ttk.LabelFrame(left, text="Files", padding=8)
    file_frame.grid(row=0, column=0, sticky="ew", pady=(0,8))
    file_frame.grid_columnconfigure(0, weight=1)

    var_kinetics  = tk.StringVar()
    var_abs_coeff = tk.StringVar()
    var_results   = tk.StringVar()

    file_row(file_frame, "Kinetics DataFrame (CSV):",      var_kinetics,  0, "Open kinetics CSV")
    file_row(file_frame, "Absorption coefficients (CSV):", var_abs_coeff, 1, "Open absorption coefficients CSV")

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

    var_led_select  = tk.StringVar(value='Custom (enter below)')
    var_led_center  = tk.StringVar(value='325')
    var_led_fwhm    = tk.StringVar(value='0')
    var_flux        = tk.StringVar(value='4.5e-9')
    var_volume      = tk.StringVar(value='3')
    var_wl_conc_min = tk.StringVar(value='330')
    var_wl_conc_max = tk.StringVar(value='380')
    var_conc        = tk.StringVar(value='40')
    var_fit_dir     = tk.StringVar(value='forward + backward')
    var_thermal     = tk.StringVar(value='none')
    var_k_thermal   = tk.StringVar(value='0')

    ttk.Label(param_frame, text="LED:").grid(row=0, column=0, sticky="w", pady=2, padx=(0,6))
    led_combo = ttk.Combobox(param_frame, textvariable=var_led_select,
                             values=list(LED_CATALOGUE.keys()), state="readonly", width=36)
    led_combo.grid(row=0, column=1, sticky="w", padx=4)

    param_entries = [
        ("LED central wavelength / nm",                 var_led_center),
        ("LED FWHM / nm  (0 = monochromatic)",         var_led_fwhm),
        ("Photon flux / mol s⁻¹",           var_flux),
        ("Volume / mL",                                 var_volume),
        ("Concentration WL min / nm",                   var_wl_conc_min),
        ("Concentration WL max / nm",                   var_wl_conc_max),
        ("Concentration / µM  (single-λ only)",  var_conc),
    ]
    entry_widgets = {}
    for i, (lbl, var) in enumerate(param_entries):
        ttk.Label(param_frame, text=lbl + ":").grid(row=i+1, column=0, sticky="w", pady=2, padx=(0,6))
        e = ttk.Entry(param_frame, textvariable=var, width=12)
        e.grid(row=i+1, column=1, sticky="w", padx=4)
        entry_widgets[lbl] = e
    led_center_entry = entry_widgets["LED central wavelength / nm"]
    led_fwhm_entry   = entry_widgets["LED FWHM / nm  (0 = monochromatic)"]

    def on_led_select(event=None):
        center, fwhm = LED_CATALOGUE.get(var_led_select.get(), (None, None))
        if center is not None:
            var_led_center.set(str(center))
            var_led_fwhm.set(str(fwhm))
        state = 'normal' if center is None else 'disabled'
        led_center_entry.config(state=state)
        led_fwhm_entry.config(state=state)
    led_combo.bind('<<ComboboxSelected>>', on_led_select)

    r = len(param_entries) + 1
    ttk.Label(param_frame, text="Photoreaction:").grid(row=r, column=0, sticky="w", pady=2, padx=(0,6))
    ttk.Combobox(param_frame, textvariable=var_fit_dir,
                 values=["forward + backward", "forward only"],
                 state="readonly", width=18).grid(row=r, column=1, sticky="w", padx=4)

    THERMAL_OPTIONS = {
        'none':                               'none',
        '2nd → 1st column (e.g. cis → trans)': 'B->A',
        '1st → 2nd column':              'A->B',
    }
    ttk.Label(param_frame, text="Thermal relaxation:").grid(row=r+1, column=0, sticky="w", pady=2, padx=(0,6))
    ttk.Combobox(param_frame, textvariable=var_thermal, values=list(THERMAL_OPTIONS),
                 state="readonly", width=30).grid(row=r+1, column=1, sticky="w", padx=4)
    ttk.Label(param_frame, text="Thermal k / s⁻¹  (= ln2 / t½):").grid(
        row=r+2, column=0, sticky="w", pady=2, padx=(0,6))
    ttk.Entry(param_frame, textvariable=var_k_thermal, width=12).grid(row=r+2, column=1, sticky="w", padx=4)

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
        ("Color palette", var_palette), ("A min", var_A_min), ("A max", var_A_max),
        ("λ min / nm", var_wl_min), ("λ max / nm", var_wl_max),
        ("t min / s", var_t_min), ("t max / s", var_t_max),
    ]
    for i, (lbl, var) in enumerate(plot_entries):
        ttk.Label(plot_frame, text=lbl + ":").grid(row=i, column=0, sticky="w", pady=2, padx=(0,6))
        ttk.Entry(plot_frame, textvariable=var, width=12).grid(row=i, column=1, sticky="w", padx=4)

    run_btn = ttk.Button(left, text="Run Analysis")
    run_btn.grid(row=3, column=0, sticky="ew", pady=(4,0), ipady=4)

    # ── Right panel ──────────────────────────────────────────────────────────
    #   [ Output text      ] [ top-right plot    ]
    #   [ bottom-left plot ] [ bottom-right plot ]
    right = ttk.Frame(root, padding=10)
    right.grid(row=0, column=1, sticky="nsew")
    right.grid_columnconfigure(0, weight=1, minsize=660)
    right.grid_columnconfigure(1, weight=1, minsize=660)
    right.grid_rowconfigure(1, weight=0)
    right.grid_rowconfigure(3, weight=0)
    right.grid_rowconfigure(4, weight=1)

    ttk.Label(right, text="Output", font=("TkDefaultFont", 10, "bold")).grid(row=0, column=0, sticky="w")
    ttk.Label(right, text="Plots",  font=("TkDefaultFont", 10, "bold")).grid(row=0, column=1, sticky="w")

    out_frame = ttk.Frame(right)
    out_frame.grid(row=1, column=0, sticky="new", padx=(0,10), pady=(4,10))
    out_frame.grid_rowconfigure(0, weight=1)
    out_frame.grid_columnconfigure(0, weight=1)
    output_box = tk.Text(out_frame, height=27, wrap="word", font=("TkFixedFont", 9))
    output_box.grid(row=0, column=0, sticky="nsew")
    out_scroll = ttk.Scrollbar(out_frame, orient="vertical", command=output_box.yview)
    out_scroll.grid(row=0, column=1, sticky="ns")
    output_box.config(yscrollcommand=out_scroll.set)

    slot_topright = ttk.Frame(right); slot_topright.grid(row=1, column=1, sticky="n", pady=(4,10))
    slot_botleft  = ttk.Frame(right); slot_botleft.grid(row=3, column=0, sticky="n", padx=(0,10))
    slot_botright = ttk.Frame(right); slot_botright.grid(row=3, column=1, sticky="n")
    plot_slots = [slot_topright, slot_botleft, slot_botright]

    PLOT_FIGSIZE = (6.5, 4.1)
    plot_canvases = []

    def _draw_placeholder(slot):
        f = plt.Figure(figsize=PLOT_FIGSIZE, dpi=100)
        f.add_subplot(111).axis('off')
        fc = FigureCanvasTkAgg(f, master=slot)
        fc.draw()
        fc.get_tk_widget().pack()
        return fc

    for _slot in plot_slots:
        plot_canvases.append(_draw_placeholder(_slot))

    def _float(s):
        return float(str(s).replace(',', '.'))

    # ── Run callback ─────────────────────────────────────────────────────────
    def run():
        output_box.delete("1.0", tk.END)
        for c in plot_canvases:
            c.get_tk_widget().destroy()
        plot_canvases.clear()

        def printer(msg):
            output_box.insert(tk.END, msg + "\n")
            output_box.see(tk.END)
            root.update_idletasks()

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

        df_kinetics = pd.read_csv(kin_file).dropna(axis=1)
        has_spectra = df_kinetics.shape[0] > 1

        try:
            params = {
                'Print function':         printer,
                'Kinetics df':            df_kinetics,
                'Has spectra':            has_spectra,
                'Abs coeff file':         coeff_file,
                'Results Folder':         results_dir,
                'Basename':               os.path.splitext(os.path.basename(kin_file))[0],
                'LED name':               var_led_select.get(),
                'Irradiation wavelength': _float(var_led_center.get()),
                'LED FWHM':               _float(var_led_fwhm.get()),
                'Photon flux':            _float(var_flux.get()),
                'Volume':                 _float(var_volume.get()),
                'Concentration WL min':   _float(var_wl_conc_min.get()),
                'Concentration WL max':   _float(var_wl_conc_max.get()),
                'Concentration':          _float(var_conc.get()) * 1e-6,
                'Fit direction':          ('forward_only' if var_fit_dir.get() == 'forward only'
                                           else 'both'),
                'Thermal relaxation':     THERMAL_OPTIONS[var_thermal.get()],
                'Thermal k':              _float(var_k_thermal.get()),
                'Color palette':          var_palette.get(),
                'Plotting A_min':         _float(var_A_min.get()),
                'Plotting A_max':         _float(var_A_max.get()),
                'Plotting wl_min':        _float(var_wl_min.get()),
                'Plotting wl_max':        _float(var_wl_max.get()),
                'Plotting t_min':         _float(var_t_min.get()),
                'Plotting t_max':         _float(var_t_max.get()),
            }
        except ValueError as e:
            messagebox.showerror("Input error", f"Invalid number: {e}")
            return

        printer(f"Kinetics df: {df_kinetics.shape[0]} row(s) × {df_kinetics.shape[1]} columns "
                f"({'spectra series' if has_spectra else 'single-wavelength kinetics'})")

        try:
            run_analysis(params)
            save_plots(params)
            for i, plot_func in enumerate(get_plot_functions(params)):
                f  = plt.Figure(figsize=PLOT_FIGSIZE, dpi=100)
                ax = f.add_subplot(111)
                plot_func(params, ax)
                f.tight_layout()
                fc = FigureCanvasTkAgg(f, master=plot_slots[i])
                fc.draw()
                fc.get_tk_widget().pack()
                plot_canvases.append(fc)
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
    run_gui()

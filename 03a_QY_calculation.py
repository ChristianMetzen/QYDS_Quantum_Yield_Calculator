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
    from scipy.integrate import trapezoid
    norm = trapezoid(S[mask], wl_eps[mask])
    return float(trapezoid(S[mask] * eps_values[mask], wl_eps[mask]) / norm)

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

    # c_A always = column 1 species, c_B always = column 2 species.
    # No swap — this keeps c_A/c_B consistent with e_A/e_B in the ODE.
    names = [df_abs_coeff.columns[1], df_abs_coeff.columns[2]]
    c_A, c_B = x_A * c * 1e6, x_B * c * 1e6

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

    print_func(f"Fraction {name_B} (cis) at t=0 : {frac_B_initial:.1f} %")
    print_func(f"Fraction {name_B} (cis) at PSS : {frac_B_pss:.1f} %")
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

    _wl_eps_arr = df_abs_coeff.iloc[:, 0].to_numpy(dtype=float)
    _eps_A_arr  = df_abs_coeff.iloc[:, 1].to_numpy(dtype=float)
    _eps_B_arr  = df_abs_coeff.iloc[:, 2].to_numpy(dtype=float)
    _led_center = params['Irradiation wavelength']
    _led_fwhm   = params.get('LED FWHM', 0.0)
    if _led_fwhm > 0:
        e_A = effective_epsilon(_wl_eps_arr, _eps_A_arr, _led_center, _led_fwhm)
        e_B = effective_epsilon(_wl_eps_arr, _eps_B_arr, _led_center, _led_fwhm)
    else:
        wl_idx = np.abs(_wl_eps_arr - _led_center).argmin()
        e_A    = float(_eps_A_arr[wl_idx])
        e_B    = float(_eps_B_arr[wl_idx])

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

    return params


def fit_photoswitching_kinetics(params):
    """Fit a mono-exponential buildup curve to extract k, t1/2 and p1/2 (Note 7).

    Model:  c_B(t) = c_B_PSS - (c_B_PSS - c_B0) * exp(-k * t)
    t1/2  = ln(2) / k                       (isomerization half-time)
    p1/2  = t1/2 * I(lambda)                (half-power, light-source independent)
    I(lambda) is the total incident optical power at the irradiation wavelength,
    obtained directly from the calibration (voltage * correction factor), in mW.
    """
    print_func = params['Print function']
    logger     = params['logger']
    df_conc    = params['Conc df']
    names      = params['Names']

    times = df_conc['time / s'].to_numpy()
    c_B   = df_conc['c_B / uM'].to_numpy()
    cB0   = c_B[0]

    def model(t, k, cB_pss):
        return cB_pss - (cB_pss - cB0) * np.exp(-k * t)

    k_guess  = 1.0 / times[-1] if times[-1] > 0 else 1.0
    p0       = [k_guess, c_B[-1]]
    popt, _  = curve_fit(model, times, c_B, p0=p0, maxfev=10000)
    k_fit, cB_pss_fit = popt
    t_half   = np.log(2) / k_fit

    # I(lambda): total incident optical power, independent of volume/Avogadro
    # normalisation — directly from the calibration (voltage * correction factor).
    wl         = params['Irradiation wavelength']
    cf         = get_correction_factor(wl, params['Channel'])
    I_power_mW = params['Voltage after ref'] * cf
    p_half_mJ  = t_half * I_power_mW   # s * mW = mJ

    print_func(f"\nPhotoswitching kinetics ({names[1]} buildup, mono-exponential fit):")
    print_func(f"  k     = {k_fit:.4e} s\u207b\u00b9")
    print_func(f"  t1/2  = {t_half:.2f} s")
    print_func(f"  I(\u03bb)  = {I_power_mW:.2f} mW")
    print_func(f"  p1/2  = {p_half_mJ:.2f} mJ")
    logger.info(f"Kinetics fit: k={k_fit:.4e} s-1, t1/2={t_half:.3f} s, "
                f"I(lambda)={I_power_mW:.3f} mW, p1/2={p_half_mJ:.3f} mJ")

    cB_fit_curve = model(times, k_fit, cB_pss_fit)
    pd.DataFrame({'time / s': times, 'cB exp / uM': c_B, 'cB fit / uM': cB_fit_curve}).to_csv(
        os.path.join(params['Results Folder'], f"{params['Basename']}_kinetics_fit.csv"), index=False)

    # Smooth curve for plotting
    t_smooth  = np.linspace(times[0], times[-1], 400)
    cB_smooth = model(t_smooth, k_fit, cB_pss_fit)

    params['k']             = k_fit
    params['t_half / s']    = t_half
    params['I_lambda / mW'] = I_power_mW
    params['p_half / mJ']   = p_half_mJ
    params['Kinetics times']     = times
    params['Kinetics cB']        = c_B
    params['Kinetics t_smooth']  = t_smooth
    params['Kinetics cB_smooth'] = cB_smooth
    params['Kinetics cB_pss']    = cB_pss_fit
    return params


def run_analysis(params):
    setup_logger(params)
    logger     = params['logger']
    print_func = params['Print function']
    logger.info('Starting Ladányi QY analysis')
    _led_name   = params.get('LED name', 'Custom')
    _led_center = params['Irradiation wavelength']
    _led_fwhm   = params.get('LED FWHM', 0.0)
    params['LED central wavelength'] = _led_center
    params['LED FWHM'] = _led_fwhm
    print_func(f'LED: {_led_name}')
    print_func(f'     central WL = {_led_center:.0f} nm,  FWHM = {_led_fwhm:.0f} nm')
    logger.info(f'LED: {_led_name}, central WL={_led_center:.0f} nm, FWHM={_led_fwhm:.0f} nm')
    logger.info(f"Irradiation: {params['Irradiation wavelength']} nm, "
                f"channel: {params['Channel']}, voltage: {params['Voltage after ref']} V")
    logger.info(f"Concentration WL range: {params['Concentration WL min']}-{params['Concentration WL max']} nm")
    logger.info(f"Volume: {params['Volume']} mL, concentration: {params['Concentration']*1e6:.1f} µM")

    calculate_photon_flux(params)

    df_abs_coeff = pd.read_csv(params['Abs coeff file'])
    params['Abs coeff df'] = df_abs_coeff

    calculate_concentration_profile(params)
    report_cis_fraction(params)
    fit_photoswitching_kinetics(params)
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

    _wl_eps_arr = df_abs_coeff.iloc[:, 0].to_numpy(dtype=float)
    _eps_A_arr  = df_abs_coeff.iloc[:, 1].to_numpy(dtype=float)
    _eps_B_arr  = df_abs_coeff.iloc[:, 2].to_numpy(dtype=float)
    _led_center = params['Irradiation wavelength']
    _led_fwhm   = params.get('LED FWHM', 0.0)
    if _led_fwhm > 0:
        e_A = effective_epsilon(_wl_eps_arr, _eps_A_arr, _led_center, _led_fwhm)
        e_B = effective_epsilon(_wl_eps_arr, _eps_B_arr, _led_center, _led_fwhm)
    else:
        wl_idx = np.abs(_wl_eps_arr - _led_center).argmin()
        e_A    = float(_eps_A_arr[wl_idx])
        e_B    = float(_eps_B_arr[wl_idx])

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


def plotting_kinetics_fit(params, ax):
    """Plot the photoswitching kinetics: c_B(t) data with mono-exponential fit,
    annotated with k, t1/2 and p1/2."""
    times     = params['Kinetics times']
    c_B       = params['Kinetics cB']
    t_smooth  = params['Kinetics t_smooth']
    cB_smooth = params['Kinetics cB_smooth']
    cB_pss    = params['Kinetics cB_pss']
    t_half    = params['t_half / s']
    names     = params['Names']
    t_min     = params['Plotting t_min']
    t_max     = params['Plotting t_max']

    ax.scatter(times, c_B, color='C1', s=14, zorder=3, label=f'{names[1]} (data)')
    ax.plot(t_smooth, cB_smooth, color='black', lw=1.5, zorder=2, label='mono-exp. fit')

    # PSS asymptote and t1/2 markers
    ax.axhline(cB_pss, color='gray', ls=':', lw=1, zorder=1)
    if t_half <= times[-1]:
        cB_half = cB_pss - (cB_pss - c_B[0]) * 0.5
        ax.axvline(t_half, color='tomato', ls='--', lw=1, zorder=1)
        ax.scatter([t_half], [cB_half], color='tomato', s=30, zorder=4)

    txt = (f"k = {params['k']:.2e} s$^{{-1}}$\n"
           f"$t_{{1/2}}$ = {t_half:.1f} s\n"
           f"I($\\lambda$) = {params['I_lambda / mW']:.1f} mW\n"
           f"$p_{{1/2}}$ = {params['p_half / mJ']:.1f} mJ")
    ax.text(0.97, 0.05, txt, transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.85))

    ax.set_xlabel('irradiation time / s')
    ax.set_ylabel('concentration / µM')
    ax.set_title('Photoswitching kinetics')
    if t_min != 0: ax.set_xlim(left=t_min)
    if t_max != 0: ax.set_xlim(right=t_max)
    ax.legend(loc='center right')


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
    n_plots = 3 if has_spectra else 2
    figure.set_size_inches(8 * n_plots, 6)   # each panel ~8x6 in the saved SVG
    try:
        figure.set_constrained_layout(True)
    except Exception:
        pass
    axs = figure.subplots(1, n_plots)

    if has_spectra:
        plotting_spectra(params, axs[0])
        plotting_conc_and_fit(params, axs[1])
        plotting_kinetics_fit(params, axs[2])
        save_individual_plot(plotting_spectra,      params, '_spectra')
        save_individual_plot(plotting_conc_and_fit, params, '_conc_fit')
        save_individual_plot(plotting_kinetics_fit, params, '_kinetics_fit')
    else:
        plotting_conc_and_fit(params, axs[0])
        plotting_kinetics_fit(params, axs[1])
        save_individual_plot(plotting_conc_and_fit, params, '_conc_fit')
        save_individual_plot(plotting_kinetics_fit, params, '_kinetics_fit')

    figure.savefig(os.path.join(params['Results Folder'], params['Basename'] + '_results.svg'),
                   bbox_inches='tight')


def get_plot_functions(params):
    """Return the plotting functions to display, in order, for the current run."""
    if params['Has spectra']:
        return [plotting_spectra, plotting_conc_and_fit, plotting_kinetics_fit]
    return [plotting_conc_and_fit, plotting_kinetics_fit]


# ─────────────────────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────────────────────

def run_gui():
    root = tk.Tk()
    root.title("QY Calculation")
    root.resizable(True, True)
    root.geometry("1850x950")     # fits the 2x2 layout with equal-width plots
    root.minsize(1100, 700)
    root.maxsize(3200, 2000)      # generous cap for 4K

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

    var_led_select   = tk.StringVar(value='Custom (enter below)')
    var_led_center   = tk.StringVar(value='325')
    var_led_fwhm     = tk.StringVar(value='0')
    var_wl_conc_min  = tk.StringVar(value='330')
    var_wl_conc_max  = tk.StringVar(value='380')
    var_voltage      = tk.StringVar(value='7.465')
    var_volume       = tk.StringVar(value='3')
    var_conc         = tk.StringVar(value='40')
    var_channel      = tk.StringVar(value='low')

    # Row 0: LED dropdown
    ttk.Label(param_frame, text="LED:").grid(row=0, column=0, sticky="w", pady=2, padx=(0,6))
    led_combo = ttk.Combobox(param_frame, textvariable=var_led_select,
                             values=list(LED_CATALOGUE.keys()),
                             state="readonly", width=36)
    led_combo.grid(row=0, column=1, sticky="w", padx=4)

    def on_led_select(event=None):
        name = var_led_select.get()
        center, fwhm = LED_CATALOGUE.get(name, (None, None))
        if center is not None:
            var_led_center.set(str(center))
            var_led_fwhm.set(str(fwhm))
        state = 'normal' if center is None else 'disabled'
        led_center_entry.config(state=state)
        led_fwhm_entry.config(state=state)
    led_combo.bind('<<ComboboxSelected>>', on_led_select)

    param_entries = [
        ("LED central wavelength / nm",          var_led_center),
        ("LED FWHM / nm  (0 = monochromatic)",  var_led_fwhm),
        ("Concentration WL min / nm",            var_wl_conc_min),
        ("Concentration WL max / nm",            var_wl_conc_max),
        ("Voltage after reference / V",          var_voltage),
        ("Volume / mL",                          var_volume),
        ("Concentration / µM",                   var_conc),
    ]
    entry_widgets = {}
    for i, (lbl, var) in enumerate(param_entries):
        ttk.Label(param_frame, text=lbl + ":").grid(row=i+1, column=0, sticky="w", pady=2, padx=(0,6))
        e = ttk.Entry(param_frame, textvariable=var, width=12)
        e.grid(row=i+1, column=1, sticky="w", padx=4)
        entry_widgets[lbl] = e
    led_center_entry = entry_widgets["LED central wavelength / nm"]
    led_fwhm_entry   = entry_widgets["LED FWHM / nm  (0 = monochromatic)"]

    ttk.Label(param_frame, text="Channel:").grid(row=9, column=0, sticky="w", pady=2, padx=(0,6))
    ttk.Combobox(param_frame, textvariable=var_channel, values=["low","high"],
                 state="readonly", width=10).grid(row=9, column=1, sticky="w", padx=4)

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
    # Layout (2 columns):
    #   [ Output text     ] [ top-right plot  (spectra) ]
    #   [ bottom-left plot ] [ bottom-right plot         ]
    right = ttk.Frame(root, padding=10)
    right.grid(row=0, column=1, sticky="nsew")
    right.grid_columnconfigure(0, weight=1, minsize=660)
    right.grid_columnconfigure(1, weight=1, minsize=660)
    right.grid_rowconfigure(1, weight=0)   # top row (output + first plot): natural height
    right.grid_rowconfigure(3, weight=0)   # bottom row (two plots): natural height
    right.grid_rowconfigure(4, weight=1)   # spacer absorbs any extra vertical space

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

    # Fixed slots for the plots. Plots are assigned in order:
    #   plot 0 -> top-right (next to output), plot 1 -> bottom-left, plot 2 -> bottom-right
    slot_topright = ttk.Frame(right)
    slot_topright.grid(row=1, column=1, sticky="n", pady=(4,10))
    slot_botleft  = ttk.Frame(right)
    slot_botleft.grid(row=3, column=0, sticky="n", padx=(0,10))
    slot_botright = ttk.Frame(right)
    slot_botright.grid(row=3, column=1, sticky="n")
    plot_slots = [slot_topright, slot_botleft, slot_botright]

    # All plots share the same box size so the columns line up.
    PLOT_FIGSIZE = (6.5, 4.1)   # inches; dpi=100 -> 650 x 410 px

    # Holds the live FigureCanvasTkAgg widgets so we can clear them between runs
    plot_canvases = []

    def _draw_placeholder(slot):
        """Blank white box that reserves the plot's space before the first run,
        so the output box doesn't resize when plots appear."""
        f = plt.Figure(figsize=PLOT_FIGSIZE, dpi=100)
        f.patch.set_facecolor('white')
        ax = f.add_subplot(111)
        ax.axis('off')
        fc = FigureCanvasTkAgg(f, master=slot)
        fc.draw()
        fc.get_tk_widget().pack()
        return fc

    # Reserve plot space immediately on launch
    for _slot in plot_slots:
        plot_canvases.append(_draw_placeholder(_slot))

    def _float(s):
        """Parse float accepting both '.' and ',' as decimal separator."""
        return float(str(s).replace(',', '.'))

    # ── Run callback ─────────────────────────────────────────────────────────
    def run():
        output_box.delete("1.0", tk.END)
        # Clear any previously drawn plots
        for c in plot_canvases:
            c.get_tk_widget().destroy()
        plot_canvases.clear()

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
            'Irradiation wavelength':  _float(var_led_center.get()),
            'LED name':                var_led_select.get(),
            'LED central wavelength':  _float(var_led_center.get()),
            'LED FWHM':                _float(var_led_fwhm.get()),
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
            # Save the combined SVG + individual SVGs to disk (unchanged)
            save_fig = plt.Figure()
            plot_results(params, save_fig)
            plt.close(save_fig)

            # Render each plot into its fixed slot:
            #   plot 0 (spectra) -> top-right, next to the output
            #   plots 1,2 -> bottom row
            plot_funcs = get_plot_functions(params)
            for i, plot_func in enumerate(plot_funcs):
                f  = plt.Figure(figsize=PLOT_FIGSIZE, dpi=100)
                ax = f.add_subplot(111)
                plot_func(params, ax)
                f.tight_layout()
                slot = plot_slots[i]
                fc = FigureCanvasTkAgg(f, master=slot)
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
    import tkinter as _tk_check
    try:
        _tk_check.Tk().destroy()
    except Exception as _e:
        print(f"Warning: could not initialise Tk ({_e}). Try restarting the Python kernel.")
    run_gui()
